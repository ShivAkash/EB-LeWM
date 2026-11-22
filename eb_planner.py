import inspect
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import stable_worldmodel as swm


def _remap_legacy_vit_keys(sd: dict) -> dict:
    from transformers import ViTConfig, ViTModel
    test_keys = set(
        ViTModel(
            ViTConfig(hidden_size=24, num_hidden_layers=1, num_attention_heads=1),
            add_pooling_layer=False,
        ).state_dict().keys()
    )
    expects_modern = any(k.startswith("layers.0.attention.q_proj") for k in test_keys)
    if not expects_modern:
        return sd

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("encoder.encoder.layer."):
            k = k.replace("encoder.encoder.layer.", "encoder.layers.")
            k = k.replace(".attention.attention.query.", ".attention.q_proj.")
            k = k.replace(".attention.attention.key.", ".attention.k_proj.")
            k = k.replace(".attention.attention.value.", ".attention.v_proj.")
            k = k.replace(".attention.output.dense.", ".attention.o_proj.")
            k = k.replace(".intermediate.dense.", ".mlp.fc1.")
            k = k.replace(".output.dense.", ".mlp.fc2.")
        new_sd[k] = v
    return new_sd


def load_lewm_model(policy_arg: str, device: str = "cuda"):
    cache_dir = Path(swm.data.utils.get_cache_dir())
    cand = Path(policy_arg).expanduser()
    if not cand.exists():
        cand = cache_dir / policy_arg

    ckpt_dir = cand if cand.is_dir() else cand.parent
    cfg_file = ckpt_dir / "config.json"
    weights_file = cand if cand.is_file() else (ckpt_dir / "weights.pt")

    if cfg_file.exists() and weights_file.exists():
        from transformers import ViTConfig, ViTModel
        from jepa import JEPA
        from module import ARPredictor, Embedder, MLP

        with open(cfg_file, "r") as f:
            cfg = json.load(f)

        sd = torch.load(weights_file, map_location="cpu", weights_only=False)
        if isinstance(sd, nn.Module):
            model = sd
        else:
            if "state_dict" in sd:
                sd = sd["state_dict"]
            sd = {k.removeprefix("model."): v for k, v in sd.items()}
            sd = _remap_legacy_vit_keys(sd)

            emb_dim = cfg.get("hidden_size", 192)
            vit_cfg = ViTConfig(
                hidden_size=emb_dim,
                num_hidden_layers=cfg.get("num_hidden_layers", 12),
                num_attention_heads=cfg.get("num_attention_heads", 3),
                intermediate_size=cfg.get("intermediate_size", emb_dim * 4),
                image_size=cfg.get("image_size", 224),
                patch_size=cfg.get("patch_size", 14),
            )
            encoder = ViTModel(vit_cfg, add_pooling_layer=False)

            act_w0 = sd["action_encoder.patch_embed.weight"]
            in_chans, action_dim, patch_sz = act_w0.shape[0], act_w0.shape[1], act_w0.shape[2]
            act_mlp_h = sd["action_encoder.embed.0.weight"].shape[0]
            emb_params = inspect.signature(Embedder.__init__).parameters
            if "in_chans" in emb_params:
                action_encoder = Embedder(
                    in_chans=in_chans, emb_dim=emb_dim, patch_size=patch_sz, mlp_scale=act_mlp_h // emb_dim
                )
            else:
                action_encoder = Embedder(
                    input_dim=action_dim, emb_dim=emb_dim, mlp_scale=act_mlp_h // emb_dim
                )

            pos_key = "predictor.pos_embedding" if "predictor.pos_embedding" in sd else "predictor.pos_emb"
            num_frames = sd[pos_key].shape[1]
            layer_keys = [k for k in sd if k.startswith("predictor.transformer.layers.")]
            depth = max(int(k.split(".")[3]) for k in layer_keys) + 1
            qkv_dim = sd["predictor.transformer.layers.0.attn.to_qkv.weight"].shape[0] // 3
            mlp_dim = sd["predictor.transformer.layers.0.mlp.net.1.weight"].shape[0]
            heads = cfg.get("predictor_heads", 16 if qkv_dim >= 512 else 4)
            dim_head = qkv_dim // heads

            pred_params = inspect.signature(ARPredictor.__init__).parameters
            pred_kwargs = dict(
                input_dim=emb_dim,
                hidden_dim=emb_dim,
                output_dim=emb_dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
            )
            if "num_frames" in pred_params:
                pred_kwargs["num_frames"] = num_frames
            elif "max_seq_len" in pred_params:
                pred_kwargs["max_seq_len"] = num_frames
            predictor = ARPredictor(**pred_kwargs)

            proj_h = sd["projector.net.0.weight"].shape[0]
            mlp_params = inspect.signature(MLP.__init__).parameters
            if "norm_fn" in mlp_params:
                projector = MLP(emb_dim, proj_h, emb_dim, norm_fn=nn.BatchNorm1d)
                pred_proj = MLP(emb_dim, proj_h, emb_dim, norm_fn=nn.BatchNorm1d)
            else:
                projector = MLP(emb_dim, proj_h, emb_dim)
                pred_proj = MLP(emb_dim, proj_h, emb_dim)

            model = JEPA(
                encoder=encoder,
                predictor=predictor,
                action_encoder=action_encoder,
                projector=projector,
                pred_proj=pred_proj,
            )
            model.load_state_dict(sd, strict=True)
    else:
        model = swm.wm.utils.load_pretrained(policy_arg)

    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


class EBHeads(nn.Module):
    """
    Lightweight heads trained on the 100% FROZEN LeWM backbone (Train Split: ep_id >= 2000):
      1. temp_dist_net     : Predicts normalized temporal distance d(z_t, z_goal) in [0, 1].
      2. bottleneck_jumper : Predicts variable-step event bottleneck z_{t*} between z_t and z_goal.
      3. fixed_k_jumper    : True Fixed-k ablation jumper trained to predict z_{t+k}.
      4. term_gate         : Learned termination classifier beta(z_t, z_sub) -> [0, 1] used at runtime.
      5. epistemic_lora    : Retained in state_dict for backward compatibility with existing checkpoints,
                             but disabled at planning time unless mode == 'ebjewm_with_lora'.
    """
    def __init__(self, dim: int = 192, num_heads: int = 4, rank: int = 16, prior_scale: float = 0.6):
        super().__init__()
        self.temp_dist_net = nn.Sequential(
            nn.Linear(dim * 2, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.bottleneck_jumper = nn.Sequential(
            nn.Linear(dim * 2, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, dim),
        )

        self.fixed_k_jumper = nn.Sequential(
            nn.Linear(dim * 2, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, dim),
        )

        self.term_gate = nn.Sequential(
            nn.Linear(dim * 2, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.prior_scale = prior_scale
        self.lora_A = nn.Parameter(torch.randn(num_heads, dim, rank) * 0.15)
        self.lora_bias = nn.Parameter(torch.zeros(1, num_heads, rank))
        self.lora_B = nn.Parameter(torch.zeros(num_heads, rank, dim))

        pA = F.normalize(torch.randn(num_heads, dim, rank), dim=1) * 2.0
        p_bias = torch.randn(1, num_heads, rank) * 0.25
        pB = F.normalize(torch.randn(num_heads, rank, dim), dim=2)
        self.register_buffer("prior_A", pA)
        self.register_buffer("prior_bias", p_bias)
        self.register_buffer("prior_B", pB)

        self.register_buffer("tau_far", torch.tensor(0.35))
        self.register_buffer("tau_term", torch.tensor(0.50))
        self.register_buffer("unc_scale", torch.tensor(0.0))

    def predict_temp_dist(self, z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return self.temp_dist_net(torch.cat([z, z_goal], dim=-1)).squeeze(-1)

    def predict_bottleneck(self, z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return self.bottleneck_jumper(torch.cat([z, z_goal], dim=-1))

    def predict_fixed_k(self, z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return self.fixed_k_jumper(torch.cat([z, z_goal], dim=-1))

    def predict_termination(self, z: torch.Tensor, z_subgoal: torch.Tensor) -> torch.Tensor:
        return self.term_gate(torch.cat([z, z_subgoal], dim=-1)).squeeze(-1)

    def epistemic_variance(self, z: torch.Tensor) -> torch.Tensor:
        orig_shape = z.shape[:-1]
        x = F.normalize(z.detach().reshape(-1, z.size(-1)), p=2, dim=-1)
        lora_h = F.silu(torch.einsum("bi,hir->bhr", x, self.lora_A) + self.lora_bias)
        lora_d = torch.einsum("bhr,hro->bho", lora_h, self.lora_B)
        prior_h = F.silu(torch.einsum("bi,hir->bhr", x, self.prior_A) + self.prior_bias)
        prior_d = self.prior_scale * torch.einsum("bhr,hro->bho", prior_h, self.prior_B)
        return ((lora_d - prior_d) ** 2).mean(dim=(1, 2)).reshape(*orig_shape)


def attach_custom_criterion(jepa_model, eb_heads: EBHeads, mode: str = "ebjewm"):
    """
    Hooks into the frozen JEPA.criterion() for paired planner evaluation.
    Unbundled Termination Modes (for Seeds 300-309):
      - 'ebjewm'          : Full method (Plan-time gate + Runtime flush + Rollout beta bonus; no LoRA).
      - 'ebjewm_noflush'  : Plan-time gate + Rollout beta bonus, WITHOUT runtime mid-plan flush.
      - 'ebjewm_nobonus'  : Plan-time gate + Runtime flush, WITHOUT rollout beta bonus / sub-goal min.
      - 'ebjewm_costonly' : Rollout beta bonus only, WITHOUT plan-time gate or runtime flush.
      - 'ebjewm_noterm'   : Neither gate, nor flush, nor rollout beta bonus.
    """
    jepa_model.eb_heads = eb_heads.to(next(jepa_model.parameters()).device)
    jepa_model.eb_heads.eval()
    jepa_model.solver_mode = mode
    orig_criterion = jepa_model.criterion

    def custom_criterion(info_dict: dict):
        if mode == "vanilla":
            return orig_criterion(info_dict)

        full_emb = info_dict["predicted_emb"]  # (B, S, K_total, D)
        if full_emb.shape[2] > 1:
            z_start = full_emb[:, :, 0, :]       # (B, S, D) observed current frame
            future_emb = full_emb[:, :, 1:, :]   # (B, S, K_future, D) strictly predicted future steps
        else:
            z_start = info_dict["emb"][:, :, -1, :] if ("emb" in info_dict and info_dict["emb"].ndim == 4) else full_emb[:, :, 0, :]
            future_emb = full_emb

        B, S, K, D = future_emb.shape

        goal_emb = info_dict["goal_emb"]
        if goal_emb.ndim == 3:
            z_goal_bs = goal_emb[:, -1:, :].expand(B, S, D)
        else:
            z_goal_bs = goal_emb.reshape(B, 1, D).expand(B, S, D)
        z_goal_bsk = z_goal_bs.unsqueeze(2).expand(B, S, K, D)

        # --- Baselines: Terminal & Min-Shaped Temporal-Distance Cost ---
        td_goal_steps = jepa_model.eb_heads.predict_temp_dist(future_emb, z_goal_bsk)  # (B, S, K)
        if mode == "tempdist":
            return td_goal_steps[:, :, -1]
        if mode == "tempdist_min":
            return 0.5 * td_goal_steps[:, :, -1] + 0.5 * td_goal_steps.min(dim=-1).values

        # --- Sub-Goal Generation ---
        heads = jepa_model.eb_heads
        dist_to_goal = heads.predict_temp_dist(z_start, z_goal_bs)  # (B, S)

        if mode == "ebjewm_fixedk":
            z_sub = heads.predict_fixed_k(z_start, z_goal_bs)
        else:
            z_sub = heads.predict_bottleneck(z_start, z_goal_bs)

        # --- Component A: Plan-Time Termination Handoff Gate ---
        term_prob = heads.predict_termination(z_start, z_sub)  # (B, S) in [0, 1]
        if mode == "ebjewm_nofallback":
            use_subgoal = (term_prob < heads.tau_term).float()
        elif mode in ("ebjewm_noterm", "ebjewm_costonly"):
            # Disables plan-time gate: routes to sub-goal based purely on distance to final goal
            use_subgoal = (dist_to_goal > heads.tau_far).float()
        else:
            # Active in ebjewm, ebjewm_noflush, ebjewm_nobonus, ebjewm_fixedk
            use_subgoal = ((dist_to_goal > heads.tau_far) & (term_prob < heads.tau_term)).float()

        z_sub_bsk = z_sub.unsqueeze(2).expand(B, S, K, D)
        td_sub_steps = heads.predict_temp_dist(future_emb, z_sub_bsk)  # (B, S, K)

        # --- Component B: Rollout Cost Shaping (beta bonus & min-over-steps) ---
        goal_min_cost = td_goal_steps.min(dim=-1).values
        goal_cost = 0.5 * td_goal_steps[:, :, -1] + 0.5 * goal_min_cost

        if mode == "ebjewm_noterm":
            goal_cost = td_goal_steps[:, :, -1]
            sub_cost = td_sub_steps[:, :, -1]
        elif mode == "ebjewm_nobonus":
            # Flush + Gate active, but NO rollout beta bonus and NO min-over-steps on the sub-goal
            sub_cost = td_sub_steps[:, :, -1]
        else:
            # Active in ebjewm, ebjewm_noflush, ebjewm_costonly, ebjewm_fixedk
            term_weights_sub = heads.predict_termination(future_emb, z_sub_bsk)  # (B, S, K)
            sub_min_cost = td_sub_steps.min(dim=-1).values
            sub_cost = 0.5 * td_sub_steps[:, :, -1] + 0.5 * sub_min_cost - 0.1 * term_weights_sub.max(dim=-1).values

        traj_cost = (1.0 - use_subgoal) * goal_cost + use_subgoal * (0.5 * goal_cost + 0.5 * sub_cost)

        if mode == "ebjewm_with_lora":
            ep_var = heads.epistemic_variance(future_emb).mean(dim=-1)
            return traj_cost + heads.unc_scale * ep_var

        # By default, drop the inert Epistemic LoRA term
        return traj_cost

    jepa_model.criterion = custom_criterion
    return jepa_model


class EventDrivenPolicyWrapper:
    """
    Event-driven replanning for swm.policy.WorldModelPolicy (swm >= 0.1).
    Modes in PASSIVE_MODES never trigger mid-plan flushes (though they still log ticks and plans).
    """

    PASSIVE_MODES = (
        "vanilla",
        "tempdist",
        "tempdist_min",
        "ebjewm_noterm",
        "ebjewm_noflush",
        "ebjewm_costonly",
    )

    def __init__(self, base_policy, jepa_model, mode: str = "ebjewm"):
        self.__dict__["base_policy"] = base_policy
        self.__dict__["jepa_model"] = jepa_model
        self.__dict__["mode"] = mode
        self.__dict__["stats"] = {"ticks": 0, "plans": 0, "subgoal_plans": 0, "flushes": 0}
        self.__dict__["_z_sub"] = None
        self.__dict__["_active"] = None

    def __getattr__(self, name):
        return getattr(self.__dict__["base_policy"], name)

    @torch.no_grad()
    def get_action(self, info_dict, *args, **kwargs):
        n = len(self.base_policy._action_buffer)
        reset = info_dict.get("_needs_flush")
        reset = np.zeros(n, dtype=bool) if reset is None else np.asarray(reset, dtype=bool).reshape(n)
        buf_len = np.array([len(b) for b in self.base_policy._action_buffer])

        if self.mode in self.PASSIVE_MODES:
            prepared = self.base_policy._prepare_info(info_dict)
            term = prepared.get("terminated")
            dead = np.zeros(n, dtype=bool) if term is None else np.asarray(term, dtype=bool).reshape(n)
            replanning = ((buf_len == 0) | reset) & (~dead)
            self.stats["ticks"] += int((~dead).sum())
            self.stats["plans"] += int(replanning.sum())
            return self.base_policy.get_action(info_dict, *args, **kwargs)

        heads = self.jepa_model.eb_heads
        device = next(self.jepa_model.parameters()).device
        prepared = self.base_policy._prepare_info(info_dict)
        z = self.jepa_model.encode({"pixels": prepared["pixels"][:, -1:].to(device)})["emb"][:, -1]
        zg = self.jepa_model.encode({"pixels": prepared["goal"][:, -1:].to(device)})["emb"][:, -1]
        if self._active is None or self._active.shape[0] != n:
            self.__dict__["_z_sub"] = torch.zeros_like(z)
            self.__dict__["_active"] = torch.zeros(n, dtype=torch.bool, device=device)

        if reset.any():
            self._active[torch.as_tensor(reset, device=device)] = False

        term = prepared.get("terminated")
        dead = torch.as_tensor(np.zeros(n, dtype=bool) if term is None else np.asarray(term, dtype=bool).reshape(n), device=device)
        idle = torch.as_tensor(buf_len == 0, device=device) & ~dead
        mid = (~idle) & self._active & ~dead

        flush = torch.zeros(n, dtype=torch.bool, device=device)
        if mid.any():
            flush = mid & (heads.predict_termination(z, self._z_sub) > heads.tau_term)

        refresh = idle | flush
        if refresh.any():
            if self.mode == "ebjewm_fixedk":
                z_sub = heads.predict_fixed_k(z, zg)
            else:
                z_sub = heads.predict_bottleneck(z, zg)
            beta0 = heads.predict_termination(z, z_sub)
            if self.mode == "ebjewm_nofallback":
                use = beta0 < heads.tau_term
            else:
                use = (heads.predict_temp_dist(z, zg) > heads.tau_far) & (beta0 < heads.tau_term)
            self._z_sub[refresh] = z_sub[refresh]
            self._active[refresh] = use[refresh]
            self.stats["plans"] += int(refresh.sum())
            self.stats["subgoal_plans"] += int((use & refresh).sum())
        self.stats["ticks"] += int((~dead).sum())
        self.stats["flushes"] += int(flush.sum())

        if flush.any():
            info_dict = dict(info_dict)
            info_dict["_needs_flush"] = reset | flush.cpu().numpy()
        return self.base_policy.get_action(info_dict, *args, **kwargs)