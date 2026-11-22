#!/usr/bin/env python
"""
Trains EBHeads on the FROZEN LeWM backbone with strict Train / Val / Test hygiene:
  - Test Split (ep_id < 1000, or first 20% of dataset)    : NEVER touched here (reserved for eval.py).
  - Val Split  (1000 <= ep_id < 2000, or next 10%)        : Used only to calibrate tau_far and tau_term.
  - Train Split(ep_id >= 2000, or remaining 70%)          : Used to train EBHeads.
Decouples --fixed_k (used ONLY for fixed_k_jumper targets) from --cem_horizon (used for far_mask
and tau_far calibration) so k-sweeps never confound the event-bottleneck arm.
"""
import argparse
from pathlib import Path
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms

from eb_planner import EBHeads, load_lewm_model


def get_img_transform(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=(img_size, img_size), antialias=True),
        ]
    )


def format_single_frame(raw_px, tf) -> torch.Tensor:
    if isinstance(raw_px, np.ndarray):
        while raw_px.ndim > 3:
            raw_px = raw_px[0]
        if raw_px.ndim == 3 and raw_px.shape[0] in (1, 3) and raw_px.shape[-1] not in (1, 3):
            raw_px = np.transpose(raw_px, (1, 2, 0))
    elif torch.is_tensor(raw_px):
        while raw_px.ndim > 3:
            raw_px = raw_px[0]
    out = tf(raw_px)
    while out.ndim > 3:
        out = out.squeeze(0)
    return out


@torch.no_grad()
def encode_episode_list(jepa, dataset, ep_col, ep_ids, tf, device):
    episodes = []
    for idx, ep_id in enumerate(ep_ids):
        rows = np.nonzero(ep_col == ep_id)[0]
        frames = torch.stack([format_single_frame(dataset[int(r)]["pixels"], tf) for r in rows]).to(device)
        info = jepa.encode({"pixels": frames.unsqueeze(1)})
        episodes.append(info["emb"].squeeze(1).cpu())
        if (idx + 1) % 50 == 0:
            print(f"  Encoded {idx + 1}/{len(ep_ids)} episodes...")
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="tworoom_lewm/weights.pt")
    ap.add_argument("--dataset", default="tworoom")
    ap.add_argument("--num_train_episodes", type=int, default=300)
    ap.add_argument("--num_val_episodes", type=int, default=50)
    ap.add_argument("--cem_horizon", type=int, default=25, help="Planner horizon for far_mask & tau_far calibration")
    ap.add_argument("--fixed_k", type=int, default=25, help="Step offset for Fixed-k Jumper ablation ONLY")
    ap.add_argument("--max_horizon", type=int, default=75)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="Explicit output path (never overwrites other checkpoints)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1/4] Loading frozen LeWM backbone from {args.policy} (seed={args.seed}, fixed_k={args.fixed_k})...")
    jepa = load_lewm_model(args.policy, device=device)

    cache_dir = Path(swm.data.utils.get_cache_dir())
    out_path = Path(args.out).expanduser()

    print(f"[2/4] Loading HDF5Dataset '{args.dataset}' with strict Train / Val / Test split...")
    dataset = swm.data.HDF5Dataset(
        args.dataset,
        keys_to_cache=["action"],
        cache_dir=cache_dir,
    )
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_col = dataset.get_col_data(col_name)
    all_eps = np.unique(ep_col)

    if len(all_eps) >= 2500:
        val_candidates = all_eps[(all_eps >= 1000) & (all_eps < 2000)]
        train_candidates = all_eps[all_eps >= 2000]
    else:
        # Automatic proportional split for smaller datasets (e.g., cube_single_expert):
        # First 20% = Held-Out Test, Next 10% = Validation, Remaining 70% = Train
        n_total = len(all_eps)
        val_candidates = all_eps[int(0.20 * n_total) : int(0.30 * n_total)]
        train_candidates = all_eps[int(0.30 * n_total) :]

    rng = np.random.default_rng(args.seed)
    train_eps = rng.choice(train_candidates, size=min(args.num_train_episodes, len(train_candidates)), replace=False)
    val_eps = rng.choice(val_candidates, size=min(args.num_val_episodes, len(val_candidates)), replace=False)

    tf = get_img_transform(224)
    print(f"  Encoding {len(train_eps)} Train episodes...")
    train_latents = [ep for ep in encode_episode_list(jepa, dataset, ep_col, train_eps, tf, device) if len(ep) > args.cem_horizon + 5]
    print(f"  Encoding {len(val_eps)} Validation episodes...")
    val_latents = [ep for ep in encode_episode_list(jepa, dataset, ep_col, val_eps, tf, device) if len(ep) > args.cem_horizon + 5]

    print(f"[3/4] Training EBHeads for {args.steps} gradient steps...")
    heads = EBHeads(dim=192).to(device)
    opt = torch.optim.AdamW(heads.parameters(), lr=1e-3, weight_decay=1e-4)

    B = 64
    for step in range(1, args.steps + 1):
        z_t_l, z_g_l, z_door_l, z_fixk_l, z_pos_l, z_neg_l, td_l = [], [], [], [], [], [], []
        for _ in range(B):
            ep = train_latents[np.random.randint(len(train_latents))]
            L = len(ep)
            t0 = np.random.randint(0, L - 10)
            delta = np.random.randint(6, min(args.max_horizon + 5, L - t0))
            tg = t0 + delta

            z_t = ep[t0]
            z_g = ep[tg]

            seg = ep[t0 : tg + 1]
            vel = (seg[1:] - seg[:-1]).pow(2).sum(dim=-1)
            t_star_rel = int(torch.argmax(vel).item()) + 1
            t_star = t0 + t_star_rel
            z_door = ep[t_star]

            pos_t = int(np.clip(t_star + np.random.randint(-2, 3), 0, L - 1))
            neg_candidates = [t for t in range(L) if abs(t - t_star) > 2]
            neg_t = int(neg_candidates[np.random.randint(len(neg_candidates))])

            tk = min(t0 + args.fixed_k, tg)
            z_fixk = ep[tk]

            z_t_l.append(z_t)
            z_g_l.append(z_g)
            z_door_l.append(z_door)
            z_fixk_l.append(z_fixk)
            z_pos_l.append(ep[pos_t])
            z_neg_l.append(ep[neg_t])
            td_l.append(min(1.0, delta / float(args.max_horizon)))

        z_t_b = torch.stack(z_t_l).to(device)
        z_g_b = torch.stack(z_g_l).to(device)
        z_door_b = torch.stack(z_door_l).to(device)
        z_fixk_b = torch.stack(z_fixk_l).to(device)
        z_pos_b = torch.stack(z_pos_l).to(device)
        z_neg_b = torch.stack(z_neg_l).to(device)
        td_t = torch.tensor(td_l, dtype=torch.float32, device=device)

        td_pred = heads.predict_temp_dist(z_t_b, z_g_b)
        td_zero = heads.predict_temp_dist(z_t_b, z_t_b)
        loss_td = F.mse_loss(td_pred, td_t) + F.mse_loss(td_zero, torch.zeros_like(td_zero))

        # Decoupled far_mask: depends ONLY on cem_horizon (25), NEVER on args.fixed_k!
        far_mask = (td_t > (args.cem_horizon / float(args.max_horizon))).float().unsqueeze(-1)
        z_sub_pred = heads.predict_bottleneck(z_t_b, z_g_b)
        z_fixk_pred = heads.predict_fixed_k(z_t_b, z_g_b)
        loss_door = (far_mask * (z_sub_pred - z_door_b).pow(2)).mean()
        loss_fixk = (far_mask * (z_fixk_pred - z_fixk_b).pow(2)).mean()

        pos_true = heads.predict_termination(z_pos_b, z_door_b)
        pos_pred = heads.predict_termination(z_pos_b, z_sub_pred.detach())
        neg_true = heads.predict_termination(z_neg_b, z_door_b)
        neg_pred = heads.predict_termination(z_neg_b, z_sub_pred.detach())

        loss_term = 0.25 * (
            F.binary_cross_entropy(pos_true, torch.ones_like(pos_true))
            + F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
            + F.binary_cross_entropy(neg_true, torch.zeros_like(neg_true))
            + F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        )

        loss = loss_td + loss_door + loss_fixk + 0.5 * loss_term
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0:
            print(
                f"  Step {step:4d}/{args.steps} | TD: {loss_td.item():.4f} | "
                f"Door: {loss_door.item():.4f} | FixK: {loss_fixk.item():.4f} | Term: {loss_term.item():.4f}"
            )

    print("[4/4] Calibrating thresholds on held-out Validation Split...")
    heads.eval()
    with torch.no_grad():
        val_z_start, val_z_k, val_z_far, val_z_at_sub = [], [], [], []
        for ep in val_latents:
            far_idx = min(60, len(ep) - 1)
            if far_idx > args.cem_horizon + 2:
                val_z_start.append(ep[0])
                # Decoupled tau_far calibration: always uses cem_horizon (25), NEVER args.fixed_k!
                val_z_k.append(ep[args.cem_horizon])
                val_z_far.append(ep[far_idx])
                seg = ep[0 : far_idx + 1]
                vel = (seg[1:] - seg[:-1]).pow(2).sum(dim=-1)
                t_star = int(torch.argmax(vel).item()) + 1
                val_z_at_sub.append(seg[t_star])

        vz0 = torch.stack(val_z_start).to(device)
        vzk = torch.stack(val_z_k).to(device)
        vzf = torch.stack(val_z_far).to(device)
        vzs = torch.stack(val_z_at_sub).to(device)

        cal_tau_far = torch.median(heads.predict_temp_dist(vz0, vzk)).clamp(0.20, 0.60)

        val_sub_pred = heads.predict_bottleneck(vz0, vzf)
        beta_start = heads.predict_termination(vz0, val_sub_pred).median()
        beta_reached = heads.predict_termination(vzs, val_sub_pred).median()
        cal_tau_term = (0.5 * (beta_start + beta_reached)).clamp(0.35, 0.85)

        heads.tau_far.copy_(cal_tau_far)
        heads.tau_term.copy_(cal_tau_term)
        heads.unc_scale.zero_()
        print(
            f"  Calibrated Val Thresholds -> tau_far: {heads.tau_far.item():.4f}, "
            f"tau_term: {heads.tau_term.item():.4f} (beta_start={beta_start.item():.2f}, beta_reached={beta_reached.item():.2f})"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(heads.state_dict(), out_path)
    print(f"Saved trained & calibrated EBHeads to {out_path}")


if __name__ == "__main__":
    main()