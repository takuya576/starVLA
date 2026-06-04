"""Smoke test for the Wan video-cotrain path on LIBERO.

Run from repo root:
    python examples/LIBERO/train_files/smoke_test_wan_cotrain.py \
        --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml

Exercises the on-the-fly path: live VAE+UMT5 encoding of a multi-frame,
multi-camera clip. Forces video_loss_weight > 0 so the co-train path runs.

What it checks (in order — bails on first failure):
  1. Dataset yields a camera-major PIL clip + num_cameras with sane counts.
  2. WanGR00T.forward returns {action_loss, video_loss} as finite scalars.
  3. Backward through the summed loss reaches the DiT and only the DiT.
  4. predict_action returns normalized_actions of shape [B, horizon, dim]
     — same call site the trainer hits during eval_action_model.

Use this before kicking off a real training run.
"""

import argparse

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
from starVLA.model.framework.base_framework import build_framework


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
    )
    parser.add_argument("--batch_size", type=int, default=2)  # >1 catches broadcast bugs
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Force the video loss on so the co-train path is exercised regardless of
    # the training YAML value (which may be 0 during an action-only warmup).
    if float(cfg.framework.world_model.get("video_loss_weight", 0.0)) <= 0:
        cfg.framework.world_model.video_loss_weight = 0.1
    print(f"[smoke] device={device}")

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    print("\n[1/3] Dataset smoke")
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    sample = dataset[0]
    assert "action" in sample, "Sample missing 'action'."

    # On-the-fly path: dataset serves a camera-major PIL clip + num_cameras.
    assert "image" in sample, "Sample missing 'image'."
    assert "num_cameras" in sample, "Sample missing 'num_cameras'."
    n_cams = sample["num_cameras"]
    n_frames = len(cfg.datasets.vla_data.video_indices)
    print(f"  image   : {len(sample['image'])} PILs (num_cameras={n_cams})")
    assert len(sample["image"]) == n_cams * n_frames, (
        f"image count {len(sample['image'])} != num_cameras*frames "
        f"({n_cams}*{n_frames})"
    )
    print(f"  action  : {sample['action'].shape}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=collate_fn,
    )
    batch = next(iter(dataloader))
    assert len(batch) == args.batch_size
    print(f"  batch len: {len(batch)}  ✓")

    # ------------------------------------------------------------------
    # 2. Forward
    # ------------------------------------------------------------------
    print("\n[2/3] Forward smoke")
    model = build_framework(cfg).to(device)
    model.train()
    out = model(batch)

    assert (
        "action_loss" in out
    ), f"forward output missing action_loss: {out.keys()}"
    assert (
        "video_loss" in out
    ), "forward output missing video_loss — is video_loss_weight > 0?"
    action_loss = out["action_loss"]
    video_loss = out["video_loss"]
    assert torch.isfinite(
        action_loss
    ).all(), f"action_loss not finite: {action_loss}"
    assert torch.isfinite(
        video_loss
    ).all(), f"video_loss not finite: {video_loss}"
    print(f"  action_loss: {action_loss.item():.4f}")
    print(f"  video_loss : {video_loss.item():.4f}")

    # ------------------------------------------------------------------
    # 3. Gradient sanity
    # ------------------------------------------------------------------
    print("\n[3/3] Gradient sanity")
    loss = action_loss + video_loss
    loss.backward()

    dit_param = next(model.backbone.transformer.parameters())
    assert (
        dit_param.grad is not None and dit_param.grad.abs().sum() > 0
    ), "DiT param has no gradient — video_loss did not backprop into the world model."
    print("  DiT grad present and non-zero  ✓")

    vae_param = next(model.backbone.vae.parameters())
    assert vae_param.grad is None, "VAE should be frozen but received gradients."
    print("  VAE grad None (frozen)         ✓")

    txt_param = next(model.backbone.text_encoder.parameters())
    assert (
        txt_param.grad is None
    ), "Text encoder should be frozen but received gradients."
    print("  Text encoder grad None (frozen) ✓")

    # ------------------------------------------------------------------
    # 4. predict_action (eval-path) smoke
    # ------------------------------------------------------------------
    # The trainer calls model.predict_action periodically during training
    # (eval_action_model). If its data contract diverges from the dataloader's
    # output, the failure only surfaces at the first eval step — sometimes
    # hours into a run. Exercising it here catches that class of bug early.
    print("\n[4/4] predict_action smoke")
    pred_out = model.predict_action(batch)

    assert (
        "normalized_actions" in pred_out
    ), f"predict_action output missing normalized_actions: {pred_out.keys()}"
    pred_actions = pred_out["normalized_actions"]
    action_horizon = int(cfg.framework.action_model.action_horizon)
    action_dim = int(cfg.framework.action_model.action_dim)
    expected_shape = (args.batch_size, action_horizon, action_dim)
    assert pred_actions.shape == expected_shape, (
        f"normalized_actions shape {pred_actions.shape} != expected {expected_shape}"
    )
    import numpy as np
    assert np.isfinite(pred_actions).all(), (
        f"normalized_actions contains non-finite values: "
        f"nan={np.isnan(pred_actions).sum()}, inf={np.isinf(pred_actions).sum()}"
    )
    print(f"  normalized_actions: {pred_actions.shape} {pred_actions.dtype}  ✓")

    print("\n[smoke] all checks passed.")


if __name__ == "__main__":
    main()
