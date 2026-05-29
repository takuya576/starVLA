"""Smoke test for the Wan video-cotrain path on LIBERO.

Run from repo root:
    python examples/LIBERO/train_files/smoke_test_wan_cotrain.py \
        --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml

What it checks (in order — bails on first failure):
  1. Dataset yields {latents, text_emb, action, ...} with sane shapes.
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
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    print("\n[1/3] Dataset smoke")
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    sample = dataset[0]

    assert "latents" in sample, (
        "Sample missing 'latents'. Did you set "
        "datasets.vla_data.use_precomputed_latents: true?"
    )
    assert "text_emb" in sample, "Sample missing 'text_emb'."
    assert "action" in sample, "Sample missing 'action'."

    latents = sample["latents"]
    text_emb = sample["text_emb"]
    print(f"  latents : {tuple(latents.shape)} {latents.dtype}")
    print(f"  text_emb: {tuple(text_emb.shape)} {text_emb.dtype}")
    print(f"  action  : {sample['action'].shape}")

    # Expected: latents [48, window_latent, H_lat, W_lat * num_cams]
    assert (
        latents.dim() == 4 and latents.shape[0] == 48
    ), f"latents should be [48, T, H, W*ncams], got {tuple(latents.shape)}"
    assert (
        text_emb.dim() == 2 and text_emb.shape[-1] == 4096
    ), f"text_emb should be [L, 4096], got {tuple(text_emb.shape)}"

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

    # VAE and text_encoder are None when precomputed_latents_only=True
    # (they're not loaded into VRAM at all in that mode).
    if model.backbone.vae is not None:
        vae_param = next(model.backbone.vae.parameters())
        assert (
            vae_param.grad is None
        ), "VAE should be frozen but received gradients."
        print("  VAE grad None (frozen)         ✓")
    else:
        print("  VAE not loaded (precomputed_latents_only) ✓")

    if model.backbone.text_encoder is not None:
        txt_param = next(model.backbone.text_encoder.parameters())
        assert (
            txt_param.grad is None
        ), "Text encoder should be frozen but received gradients."
        print("  Text encoder grad None (frozen) ✓")
    else:
        print("  Text encoder not loaded (precomputed_latents_only) ✓")

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
