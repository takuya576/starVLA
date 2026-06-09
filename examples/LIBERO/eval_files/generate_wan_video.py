# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Standalone video generation from a trained WanGR00T checkpoint.

Loads a WanGR00T LoRA checkpoint, merges the adapters into the DiT, and runs
the Wan2.2-TI2V image-to-video pipeline (via _Wan2_Interface.generate, surfaced
through Wan_GR00T.generate_video) to imagine the future from a single LIBERO
observation frame + language instruction.

Usage:
    python examples/LIBERO/eval_files/generate_wan_video.py \
        --ckpt results/Checkpoints/20260604_172117_libero_all_WanGR00T_lora_true/checkpoints/steps_10000_pytorch_model.pt \
        --libero-suite libero_spatial --episode 0 \
        --num-frames 49 --num-inference-steps 50 \
        --out results/imagined/wan_libero_ep0.mp4
"""

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import torch
from PIL import Image

from starVLA.model.framework.base_framework import baseframework


def load_libero_condition(libero_root: Path, episode_idx: int):
    """Return (frame: PIL.Image, instruction: str) for one LIBERO episode.

    The primary-camera video for episode N lives at
        videos/chunk-000/observation.images.image/episode_{N:06d}.mp4
    and the per-episode instruction is the first task in meta/episodes.jsonl.
    """
    # TODO(human)
    video_path = (
        libero_root
        / f"videos/chunk-000/observation.images.image/episode_{episode_idx:06d}.mp4"
    )
    frame_np = iio.imread(str(video_path), index=0, plugin="pyav")
    frame = Image.fromarray(frame_np)

    inst_path = libero_root / "meta/episodes.jsonl"
    episodes = [json.loads(line) for line in inst_path.open()]
    instruction = episodes[episode_idx]["tasks"][0]

    return frame, instruction


def build_model(ckpt_path: str):
    """Build WanGR00T from checkpoint and merge LoRA into the DiT for inference."""
    model = baseframework.from_pretrained(ckpt_path)

    # generate() assembles a diffusers pipeline that expects a plain
    # WanTransformer3DModel; merge_and_unload folds the LoRA adapters into the
    # base DiT so the pipeline gets a compatible module (equivalent weights).
    from peft import PeftModel

    if isinstance(model.backbone.transformer, PeftModel):
        model.backbone.transformer = (
            model.backbone.transformer.merge_and_unload()
        )

    # NB: don't .to("cuda") here — generate() calls enable_model_cpu_offload(),
    # which owns device placement for the text_encoder / VAE / DiT itself.
    return model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", required=True, help="Path to steps_*_pytorch_model.pt"
    )
    parser.add_argument(
        "--libero-suite",
        default="libero_spatial",
        help="LIBERO suite dir prefix under playground/Datasets/LEROBOT_LIBERO_DATA",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", default="results/imagined/wan_libero_ep{idx}.mp4"
    )
    args = parser.parse_args()

    libero_root = (
        Path("playground/Datasets/LEROBOT_LIBERO_DATA")
        / f"{args.libero_suite}_no_noops_1.0.0_lerobot"
    )

    frame, instruction = load_libero_condition(libero_root, args.episode)
    print(f"[{args.libero_suite} ep{args.episode}] {instruction!r}")

    model = build_model(args.ckpt)

    # generate_video conditions on a single image (imgs[-1] if a list is given),
    # so pass the primary frame directly.
    sample = {"image": frame, "lang": instruction, "num_cameras": 1}

    out = model.generate_video(
        examples=[sample],
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        save_path=args.out,
        fps=args.fps,
        height=args.height,
        width=args.width,
    )
    print(f"Generated video saved to: {out['videos']}")


if __name__ == "__main__":
    main()
