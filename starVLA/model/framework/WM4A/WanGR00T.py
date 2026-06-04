# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
WanGR00T Framework — Wan2.2-TI2V World Model for Action Prediction.

Uses Wan2.2-TI2V-5B (DiT-based Text+Image-to-Video model) as the perception
backbone. The DiT's intermediate representations encode spatiotemporal
dynamics learned from large-scale video generation pretraining,
which are passed directly to the action head as cross-attention condition
for continuous action prediction.

Architecture:
  UMT5 (text) + VAE (image→latent) → WanTransformer3D
    → hidden_states [B, N, 3072]
    → FlowmatchingActionHead (cross-attention on 3072-dim tokens)
    → action predictions

Key differences from CosmoPredict2GR00T:
  - Text encoder: UMT5-XXL (dim=4096) vs T5 (dim=1024)
  - VAE latent channels: 48 vs 16
  - DiT hidden dim: 3072 (24×128) vs 2048 (16×128)
  - 30 transformer blocks vs 28
"""

import os
import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
    get_action_model,
)
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class WanGR00TDefaultConfig:
    """WanGR00T default parameters."""

    name: str = "WanGR00T"

    # === World Model backbone (Wan2.2-TI2V-5B-Diffusers) ===
    world_model: dict = field(
        default_factory=lambda: {
            "base_wm": (
                "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
            ),
            "extract_layers": [-1],
            # 0.0 = action-only (default); >0 enables co-training the DiT with
            # a flow-matching video loss, weighted by this scalar.
            "video_loss_weight": 0.0,
            # DiT denoising steps at inference (future imagination). 1 = single
            # forward at σ=1 (faithful, cheapest); >1 = progressive denoise.
            "video_inference_steps": 1,
        }
    )

    # Legacy compat: factory functions (vlm/__init__, world_model/__init__)
    # fall back to qwenvl.base_vlm when world_model.base_wm is absent.
    # vl_hidden_dim is read by some action heads (VLA_AdapterHeader, LayerwiseFM).
    # TODO next version should refactor to remove this redundant config section and update all shared utilities to read from world_model.base_wm instead of qwenvl.base_vlm.
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": (
                "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
            ),
            "vl_hidden_dim": 3072,
        }
    )

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 7,
            "state_dim": 7,
            "future_action_window_size": 7,
            "action_horizon": 8,
            "past_action_window_size": 0,
            "repeated_diffusion_steps": 8,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "diffusion_model_cfg": {
                # Decoupled from world model hidden_size; wm_projector bridges the gap
                "cross_attention_dim": 512,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("WanGR00T")
class Wan_GR00T(baseframework):
    """
    World-Model-for-Action framework using Wan2.2-TI2V-5B backbone.

    Components:
      - Wan2.2-TI2V DiT (UMT5 + VAE + WanTransformer3D) for features
      - Flow-matching (DiT) diffusion head for continuous action prediction

    The Wan world model provides spatiotemporal representations learned
    from large-scale video generation pretraining with expand_timesteps
    image conditioning (per-token timestep expansion).
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(WanGR00TDefaultConfig, config)

        # Load world model backbone
        self.backbone = get_world_model(config=self.config)

        # Project world model features to action model's cross-attention dim
        wm_hidden = self.backbone.model.config.hidden_size
        cross_attn_dim = (
            self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim
        )
        self.wm_projector = torch.nn.Linear(wm_hidden, cross_attn_dim)

        self.action_model: FlowmatchingActionHead = get_action_model(
            config=self.config
        )

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(
            self.config.framework.action_model.action_horizon
        )

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        actions = [example["action"] for example in examples]
        state = (
            [example["state"] for example in examples]
            if "state" in examples[0]
            else None
        )

        video_loss_weight = float(
            self.config.framework.world_model.get("video_loss_weight", 0.0)
        )

        # Step 1+2: imagine the future (no_grad) → features, + video loss (gt_future)
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        num_cameras = examples[0]["num_cameras"]
        wm_outputs = self.backbone(
            images=batch_images,
            instructions=instructions,
            num_cameras=num_cameras,
            gt_future=video_loss_weight > 0,
        )
        # hidden_states[-1]: [B, N_tokens, hidden_dim=3072] (detached features)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            last_hidden = self.wm_projector(wm_outputs["hidden_states"][-1])

        # Step 3: Action head forward and loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions),
                device=last_hidden.device,
                dtype=last_hidden.dtype,
            )
            actions_target = actions[:, -self.action_horizon :, :]

            repeated_diffusion_steps = (
                self.config.framework.action_model.get(
                    "repeated_diffusion_steps", 4
                )
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(
                repeated_diffusion_steps, 1, 1
            )
            last_hidden_repeated = last_hidden.repeat(
                repeated_diffusion_steps, 1, 1
            )

            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state),
                    device=last_hidden.device,
                    dtype=last_hidden.dtype,
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated
            )

        out = {"action_loss": action_loss}
        if wm_outputs["loss"] is not None:
            out["video_loss"] = video_loss_weight * wm_outputs["loss"]
        return out

    def _prepare_inputs(self, examples: List[dict]):
        """Shared preprocessing — guarantees predict_action and generate_video see identical inputs."""
        if type(examples) is not list:
            examples = [examples]
        batch_images = [
            to_pil_preserve(example["image"]) for example in examples
        ]
        instructions = [example["lang"] for example in examples]
        state = (
            [example["state"] for example in examples]
            if "state" in examples[0]
            else None
        )

        train_obs_image_size = getattr(
            self.config.datasets.vla_data, "obs_image_size", None
        )
        if train_obs_image_size:
            batch_images = resize_images(
                batch_images, target_size=train_obs_image_size
            )

        return batch_images, instructions, state

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]

        # Imagine the future from the current frame(s) → features (same path as
        # training, gt_future=False so no video loss).
        batch_images, instructions, state = self._prepare_inputs(examples)
        num_cameras = examples[0]["num_cameras"]
        wm_outputs = self.backbone(
            images=batch_images,
            instructions=instructions,
            num_cameras=num_cameras,
            gt_future=False,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            last_hidden = self.wm_projector(wm_outputs["hidden_states"][-1])

        state = (
            torch.from_numpy(np.array(state)).to(
                last_hidden.device, dtype=last_hidden.dtype
            )
            if state is not None
            else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    @torch.inference_mode()
    def generate_video(
        self,
        examples: List[dict],
        num_frames: int = 121,
        num_inference_steps: int = 10,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
        save_path: Optional[str] = None,
        fps: int = 16,
        height: int = 480,
        width: int = 832,
        **kwargs,
    ) -> dict:
        """Generate predicted future video from current observation + language.

        Mirrors CosmoPredict2GR00T.generate_video, but Wan2.2-TI2V uses a
        different pipeline contract (expand_timesteps for image conditioning),
        so the actual pipeline invocation is left to the implementer.
        """
        batch_images, instructions, _ = self._prepare_inputs(examples)

        device = next(self.backbone.transformer.parameters()).device
        generator = (
            torch.Generator(device=device).manual_seed(seed)
            if seed is not None
            else None
        )

        videos = []
        for i, (imgs, prompt) in enumerate(zip(batch_images, instructions)):
            cond_image = imgs[-1] if isinstance(imgs, list) else imgs

            result = self.backbone.generate(
                image=cond_image,
                prompt=prompt,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil",
                height=height,
                width=width,
                **kwargs,
            )
            frames = result.frames[0]

            if save_path is not None:
                from diffusers.utils import export_to_video

                output_path = (
                    save_path.format(idx=i)
                    if "{idx}" in save_path
                    else save_path
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                export_to_video(frames, output_path, fps=fps)
                videos.append(output_path)
            else:
                videos.append(frames)
        return {"videos": videos}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf
    from PIL import Image

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    # Point to the diffusers-format model for this dev test.
    cfg.framework.qwenvl.base_vlm = (
        "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    )
    cfg.framework.world_model.base_wm = (
        "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    )

    model: Wan_GR00T = Wan_GR00T(cfg)
    print(model)

    # --- Load a real libero frame + its instruction (libero_spatial ep 0) ---
    import json

    import imageio.v3 as iio

    libero_root = Path(
        "playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot"
    )
    episode_idx = 0
    video_path = (
        libero_root
        / "videos/chunk-000/observation.images.image"
        / f"episode_{episode_idx:06d}.mp4"
    )
    frame_np = iio.imread(str(video_path), index=0, plugin="pyav")
    frame = Image.fromarray(frame_np)

    episodes = [
        json.loads(line)
        for line in (libero_root / "meta/episodes.jsonl").open()
    ]
    instruction = episodes[episode_idx]["tasks"][0]
    print(f"[libero ep{episode_idx}] {instruction!r}")

    sample = {
        "action": np.zeros((16, 7), dtype=np.float16),
        "image": [frame, frame],  # 2 cameras (primary + wrist)
        "lang": instruction,
        "num_cameras": 2,
    }
    sample2 = sample.copy()

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")
    if "video_loss" in forward_output:
        print(f"Video Loss:  {forward_output['video_loss'].item():.4f}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    # Test generate_video — Quick preset: 4 steps, 49 frames, 480x832
    print("Testing generate_video...")
    video_output = model.generate_video(
        examples=[sample],
        num_frames=49,
        num_inference_steps=50,
        guidance_scale=5.0,
        seed=42,
        save_path="results/imagined/wan_libero_spatial_ep0.mp4",
        fps=8,
        height=480,
        width=832,
    )
    print(f"Generated video saved to: {video_output['videos']}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # vla_dataset_cfg = cfg.datasets.vla_data
    # from torch.utils.data import DataLoader
    # from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
    # cfg.datasets.vla_data.include_state = "False"
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    # train_dataloader = DataLoader(dataset, batch_size=2, num_workers=1, collate_fn=collate_fn)
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model = model.to(device)
    #     model(batch)
    # action = model.predict_action(examples=batch)
    print("Finished")
