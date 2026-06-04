# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Wan2.2-TI2V World Model Interface.

Wraps Wan-AI/Wan2.2-TI2V-5B-Diffusers (diffusion-based Text+Image-to-Video model)
as a world-model backend for starVLA action prediction frameworks.

Architecture (diffusers format):
  - UMT5EncoderModel: text instruction → text embeddings [B, L_text, 4096]
  - AutoencoderKLWan (VAE): observation image → video latents [B, 48, T, H/16, W/16]
  - WanTransformer3DModel: 30-layer DiT, hidden_dim=3072 (24 heads × 128 dim)
    Takes noised latents + text embeddings → denoised latents
    We extract intermediate hidden states for action-conditioning.

Note: The diffusers version of Wan2.2-TI2V-5B uses WanPipeline (text-only
conditioning) with expand_timesteps=True for TI2V mode. There is NO CLIP
image_encoder in this model variant — image conditioning is achieved through
per-token timestep expansion where the first frame's latent is conditioned
via timestep=0 (clean).

Key differences from CosmoPredict2:
  - Text encoder: UMT5 (dim=4096) vs T5 (dim=1024)
  - VAE latent channels: 48 vs 16
  - DiT hidden dim: 3072 (24×128) vs 2048 (16×128)
  - Scheduler: UniPCMultistepScheduler vs FlowMatchEulerDiscreteScheduler
  - No condition_mask / padding_mask (those are Cosmos-specific)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class _Wan2_Interface(nn.Module):
    """
    World model wrapper for Wan2.2-TI2V-5B-Diffusers.

    The key methods are:
      - forward(**kwargs) → model outputs with hidden_states
      - build_inputs(images, instructions) → dict of tensors
      - generate(**kwargs) → video generation (optional)

    Representation extraction strategy:
      We run a single DiT forward pass at noise level σ≈0 and register
      forward hooks to capture intermediate block outputs. These are
      collected into a [B, N_tokens, hidden_dim] tensor that the action
      head can consume — analogous to VLM hidden_states.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        wm_cfg = config.framework.get("world_model", {})
        model_name = wm_cfg.get(
            "base_wm",
            config.framework.get("qwenvl", {}).get(
                "base_vlm", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
            ),
        )
        self.config = config

        from diffusers import (
            AutoencoderKLWan,
            UniPCMultistepScheduler,
            WanTransformer3DModel,
        )
        from transformers import T5TokenizerFast, UMT5EncoderModel

        logger.info(f"Loading Wan2.2-TI2V from {model_name}")

        # --- Text encoder: UMT5-XXL ---
        self.tokenizer = T5TokenizerFast.from_pretrained(
            model_name, subfolder="tokenizer"
        )
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )

        # --- DiT transformer ---
        self.transformer = WanTransformer3DModel.from_pretrained(
            model_name, subfolder="transformer", torch_dtype=torch.bfloat16
        )

        # --- VAE (image → latents for DiT input, z_dim=48) ---
        self.vae = AutoencoderKLWan.from_pretrained(
            model_name, subfolder="vae", torch_dtype=torch.bfloat16
        )

        # --- Scheduler ---
        # CRITICAL: use_flow_sigmas=True makes UniPC's add_noise() use FM math:
        #   noisy = (1 - σ) · x_0 + σ · noise        (matches Wan's FM pretraining)
        # Without this flag, add_noise() silently defaults to DDPM α/β math and
        # training will silently produce a broken model.
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            model_name,
            subfolder="scheduler",
            use_flow_sigmas=True,
        )
        # Populate self.scheduler.timesteps / .sigmas for training-time sampling.
        self.scheduler.set_timesteps(
            num_inference_steps=self.scheduler.config.num_train_timesteps
        )

        # Use diffusers' VideoProcessor for image/video preprocessing (resize, normalize, etc.)
        from diffusers.video_processor import VideoProcessor

        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

        # Live VAE encode resolution (default = Wan pretrained 480x832).
        self._vae_height = wm_cfg.get("vae_height", 480)
        self._vae_width = wm_cfg.get("vae_width", 832)

        # Co-training: only DiT (WanTransformer3D) receives gradients; VAE & UMT5 frozen.
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # DiT: 24 heads × 128 dim = 3072
        self._hidden_size = (
            self.transformer.config.num_attention_heads
            * self.transformer.config.attention_head_dim
        )

        # Config-like shim for framework to read hidden_size
        class _FakeConfig:
            pass

        self._model_config = _FakeConfig()
        self._model_config.hidden_size = self._hidden_size

        # Hook storage for intermediate features
        self._intermediate_features = []
        self._hooks = []

        extract_layers = wm_cfg.get("extract_layers", [-1])
        self._extract_layers = extract_layers
        self._register_hooks()

    @property
    def model(self):
        """Compatibility shim: framework code accesses self.backbone.model.config.hidden_size"""

        class _ModelShim:
            pass

        shim = _ModelShim()
        shim.config = self._model_config
        return shim

    def _register_hooks(self):
        """Register forward hooks on selected transformer blocks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        num_blocks = len(self.transformer.blocks)
        for layer_idx in self._extract_layers:
            actual_idx = (
                layer_idx if layer_idx >= 0 else num_blocks + layer_idx
            )
            if 0 <= actual_idx < num_blocks:
                block = self.transformer.blocks[actual_idx]
                hook = block.register_forward_hook(self._capture_hook)
                self._hooks.append(hook)

    def _capture_hook(self, module, input, output):
        """Capture intermediate transformer block output."""
        if isinstance(output, tuple):
            self._intermediate_features.append(output[0])
        else:
            self._intermediate_features.append(output)

    def _encode_text(self, instructions, max_length=512):
        """Encode text instructions using UMT5."""
        device = next(self.text_encoder.parameters()).device

        text_inputs = self.tokenizer(
            instructions,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            text_embeds = self.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            ).last_hidden_state  # [B, L, 4096]

        return text_embeds.to(dtype=torch.bfloat16)  # [B, max_length, 4096]

    def _encode_cameras(self, images, num_frames=None, num_cameras=1):
        """Encode a multi-camera clip to fused latents.

        `images` is camera-major per sample: [cam0_t0..tN, cam1_t0..tN, ...].
        Each camera is VAE-encoded separately via _encode_one_camera, then
        width-concatenated. num_cameras=1 → single clip, identical to original.

        Returns:
            latents: [B, 48, T_latent, H/16, (W/16)*num_cameras]
        """
        per_cam = []
        for c in range(num_cameras):
            clips = []
            for sample_imgs in images:
                if not isinstance(sample_imgs, (list, tuple)):
                    sample_imgs = [sample_imgs]
                fpc = len(sample_imgs) // num_cameras  # frames per camera
                clips.append(sample_imgs[c * fpc : (c + 1) * fpc])
            per_cam.append(self._encode_one_camera(clips, num_frames))
        return torch.cat(per_cam, dim=-1)

    def _encode_one_camera(self, clips, num_frames=None):
        """Encode one camera's clip (list of frames per sample) through the VAE.

        Two-pass approach (same as CosmoPredict2):
          Pass 1: preprocess each sample, record real frame counts.
          Determine target_frames = num_frames if given, else batch max.
          Pass 2: truncate or pad each sample to target_frames.

        VAE config: z_dim=48, scale_factor_spatial=16, scale_factor_temporal=4
        T_latent = (target_frames - 1) // 4 + 1

        Args:
            clips: List of List of PIL Images [B, [frames...]] (one camera).
            num_frames: If given, pad/truncate to this exact count.
                If None (default), pad to the max frame count in the batch.

        Returns:
            latents: [B, 48, T_latent, H/16, W/16] video latent tensor
        """
        device = next(self.vae.parameters()).device
        dtype = self.vae.dtype
        height, width = self._vae_height, self._vae_width

        # Pass 1: preprocess each sample, record real frame counts
        preprocessed = []
        frame_counts = []
        for sample_imgs in clips:
            if not isinstance(sample_imgs, (list, tuple)):
                sample_imgs = [sample_imgs]

            video_tensor = self.video_processor.preprocess_video(
                sample_imgs, height=height, width=width
            )
            video_tensor = video_tensor.to(
                device=device, dtype=dtype
            )  # [1, C, n_imgs, H, W]
            preprocessed.append(video_tensor)
            frame_counts.append(video_tensor.shape[2])

        # Determine target frame count: use num_frames if specified, otherwise batch max
        target_frames = (
            num_frames if num_frames is not None else max(frame_counts)
        )

        # Pass 2: truncate or pad each sample to target_frames
        batch_videos = []
        for video_tensor in preprocessed:
            n = video_tensor.shape[2]
            if n > target_frames:
                video_tensor = video_tensor[:, :, :target_frames]
            elif n < target_frames:
                # Pad with last-frame repetition (matches official Wan pipeline)
                last_frame = video_tensor[:, :, -1:]
                padding = last_frame.repeat(1, 1, target_frames - n, 1, 1)
                video_tensor = torch.cat([video_tensor, padding], dim=2)
            batch_videos.append(
                video_tensor.squeeze(0)
            )  # [C, target_frames, H, W]

        # Stack to [B, C, target_frames, H, W]
        video = torch.stack(batch_videos, dim=0)

        with torch.no_grad():
            latents = self.vae.encode(
                video
            ).latent_dist.sample()  # [B, 48, T_latent, H/16, W/16]

        # Normalize latents (matches official Wan pipeline: (latent - mean) * (1/std))
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * latents_std

        return latents

    def _current_frames(self, images, num_cameras):
        """Pull the current frame (frame 0 of each camera) from a camera-major clip.

        Camera-major layout means camera c's block starts at index c*fpc, so
        sample_imgs[c*fpc] is its current frame. Returns a per-sample list of
        [cam0_f0, cam1_f0, ...] — identical structure at train (clip) and eval (1 frame).
        """
        out = []
        for sample_imgs in images:
            if not isinstance(sample_imgs, (list, tuple)):
                sample_imgs = [sample_imgs]
            fpc = len(sample_imgs) // num_cameras  # frames per camera
            out.append([sample_imgs[c * fpc] for c in range(num_cameras)])
        return out

    def _n_latent_frames(self):
        """Latent-frame count from the dataloader's video_indices (current + future)."""
        vla_cfg = getattr(self.config.datasets, "vla_data", None)
        vi = getattr(vla_cfg, "video_indices", None) if vla_cfg else None
        n_raw = len(vi) if vi else 1
        return (n_raw - 1) // self.vae_scale_factor_temporal + 1

    def _inference_timestep(self, latents, n_current_frames, future_t):
        """Per-token timestep (Wan 2.2 TI2V 2D mode): frame 0 clean, future = future_t.

        future_t is a [B] per-sample tensor in the DiT timestep scale
        ~[0, num_train_timesteps].
        """
        B, C, T, H, W = latents.shape
        p_t, p_h, p_w = self.transformer.config.patch_size
        T_p, H_p, W_p = T // p_t, H // p_h, W // p_w
        seq_len = T_p * H_p * W_p
        assert seq_len <= 1024, (
            f"seq_len={seq_len} exceeds WanTransformer3D rope_max_seq_len=1024 — "
            f"reduce vae_height/width, video_indices length, or cameras. "
            f"(T_lat={T}, H_lat={H}, W_lat={W}, patch={p_t},{p_h},{p_w})"
        )
        device, dtype = latents.device, self.transformer.dtype
        # future_t is a [B] per-sample timestep (callers normalize: extract_features
        # broadcasts its per-step scalar; video_loss already has per-sample t).
        future_t = future_t.to(device, dtype).reshape(B)
        n_cur_p = max(1, n_current_frames // p_t)
        t_bf = future_t[:, None].expand(B, T_p).clone()  # [B, T_p]
        t_bf[:, :n_cur_p] = 0.0  # current frame(s) clean
        t_per_token = t_bf[:, :, None, None].expand(B, T_p, H_p, W_p)
        return t_per_token.reshape(B, -1).contiguous()

    def _dit_forward(self, latents, timestep, text_embeds):
        """Single DiT forward + block-hook feature collection.

        Returns (hidden_states_tuple, velocity). Block hooks (from _register_hooks)
        fire on self.transformer regardless of caller, so feature capture works the
        same for the feature and video-loss paths.
        """
        self._intermediate_features.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            dit_output = self.transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=text_embeds,
            )
        velocity = (
            dit_output.sample if hasattr(dit_output, "sample") else dit_output
        )
        if isinstance(velocity, tuple):
            velocity = velocity[0]

        extracted = []
        for feat in self._intermediate_features:
            if feat.dim() == 5:  # [B, C, T, H, W] -> [B, T*H*W, C]
                B, C, T, H, W = feat.shape
                feat = feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            extracted.append(feat)
        if not extracted:
            # Fail fast: a misconfigured extract_layers means no hook fired. The
            # velocity (48-dim latent) is the wrong dimension for the action head's
            # projector (expects hidden_dim), so don't fabricate a bogus feature.
            raise RuntimeError(
                "No DiT hidden captured — check extract_layers is a valid block "
                f"index (got {self._extract_layers}, transformer has "
                f"{len(self.transformer.blocks)} blocks)."
            )
        return tuple(extracted), velocity

    @torch.no_grad()
    def _extract_features(
        self, images, instructions, num_cameras, n_latent_frames, num_steps, capture_step
    ):
        """Imagine the future (no_grad) and capture DiT features for the action head.

        Anchors on the current frame (clean), fills the future with noise, denoises
        `num_steps` steps (frame 0 re-clamped each step), and returns the detached
        features captured at `capture_step` (-1 = final). Identical at train & eval.
        """
        text_embeds = self._encode_text(instructions)
        cur = self._encode_cameras(
            self._current_frames(images, num_cameras), num_cameras=num_cameras
        )
        B, C, T_cur, H, W = cur.shape
        n_future = max(0, n_latent_frames - T_cur)
        if n_future > 0:
            noise = torch.randn(
                B, C, n_future, H, W, device=cur.device, dtype=cur.dtype
            )
            latents = torch.cat([cur, noise], dim=2)
        else:
            latents = cur

        self.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.timesteps
        cap = capture_step if capture_step >= 0 else len(timesteps) + capture_step
        cap = max(0, min(cap, len(timesteps) - 1))

        captured = None
        for i, t in enumerate(timesteps):
            # per-step σ applies to all samples → broadcast t to [B] inline
            # (keep `t` scalar for scheduler.step below)
            ts = self._inference_timestep(latents, T_cur, t.reshape(1).expand(B))
            hidden, vel = self._dit_forward(latents, ts, text_embeds)
            captured = hidden
            if i == cap:
                break
            future = self.scheduler.step(
                vel[:, :, T_cur:], t, latents[:, :, T_cur:]
            ).prev_sample
            latents = torch.cat([cur, future], dim=2)  # re-clamp frame 0
        return tuple(h.detach() for h in captured)

    def _video_loss(self, images, instructions, num_cameras):
        """Flow-matching video loss on the full clip (training only, gradient path).

        Encodes the full clip (current + GT future), noises the future at a random
        continuous t (logit-normal), and supervises the DiT's velocity on future
        frames only. The current frame stays clean (the TI2V image condition).
        """
        text_embeds = self._encode_text(instructions)
        full = self._encode_cameras(images, num_cameras=num_cameras)
        B, C, N, H, W = full.shape
        T_cur = 1  # latent frame 0 = current observation

        t = torch.sigmoid(torch.randn(B, device=full.device, dtype=torch.float32))
        noise = torch.randn_like(full)
        t_b = t.view(B, 1, 1, 1, 1).to(full.dtype)
        xt = (1 - t_b) * full + t_b * noise
        latents = torch.cat([full[:, :, :T_cur], xt[:, :, T_cur:]], dim=2)

        n_train = self.scheduler.config.num_train_timesteps
        timestep = self._inference_timestep(latents, T_cur, t * float(n_train))
        _, velocity = self._dit_forward(latents, timestep, text_embeds)

        target = (noise - full).detach()  # flow-matching velocity
        vel_future = velocity[:, :, T_cur:].float()
        tgt_future = target[:, :, T_cur:].float()
        return F.mse_loss(vel_future, tgt_future)

    def forward(
        self, images=None, instructions=None, num_cameras=1, gt_future=False, **kwargs
    ):
        """World-model forward (single entry, flag-selected).

        Always extracts action-conditioning features (no_grad, imagined future).
        When `gt_future=True` (training), also computes the flow-matching video
        loss on the full clip (gradient path; trains the DiT).

        Returns:
            dict with "hidden_states" (tuple of [B, N_tokens, hidden_dim]) and
            "loss" (video loss tensor or None).
        """
        wm_cfg = self.config.framework.world_model
        num_steps = int(wm_cfg.get("feature_denoise_steps", 1))
        capture_step = int(wm_cfg.get("feature_capture_step", -1))
        n_latent_frames = self._n_latent_frames()

        hidden = self._extract_features(
            images, instructions, num_cameras, n_latent_frames, num_steps, capture_step
        )
        loss = (
            self._video_loss(images, instructions, num_cameras)
            if gt_future
            else None
        )
        return {"hidden_states": hidden, "loss": loss}

    def generate(self, **kwargs):
        """Video generation using the WanPipeline.

        Not used during standard VLA training, but useful for visualization
        and planning-based approaches.
        #"""
        # from diffusers import WanPipeline

        # pipe = WanPipeline(
        #     tokenizer=self.tokenizer,
        #     text_encoder=self.text_encoder,
        #     vae=self.vae,
        #     transformer=self.transformer,
        #     scheduler=self.scheduler,
        # )
        from diffusers import WanImageToVideoPipeline

        pipe = WanImageToVideoPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            scheduler=self.scheduler,
            image_processor=None,
            image_encoder=None,
            transformer=self.transformer,
            transformer_2=None,
            boundary_ratio=None,
            expand_timesteps=True,
        )
        # device = next(self.transformer.parameters()).device
        # pipe.to(device)
        pipe.enable_model_cpu_offload()
        return pipe(**kwargs)
