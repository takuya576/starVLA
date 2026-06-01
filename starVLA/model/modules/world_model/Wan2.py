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

    def _sample_noise_and_target(self, latents):
        """Sample per-frame timesteps and build flow-matching noise + velocity target.

        Uses self.scheduler (UniPCMultistepScheduler, use_flow_sigmas=True) as the
        source of truth for the σ grid. Mirrors lingbot-va/wan_va/train.py::_add_noise
        in structure: one timestep per (sample, frame), broadcast across the spatial
        tokens within that frame.

        Returns:
            noisy_latents : [B, C, T, H, W]   (1-σ)·x_0 + σ·noise
            timestep      : [B, seq_len]      per-token float (scheduler.timesteps scale)
            target        : [B, C, T, H, W]   FM velocity (noise - x_0)
        """
        B, _, T, H, W = latents.shape
        device, dtype = latents.device, latents.dtype

        n_steps = len(self.scheduler.timesteps)
        idx = torch.randint(0, n_steps, (B, T), device="cpu")
        # Frame 0 is the image condition (Wan TI2V convention); keep it clean.
        idx[:, 0] = n_steps - 1
        sigmas = self.scheduler.sigmas[idx].to(device, dtype)
        t_frame = self.scheduler.timesteps[idx].to(device, dtype)

        s = sigmas[:, None, :, None, None]
        noise = torch.randn_like(latents)
        noisy_latents = (1 - s) * latents + s * noise
        target = noise - latents

        p_t, p_h, p_w = self.transformer.config.patch_size
        T_p, H_p, W_p = T // p_t, H // p_h, W // p_w
        t_patched = t_frame[:, ::p_t] if p_t > 1 else t_frame
        t_per_token = t_patched[:, :, None, None].expand(B, T_p, H_p, W_p)
        timestep = t_per_token.reshape(B, -1)
        return noisy_latents, timestep, target

    def build_inputs(
        self,
        images=None,
        instructions=None,
        *,
        noise_mode: bool = False,
        num_cameras: int = 1,
        **kwargs,
    ):
        """Build inputs for the Wan DiT world model.

        Encoding pipeline:
        1. Text → UMT5 → text embeddings [B, L, 4096]
        2. Image → VAE → latents [B, 48, T, H', W'] (DiT input)

        Note: No CLIP image conditioning — this diffusers variant uses
        expand_timesteps mode (per-token timesteps) instead.

        Args:
            noise_mode: If True, inject flow-matching noise (training co-train path).
                If False (default), feed clean latents at timestep=0 (inference / legacy).

        Returns:
            dict with keys matching forward() expectations
        """
        assert images is not None and instructions is not None
        assert len(images) == len(instructions)
        text_embeds = self._encode_text(instructions)
        latents = self._encode_cameras(images, num_cameras=num_cameras)

        p_t, p_h, p_w = self.transformer.config.patch_size
        _, _, T, H, W = latents.shape
        seq_len = (T // p_t) * (H // p_h) * (W // p_w)
        assert seq_len <= 1024, (
            f"seq_len={seq_len} exceeds WanTransformer3D rope_max_seq_len=1024. "
            f"Reduce num_frames or image resolution. "
            f"(T_lat={T}, H_lat={H}, W_lat={W}, patch={p_t},{p_h},{p_w})"
        )

        if noise_mode:
            noisy_latents, timestep, target = self._sample_noise_and_target(
                latents
            )
            return {
                "hidden_states": noisy_latents,
                "timestep": timestep,
                "encoder_hidden_states": text_embeds,
                "_target": target,
                "_is_wm_input": True,
            }

        batch_size = latents.shape[0]
        device = latents.device
        timestep = torch.zeros(
            batch_size, seq_len, device=device, dtype=torch.long
        )
        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": text_embeds,
            "_is_wm_input": True,
        }

    def forward(self, **kwargs):
        """Forward pass through the Wan DiT transformer.

        Runs a single-step forward to extract rich spatiotemporal features.
        When `_target` is supplied (cotrain path), also computes a flow-matching
        video loss against the DiT's velocity prediction.
        Returns an output object with .hidden_states (and optional .loss).
        """
        kwargs.pop(
            "_is_wm_input", False
        )  # pop internal routing flags from kwargs to avoid passing them downstream
        kwargs.pop("output_hidden_states", False)
        kwargs.pop("return_dict", True)
        kwargs.pop("output_attentions", None)
        target = kwargs.pop("_target", None)

        self._intermediate_features.clear()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            dit_output = self.transformer(
                hidden_states=kwargs["hidden_states"],
                timestep=kwargs["timestep"],
                encoder_hidden_states=kwargs["encoder_hidden_states"],
            )

        # Collect features from hooks
        # WanTransformer3DModel blocks output [B, seq_len, hidden_dim] (already flattened)
        extracted = []
        for feat in self._intermediate_features:
            if feat.dim() == 5:
                # [B, C, T, H, W] -> [B, T*H*W, C]
                B, C, T, H, W = feat.shape
                feat = feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            extracted.append(feat)

        # Fallback: use transformer output directly
        if not extracted:
            out = (
                dit_output.sample
                if hasattr(dit_output, "sample")
                else dit_output
            )
            if isinstance(out, tuple):
                out = out[0]
            if out.dim() == 5:
                B, C, T, H, W = out.shape
                out = out.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            extracted.append(out)

        # Flow-matching video loss (cotrain path; only computed when `_target` was passed)
        video_loss = None
        if target is not None:
            pred = (
                dit_output.sample
                if hasattr(dit_output, "sample")
                else dit_output
            )
            if isinstance(pred, tuple):
                pred = pred[0]
            # TODO(human): compute the flow-matching video loss between `pred`
            # and `target`. Both are [B, C, T, H, W] velocity tensors (pred may
            # be bf16; cast to float32 before reducing for numerical stability).
            # A mean-reduced MSE is a reasonable starting point; lingbot-va uses
            # frame-wise normalization with SNR weighting as a richer alternative.
            #
            # Frame 0 is the clean image condition (σ ≈ 0). Its target velocity
            # is independent random noise, which the model can't predict from a
            # clean input — so its loss contribution is a constant noise floor,
            # not a learning signal. Mask it out to avoid diluting the gradient.
            frame_mask = torch.ones_like(target)
            frame_mask[:, :, 0] = 0.0
            sq_err = (pred.float() - target.float().detach()) ** 2
            video_loss = (
                sq_err * frame_mask
            ).sum() / frame_mask.sum().clamp_min(1.0)

        class _WMOutput:
            def __init__(self, hidden_states_tuple, loss=None):
                self.hidden_states = hidden_states_tuple
                self.loss = loss  # TODO if you want to add loss for image reconstruction or other auxiliary objectives, you can include it here and return it in the forward pass

        return _WMOutput(hidden_states_tuple=tuple(extracted), loss=video_loss)

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
