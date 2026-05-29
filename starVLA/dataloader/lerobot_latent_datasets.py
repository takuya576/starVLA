"""LeRobotLatentDataset: augments LeRobotSingleDataset samples with
pre-extracted Wan VAE latents + UMT5 text embeddings.

Use this when a dataset ships with `latents/` cached on disk (e.g.
robbyant/libero-long-lerobot) and you want to skip VAE+UMT5 during training.
The inference path (predict_action) keeps using live VAE encoding via
Wan2._encode_images_vae.

Design follows lingbot-va/wan_va/dataset/lerobot_latent_dataset.py:
  - per-camera .pth files, each holding one episode's full latent
  - width-wise spatial concat across cameras at load time
  - runtime windowing with a random cursor (Option A — see plan doc)

Implemented as a subclass of LeRobotSingleDataset rather than a wrapper,
because LeRobotMixtureDataset.__getitem__ bypasses children's __getitem__
and calls get_step_data / transforms / _pack_sample directly. Subclassing
puts our overrides on the exact seams the mixture uses.
"""

from pathlib import Path

import torch

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


class LeRobotLatentDataset(LeRobotSingleDataset):
    """Adds {latents, text_emb} to each sample by reading per-camera .pth files,
    slicing a window around the sample's base_index, and width-fusing cameras.

    Each sample yields (in addition to action/state/lang/robot_tag from the base):
        latents  : Tensor [48, window_latent, H_lat, W_lat * num_cams]
        text_emb : Tensor [512, 4096]
    """

    VAE_TEMPORAL_FACTOR = 4
    LATENT_CHANNELS = 48

    def __init__(
        self,
        *,
        latent_root: Path,
        video_keys: list,
        window_latent: int = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_root = Path(latent_root)
        self.video_keys_latent = list(video_keys)
        self.window_latent = int(window_latent)
        self._latent_index = self._build_index()
        # Stash trajectory_id and base_index in get_step_data so _pack_sample
        # (which only receives the transformed data dict) can use them.
        # Safe under DataLoader num_workers > 0: each worker holds its own
        # dataset instance, and the get_step_data → _pack_sample call chain
        # is sequential within a worker.
        self._last_trajectory_id = None
        self._last_base_index = None

    def _build_index(self) -> dict:
        """Map (episode_id, video_key) -> .pth path.

        Filename pattern: episode_{episode_index:06d}_{start_frame}_{end_frame}.pth
        For datasets with one entry per episode (LIBERO), each episode has one file.
        """
        idx = {}
        for cam_key in self.video_keys_latent:
            cam_dir = self.latent_root / cam_key
            assert cam_dir.is_dir(), f"Missing latent dir: {cam_dir}"
            for fp in cam_dir.glob("episode_*.pth"):
                ep = int(fp.stem.split("_")[1])
                idx[(ep, cam_key)] = fp
        return idx

    def get_step_data(self, trajectory_id: int, base_index: int):
        """Override: stash IDs so _pack_sample can look up the latent window."""
        self._last_trajectory_id = trajectory_id
        self._last_base_index = base_index
        return super().get_step_data(trajectory_id, base_index)

    def _pack_sample(self, data: dict) -> dict:
        """Override: augment the base sample with latents + text_emb."""
        sample = super()._pack_sample(data)
        if self._last_trajectory_id is None:
            return sample
        latents, text_emb = self._load_window(
            self._last_trajectory_id, self._last_base_index
        )
        sample["latents"] = latents
        sample["text_emb"] = text_emb
        sample.pop("image", None)
        return sample

    def _load_window(self, ep_id: int, base_index: int):
        per_cam = []
        text_emb = None
        for cam_key in self.video_keys_latent:
            fp = self._latent_index[(ep_id, cam_key)]
            data = torch.load(fp, map_location="cpu", weights_only=False)
            T_l = data["latent_num_frames"]
            H_l, W_l = data["latent_height"], data["latent_width"]
            # Stored as [T*H*W, C]; reshape to [C, T, H, W]
            full = (
                data["latent"]
                .view(T_l, H_l, W_l, self.LATENT_CHANNELS)
                .permute(3, 0, 1, 2)
                .contiguous()
            )
            per_cam.append(full)
            text_emb = data["text_emb"]  # same text across cameras

        T_l_total = per_cam[0].shape[1]

        # base_index sits at window position 0 (Wan TI2V convention: the
        # current observation IS the clean conditioning frame). The window
        # then covers [now, now+1, ..., now+window_latent-1] in latent space.
        base_latent = base_index // self.VAE_TEMPORAL_FACTOR
        win = self.window_latent
        start = base_latent
        if start + win > T_l_total:
            start = max(0, T_l_total - win)
        end = start + win

        sliced = [
            c[:, start:end] for c in per_cam
        ]  # c : [48, cur_len, H_lat, W_lat]

        # Pad short episodes with last-frame repetition.
        cur_len = sliced[0].shape[1]
        if cur_len < win:
            pad_n = win - cur_len
            sliced = [
                torch.cat([s, s[:, -1:].repeat(1, pad_n, 1, 1)], dim=1)
                for s in sliced
            ]  # s[:, -1:] : [48, 1, H_lat, W_lat]

        # Width-wise spatial fusion (lingbot-va style).
        fused = torch.cat(sliced, dim=-1)

        return fused.float(), text_emb.float()
