"""
Base framework abstraction providing:
- Pretrained loading (config + normalization stats + weights)
- Action space utilities (dimension, stats, (un)normalization)
- Trainable module discovery helper
Note: No device placement or optimizer concerns handled here (delegated to trainer).
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import PretrainedConfig, PreTrainedModel

from starVLA.model.framework.share_tools import (
    dict_to_namespace,
    read_mode_config,
)
from starVLA.model.tools import (
    FRAMEWORK_REGISTRY,
    FrameworkTools,
    auto_get_trainable_modules,
)
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)
_FRAMEWORKS_IMPORTED = False


def _auto_import_framework_modules() -> None:
    global _FRAMEWORKS_IMPORTED
    if _FRAMEWORKS_IMPORTED:
        return

    _SKIP = {"__init__", "base_framework", "share_tools"}
    framework_dir = Path(__file__).resolve().parent

    # Scan top-level modules (backwards compat)
    for _, module_name, is_pkg in pkgutil.iter_modules([str(framework_dir)]):
        if module_name in _SKIP:
            continue
        if is_pkg:
            # Scan sub-packages (VLM4A/, WM4A/, etc.)
            sub_dir = framework_dir / module_name
            for _, sub_name, _ in pkgutil.iter_modules([str(sub_dir)]):
                if sub_name.startswith("_"):
                    continue
                importlib.import_module(
                    f"starVLA.model.framework.{module_name}.{sub_name}"
                )
        else:
            importlib.import_module(f"starVLA.model.framework.{module_name}")

    _FRAMEWORKS_IMPORTED = True


def build_framework(
    cfg,
):  # The single entry point for building different model frameworks
    """
    Build a framework model from config.
    Args:
        cfg: Config object containing `cfg.framework.name`.
    Returns:
        nn.Module: Instantiated framework model.
    """
    if not hasattr(cfg, "framework") or not hasattr(cfg.framework, "name"):
        raise ValueError(
            "Missing `cfg.framework.name`. The framework API now only accepts `framework.name`."
        )

    _auto_import_framework_modules()

    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        available = sorted(FRAMEWORK_REGISTRY._registry.keys())
        raise NotImplementedError(
            f"Framework `{framework_id}` is not implemented. Available frameworks: {available}"
        )

    model_class = FRAMEWORK_REGISTRY[framework_id]
    return model_class(cfg)


# PreTrainedModel, AutoModel, PretrainedConfig,  are so good, find sometime to study them
# TODO @JinhuiYE find sometime to merge yaml config with transformer config


class baseframework(PreTrainedModel):
    """
    Lightweight base class for higher-level VLA model assemblies.
    Subclasses are expected to:
      - Accept a structured config
      - Register components in __init__
      - Use provided helpers for action normalization handling
    """

    def __init__(self, hf_config=PretrainedConfig()) -> None:
        """
        Initialize base nn.Module. Subclasses add components.
        """

        super().__init__(hf_config)

    # ------------------------------------------------------------------
    # Soft-constraint interface: subclasses should override these.
    # Default implementations raise NotImplementedError so that IDE
    # tooling (e.g. pylance, mypy) flags missing overrides, while
    # still allowing PreTrainedModel instantiation (no ABC).
    # ------------------------------------------------------------------

    def forward(self, examples: List[dict], **kwargs) -> dict:
        """Training forward pass.

        Args:
            examples: List[dict], each dict requires at least:
                - image: List[PIL.Image]
                - lang: str
                - action: np.ndarray shaped [T, action_dim]

        Returns:
            dict: Must contain ``"action_loss"`` (torch.Tensor scalar).
                  May contain extra keys for logging (e.g. ``"kl_loss"``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(examples) -> dict with 'action_loss' key."
        )

    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        """Inference: predict future actions from observations.

        Args:
            examples: Same schema as *forward* (minus ``action`` which is optional).
            **kwargs: Framework-specific inference options (e.g. ``use_ddim``).

        Returns:
            dict: Must contain ``"normalized_actions"`` (np.ndarray [B, T, action_dim]).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement predict_action(examples) -> dict with 'normalized_actions' key."
        )

    # ------------------------------------------------------------------
    # Unified loss interface for Trainer
    # ------------------------------------------------------------------

    def supports_training_tag(self, tag: str) -> bool:
        """Return whether this framework can consume batches for *tag*."""
        if tag == "vla":
            return type(self).forward is not baseframework.forward
        if tag == "vlm":
            return (
                hasattr(self, "qwen_vl_interface")
                or type(self).forward_vlm is not baseframework.forward_vlm
            )
        return False

    def compute_loss(
        self, tag: str, batch, loss_scale: dict = None
    ) -> Dict[str, torch.Tensor] | None:
        """Unified forward entry-point: route to the right forward by *tag*.

        The trainer calls ``model.compute_loss(tag, batch)`` for every
        ``(tag, batch)`` pair produced by :class:`DataLoaderManager`.
        The model internally dispatches:

        - ``"vla"`` → ``self.forward(batch)``
        - ``"vlm"`` → ``self.forward_vlm(batch)``

        Subclasses can override this to add more tags (e.g. ``"world"``).

        Args:
            tag: dataset type tag (``"vla"``, ``"vlm"``, …)
            batch: the batch produced by the corresponding DataLoader.
            loss_scale: ``{"vla": 1.0, "vlm": 0.1}`` per-tag loss multiplier.
                        Defaults to 1.0 for unspecified tags.

        Returns:
            dict[str, Tensor] | None: keyed losses (e.g. ``{"action_loss": ...}``).
                Returns ``None`` when this framework does not support the
                incoming dataloader tag so the trainer can ``continue``.
        """
        if not self.supports_training_tag(tag):
            return None

        scale = (loss_scale or {}).get(tag, 1.0)

        if tag == "vla":
            out = self.forward(batch)
        elif tag == "vlm":
            out = self.forward_vlm(batch)
        else:
            return None

        # Apply loss scale and filter to Tensor values only
        return {
            k: v * scale for k, v in out.items() if isinstance(v, torch.Tensor)
        }

    def forward_vlm(self, batch) -> Dict[str, torch.Tensor]:
        """VLM forward pass (default implementation).

        Delegates to ``self.qwen_vl_interface(**batch)`` which is present on
        every framework subclass that uses a Qwen VL backbone.

        Subclasses may override to add custom VLM logic.

        Args:
            batch: dict produced by the VLM dataloader.

        Returns:
            dict: Must contain ``"vlm_loss"`` (torch.Tensor scalar).
        """
        if not hasattr(self, "qwen_vl_interface"):
            raise NotImplementedError(
                f"{type(self).__name__} has no `qwen_vl_interface`. "
                "Override forward_vlm() to support VLM training."
            )
        out = self.qwen_vl_interface(**batch)
        return {"vlm_loss": out.loss}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: str,
        **kwargs,
    ) -> None:
        """
        Restore a model instance from a saved checkpoint.

        Workflow:
            1. Resolve checkpoint path
            2. Load config + dataset normalization statistics
            3. Build model with loaded config
            4. Load state_dict strictly (reports missing/unexpected keys)
            5. Attach normalization stats for later un-normalization

        Args:
            pretrained_checkpoint: Path to .pt file inside run/checkpoints directory.
            **kwargs: Extra constructor overrides passed to subclass.

        Returns:
            baseframework: Instantiated model (left on CPU; caller decides device).

        Raises:
            RuntimeError: If state_dict key mismatch occurs under strict=True.
            FileNotFoundError: If underlying files are missing (surfaced earlier).
        """
        pretrained_checkpoint = Path(pretrained_checkpoint)
        model_config, norm_stats = read_mode_config(
            pretrained_checkpoint
        )  # read config and norm_stats

        config = dict_to_namespace(model_config)
        model_config = config
        model_config.trainer.pretrained_checkpoint = None

        FrameworkModel = build_framework(cfg=model_config)
        # set for action un-norm
        FrameworkModel.norm_stats = norm_stats
        # Apply LoRA wrapper if the checkpoint was trained with LoRA enabled
        FrameworkModel.apply_lora(model_config)
        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        if pretrained_checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file

            model_state_dict = load_file(str(pretrained_checkpoint))
        else:
            model_state_dict = torch.load(
                pretrained_checkpoint, map_location="cpu"
            )
        # logger.info(f"Loading model weights from `{pretrained_checkpoint}`")
        model_keys = set(FrameworkModel.state_dict().keys())
        checkpoint_keys = set(model_state_dict.keys())
        try:
            FrameworkModel.load_state_dict(model_state_dict, strict=True)
        except RuntimeError as e:
            # must keep all keys matched
            common_keys = model_keys.intersection(checkpoint_keys)
            missing_keys = model_keys - common_keys
            unexpected_keys = checkpoint_keys - common_keys
            if missing_keys:
                logger.warning(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(
                    f"Unexpected keys in state_dict: {unexpected_keys}"
                )

            raise e

        # **ensure model is on GPU**
        FrameworkModel = FrameworkModel
        return FrameworkModel

    def apply_lora(self, cfg):
        lora_cfg = cfg.trainer.get("lora", None)
        if lora_cfg is None or not lora_cfg.get("enable", False):
            return

        from peft import LoraConfig, get_peft_model

        tm = lora_cfg.target_modules
        target_modules = tm if isinstance(tm, str) else list(tm)

        lora_config = LoraConfig(
            r=lora_cfg.rank,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=target_modules,
            bias=lora_cfg.bias,
        )

        if hasattr(self, "qwen_vl_interface"):
            target_name = "qwen_vl_interface"
            self.qwen_vl_interface.model = get_peft_model(
                self.qwen_vl_interface.model, lora_config
            )
            wrapped = self.qwen_vl_interface.model
        elif hasattr(self, "backbone") and hasattr(
            self.backbone, "transformer"
        ):
            target_name = "backbone.transformer"
            self.backbone.transformer = get_peft_model(
                self.backbone.transformer, lora_config
            )
            wrapped = self.backbone.transformer
        else:
            logger.warning(
                "[LoRA] no supported LoRA target found (expected qwen_vl_interface or backbone.transformer), skipping"
            )
            return

        trainable = sum(
            p.numel() for p in wrapped.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in wrapped.parameters())
        if trainable == 0:
            raise ValueError(
                f"[LoRA] target_modules={target_modules} matched nothing in {target_name}. "
                f"Check module naming (Qwen uses q_proj/v_proj; diffusers DiT uses to_q/to_v/to_out.0)."
            )
        logger.info(
            f"[LoRA] Applied: rank={lora_cfg.rank}, alpha={lora_cfg.alpha}, "
            f"targets={target_modules}, "
            f"trainable={trainable:,}/{total:,} ({100*trainable/total:.2f}%)"
        )

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Infer or validate the dataset stats key used for un-normalization.

        Args:
            norm_stats: Dict[str, dict] mapping dataset key -> stats block.
            unnorm_key: Optional explicit dataset key.

        Returns:
            str: Resolved key.

        Raises:
            AssertionError: If multiple datasets present and key not provided,
                            or provided key not found.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    @classmethod
    def get_action_stats(self, unnorm_key=None):
        """
        Retrieve raw action normalization statistics.

        Args:
            unnorm_key: Optional dataset stats key.

        Returns:
            dict: Stats structure (e.g. q01, q99, mask).
        """
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    @property
    def trainable_module_keys(self, max_depth=1) -> List[str]:
        """
        Enumerate trainable submodule names up to a depth.

        Args:
            max_depth: Descent depth when traversing module tree.

        Returns:
            List[str]: Module path names considered trainable.
        """
        keys = auto_get_trainable_modules(
            self, max_depth=max_depth
        )  # auto check which modules are trainable
        return keys

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Map normalized actions (≈[-1, 1]) back to original value range.

        Steps:
            - Clamp values to [-1, 1]
            - Threshold channel index 6 to {0,1} (binary semantic)
            - Apply linear scaling for masked dimensions using:
                original = 0.5 * (norm + 1) * (q99 - q01) + q01

        Args:
            normalized_actions: Array shape [T, D] (or chunk length × action_dim).
            action_norm_stats: Dict containing:
                q01 (array-like): Lower percentile (per-dimension).
                q99 (array-like): Upper percentile (per-dimension).
                mask (optional bool array): True => apply de-normalization; False => keep original normalized value.

        Returns:
            np.ndarray: Unnormalized actions (same shape as input).
        """
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(
            action_norm_stats["q01"]
        )
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(
            normalized_actions[:, 6] < 0.5, 0, 1
        )
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low)
            + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    @classmethod
    def get_action_stats(self, unnorm_key=None, norm_stats=None):
        """
        Duplicate stats accessor (retained for backward compatibility).
        # in future, it will own to policy interface and pack as
        """
        if norm_stats == None:
            norm_stats = self.norm_stats
        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]
