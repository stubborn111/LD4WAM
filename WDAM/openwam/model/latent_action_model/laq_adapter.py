"""LAQ adapter for the online LAM port.

Wraps a LAQ (latent action quantization) model behind :class:`LatentActionModel`
WITHOUT importing LAQ at module load — the implementation lives in a separate
repo (e.g. ``/path/to/laq/laq_infere``) that evolves on its own.
The LAQ package is imported dynamically from a config-supplied ``repo_path`` only
when the adapter is constructed, so OpenWAM keeps no static dependency on it.

Call contract (verified against laq_infere):
  - build:  ``LAQ(**model_cfg)`` then ``load_weights(ckpt, strict=False)``
  - encode: ``inference(videos=(B,T,3,224,224) float[0,1], soft_quantize=True,
            return_quantized_actions=True, return_reconstructions=False)``
            -> ``{"quantized_actions": (B, T-1, Hq, Wq, quant_dim)}``.
  We flatten the ``(Hq, Wq, quant_dim)`` grid to ``la_dim`` (e.g. 4*4*32 = 512).
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from openwam.model.latent_action_model.base import LatentActionModel
from openwam.model.latent_action_model.registry import register_latent_action_model

logger = logging.getLogger(__name__)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _import_from(spec: str, repo_path: str):
    """Import ``module.path:attr`` after putting ``repo_path`` on sys.path.

    Kept fully dynamic so a change in the LAM codebase never breaks OpenWAM's
    import graph — the LAQ modules are only touched here, at build time.
    """
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, str(repo_path))
    if ":" in spec:
        mod_name, attr = spec.split(":", 1)
    else:
        mod_name, attr = spec.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


# DINOv3 ViT-encoder key prefixes that drift by one ``.model.`` segment across
# transformers versions (see ``_reconcile_dinov3_keys``).
_DINOV3_LAYER_SWAPS = (
    ("enc_spatial_transformer.model.layer.", "enc_spatial_transformer.model.model.layer."),
    ("enc_spatial_transformer.model.model.layer.", "enc_spatial_transformer.model.layer."),
)


def _unwrap_state_dict(payload):
    """Extract the raw state dict from the common LAQ checkpoint layouts.

    Mirrors ``LAQ.load_weights``: trainer save (``model``), DDP (``module``),
    lightning (``state_dict``), or a bare state dict.
    """
    if isinstance(payload, dict):
        for key in ("model", "module", "state_dict"):
            if key in payload:
                return payload[key]
    return payload


def _reconcile_dinov3_keys(model, state_dict: dict) -> tuple[dict, int]:
    """Bridge the HF DINOv3 module-nesting drift between transformers versions.

    transformers 5.x wraps the DINOv3 ViT encoder blocks in an extra
    ``DINOv3ViTEncoder`` submodule named ``.model``, so encoder-layer params live
    at ``enc_spatial_transformer.model.model.layer.N``. Checkpoints saved with
    transformers 4.56 kept them flat at ``enc_spatial_transformer.model.layer.N``.
    Same (frozen) weights, key paths differ by one ``.model.`` segment;
    ``embeddings`` / ``norm`` are unaffected.

    A source key is remapped ONLY when the remapped name exists in the model and
    the original does not — so this strictly improves the match, never breaks an
    already-aligned checkpoint, and no-ops in either transformers direction.
    """
    model_keys = set(model.state_dict().keys())
    remapped: dict = {}
    n = 0
    for k, v in state_dict.items():
        if k in model_keys:
            remapped[k] = v
            continue
        for src, dst in _DINOV3_LAYER_SWAPS:
            if src in k:
                cand = k.replace(src, dst)
                if cand in model_keys and cand not in state_dict:
                    remapped[cand] = v
                    n += 1
                    break
        else:
            remapped[k] = v
    return remapped, n


@register_latent_action_model("laq")
class LAQLatentActionModel(LatentActionModel):
    """Frozen LAQ, adapted to the online LAM port."""

    def __init__(
        self,
        laq_module: torch.nn.Module,
        *,
        la_dim: int,
        image_size: int,
        soft_quantize: bool,
    ):
        super().__init__()
        self.laq = laq_module
        self._la_dim = int(la_dim)
        self._image_size = int(image_size)
        self._soft_quantize = bool(soft_quantize)
        self.laq.eval()
        for p in self.laq.parameters():
            p.requires_grad_(False)

    # -- construction from config ------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "LAQLatentActionModel":
        repo_path = _cfg_get(cfg, "repo_path")
        if not repo_path:
            raise ValueError("latent_action_model.repo_path is required for the 'laq' adapter.")
        ckpt = _cfg_get(cfg, "ckpt")
        if not ckpt:
            raise ValueError("latent_action_model.ckpt is required for the 'laq' adapter.")

        class_import = _cfg_get(cfg, "class_import", "model.latent_action_quantization:LAQ")
        LAQ = _import_from(class_import, repo_path)

        # MODEL_CFG: inline dict wins; otherwise pull it from the repo (mirrors
        # what tools/export_robotwin_latent_actions.py does).
        model_cfg = _cfg_get(cfg, "model_cfg")
        if model_cfg is None:
            model_cfg_import = _cfg_get(cfg, "model_cfg_import", "configs.config_egodex:MODEL_CFG")
            model_cfg = dict(_import_from(model_cfg_import, repo_path))
        else:
            model_cfg = {k: v for k, v in dict(model_cfg).items()}

        laq = LAQ(**model_cfg)
        # Load the checkpoint ourselves (instead of LAQ.load_weights) so we can
        # reconcile the DINOv3 encoder key-nesting drift across transformers
        # versions before load_state_dict. The frozen DINOv3 backbone is already
        # loaded correctly from `dinov3-*-local` in LAQ.__init__, so this only
        # affects whether the checkpoint's (identical) copies land cleanly.
        if not Path(ckpt).exists():
            raise RuntimeError(f"latent_action_model: LAQ checkpoint not found: {ckpt}")
        state_dict = _unwrap_state_dict(torch.load(str(ckpt), map_location="cpu"))
        state_dict, n_remap = _reconcile_dinov3_keys(laq, state_dict)
        incompatible = laq.load_state_dict(state_dict, strict=False)
        miss, unexp = len(incompatible.missing_keys), len(incompatible.unexpected_keys)
        if n_remap:
            logger.info("Remapped %d DINOv3 encoder keys for transformers-version nesting drift.", n_remap)
        logger.info("Loaded LAQ weights from %s (%d missing / %d unexpected).", ckpt, miss, unexp)

        image_size = int(_cfg_get(cfg, "image_size", model_cfg.get("image_size", 224)))
        # la_dim: explicit, else derived from the LAQ grid (code_seq_len * quant_dim).
        la_dim = _cfg_get(cfg, "la_dim")
        if la_dim is None:
            la_dim = int(model_cfg.get("code_seq_len", 16)) * int(model_cfg.get("quant_dim", 32))
        soft_quantize = bool(_cfg_get(cfg, "soft_quantize", True))
        logger.info(
            "Built LAQ LAM from %s (ckpt=%s, image_size=%d, la_dim=%d, soft=%s)",
            repo_path, ckpt, image_size, la_dim, soft_quantize,
        )
        return cls(laq, la_dim=int(la_dim), image_size=image_size, soft_quantize=soft_quantize)

    # -- interface ---------------------------------------------------------
    @property
    def la_dim(self) -> int:
        return self._la_dim

    @property
    def image_size(self) -> int:
        return self._image_size

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) in [0,1] -> (B, T-1, la_dim) latent actions.

        Dtype is handled entirely INSIDE the LAM: the frames may arrive in any
        dtype (typically the caller's model dtype); the LAM upcasts to fp32 for
        its own forward (DINOv3 stability) and returns the target cast back to
        the *input* dtype. The caller therefore gets its own dtype back without
        doing any conversion itself.
        """
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError(f"LAM.encode expects (B, T, 3, H, W); got {tuple(frames.shape)}.")
        B, T, C, H, W = frames.shape
        if T < 2:
            raise ValueError(f"LAM.encode needs T>=2 frames to form a transition; got T={T}.")

        in_dtype = frames.dtype  # caller (model) dtype — the returned target uses this
        dev = frames.device
        # Run the LAM in its OWN parameter dtype (fp32 unless something external
        # cast it) and with autocast DISABLED, so neither a dtype mismatch nor a
        # silent autocast downcast can corrupt the (frozen) target. Frame prep
        # and the resize stay fp32; the input is cast to the LAM dtype last.
        try:
            p_dtype = next(self.laq.parameters()).dtype
        except StopIteration:
            p_dtype = torch.float32
        x = frames.reshape(B * T, C, H, W).float()
        if (H, W) != (self._image_size, self._image_size):
            x = F.interpolate(x, size=(self._image_size, self._image_size), mode="bilinear", align_corners=False)
        x = x.clamp(0.0, 1.0).reshape(B, T, C, self._image_size, self._image_size).to(device=dev, dtype=p_dtype)

        # Older LAQ builds took an explicit ``soft_quantize`` flag on inference();
        # newer ones (v10 final-multi) fix the quantization mode inside
        # SoftVQ.inference (the same path the offline export tool uses), so the
        # kwarg is gone. Pass it only when the signature still accepts it, so the
        # adapter works across LAQ versions without changing target semantics.
        import inspect as _inspect

        inf_kwargs = dict(
            return_reconstructions=False,
            return_quantized_actions=True,
            return_quantized_actions_idxs=False,
        )
        try:
            if "soft_quantize" in _inspect.signature(self.laq.inference).parameters:
                inf_kwargs["soft_quantize"] = self._soft_quantize
        except (TypeError, ValueError):
            pass

        autocast_dev = dev.type if dev.type in ("cuda", "cpu") else "cuda"
        with torch.autocast(device_type=autocast_dev, enabled=False):
            res = self.laq.inference(x, **inf_kwargs)
        qa = res["quantized_actions"]  # (B, T-1, Hq, Wq, quant_dim)
        la = qa.reshape(B, qa.shape[1], -1)  # (B, T-1, Hq*Wq*quant_dim)
        if la.shape[-1] != self._la_dim:
            raise ValueError(
                f"LAM produced la_dim={la.shape[-1]} but adapter expects {self._la_dim}. "
                "Set latent_action_model.la_dim to match (code_seq_len*quant_dim), and keep "
                "the expert's latent_action.la_dim equal to it."
            )
        return la.to(in_dtype)  # hand back in the caller's dtype (no conversion needed outside)


__all__ = ["LAQLatentActionModel"]
