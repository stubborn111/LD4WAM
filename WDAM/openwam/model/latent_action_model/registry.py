"""Registry + factory for online Latent Action Models.

Mirrors the video-encoder / video-backbone registry pattern
(``openwam/model/video_backbone/encoder/registry.py``): a name -> class map,
a decorator to register, and a ``build_*`` factory that reads a config block.
Concrete adapters self-register on import (see this package's ``__init__``).
"""

from __future__ import annotations

from typing import Type

from openwam.model.latent_action_model.base import LatentActionModel

_LAM_REGISTRY: dict[str, Type[LatentActionModel]] = {}


def register_latent_action_model(name: str):
    def _wrap(cls: Type[LatentActionModel]) -> Type[LatentActionModel]:
        if name in _LAM_REGISTRY and _LAM_REGISTRY[name] is not cls:
            raise KeyError(f"latent_action_model '{name}' already registered.")
        _LAM_REGISTRY[name] = cls
        return cls

    return _wrap


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_latent_action_model(cfg) -> LatentActionModel:
    """Build a frozen LAM from the ``latent_action_model`` config block.

    Requires ``name`` (registry key). Everything else is adapter-specific and
    read by that adapter's ``from_config``.
    """
    name = _cfg_get(cfg, "name")
    if not name:
        raise ValueError("latent_action_model.name is required to build a LAM.")
    if name not in _LAM_REGISTRY:
        available = ", ".join(sorted(_LAM_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown latent_action_model '{name}'. Available: {available}")
    return _LAM_REGISTRY[name].from_config(cfg)


__all__ = ["register_latent_action_model", "build_latent_action_model"]
