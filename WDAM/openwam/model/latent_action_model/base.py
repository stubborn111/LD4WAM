"""Port for an online Latent Action Model (LAM).

The ``la_tri_system`` framework supervises its LatentActionExpert with a
frame-to-frame "latent action" target. Originally that target was precomputed
offline and read from the dataloader; this port lets the architecture instead
hold a *frozen* LAM and compute the target ONLINE — the same way the video
backbone holds a VAE and encodes RGB frames to latents on the fly.

Design intent (why this is an interface, not a direct import):

- OpenWAM must NOT import the concrete LAM implementation (e.g. LAQ) directly,
  because that codebase evolves independently. Concrete adapters live behind
  this interface and are constructed via
  :func:`openwam.model.latent_action_model.build_latent_action_model`, which
  dynamically imports the implementation from a configurable ``repo_path``.
- The only surface the architecture depends on is :class:`LatentActionModel`
  below: ``encode(frames) -> latent_actions`` plus ``la_dim`` / ``image_size``.

The LAM is always frozen and inference-only (``eval`` + ``requires_grad_(False)``);
it is loaded from its own checkpoint and is excluded from the OpenWAM training
optimizer and from saved architecture checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentActionModel(nn.Module):
    """Frozen online LAM: turns a video clip into per-transition latent actions.

    Concrete subclasses adapt a specific model (e.g. LAQ) to this contract. They
    are registered with :func:`register_latent_action_model` and built from a
    config block via :func:`build_latent_action_model` — never imported into
    OpenWAM at module load time.
    """

    def train(self, mode: bool = True):
        """Frozen, inference-only: never enter train mode.

        The LAM is a supervision *oracle* — its output must be deterministic. If
        a parent ``arch.train()`` recursed into it, LAQ's dropout / DINOv3 would
        activate and the la target would jitter step to step. Pin it to eval so
        that can never happen, regardless of how the parent toggles modes.
        """
        return super().train(False)

    @classmethod
    def from_config(cls, cfg) -> "LatentActionModel":
        """Build an instance from the ``latent_action_model`` config block.

        Subclasses override this to read their own keys (repo_path, ckpt, ...).
        """
        raise NotImplementedError

    @property
    def la_dim(self) -> int:
        """Dimension of one latent action (the flattened per-transition code)."""
        raise NotImplementedError

    @property
    def image_size(self) -> int:
        """Square input resolution the LAM expects (frames are resized to this)."""
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode a clip to per-adjacent-pair latent actions.

        Args:
            frames: ``(B, T, 3, H, W)`` RGB in ``[0, 1]``. Any ``H/W`` — the
                adapter resizes to :attr:`image_size` and applies whatever
                normalization the underlying model needs.

        Returns:
            ``(B, T-1, la_dim)`` latent actions (one per adjacent frame pair;
            ``T`` frames → ``T-1`` actions). Returned in the SAME dtype as
            ``frames`` — the implementation may upcast to fp32 internally for its
            own forward, but converts the target back to the caller's dtype so no
            dtype conversion is needed outside the LAM.
        """
        raise NotImplementedError


__all__ = ["LatentActionModel"]
