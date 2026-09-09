"""Shared offline latent-action (la) loading — used by both the pretrain video
reader and the midtrain robot reader.

Offline la means a precomputed per-episode latent-action tensor (from a trained
LAQ/LAM, e.g. ``LatentAction_v10``) is loaded per window and emitted as
``la_gt`` / ``la_gt_mask``, instead of running the LAM online on raw frames.
Pair this with a model config that has NO ``latent_action_model`` block (the LAM
is then never built and the architecture reads ``la_gt`` from the batch).

The token layout produced here matches the ONLINE LAM output exactly
(``T_video-1`` tokens = ``(Fv_latent-1) * k_la``): latent frame 0 is the
conditioning frame (given, no la), so only the predicted latent frames carry la.
This is the single source of truth for that layout so pretrain and midtrain can
never drift from each other or from inference.

Usage — mix into a :class:`LeRobotV3Reader` subclass::

    class MyReader(OfflineLaMixin, LeRobotV3Reader):
        def __init__(self, dataset_dir, *, <la knobs...>, **kwargs):
            super().__init__(dataset_dir, **kwargs)
            self._init_offline_la(load_la=..., la_root=..., ...)
        def __getitem__(self, idx):
            sample = super().__getitem__(idx)
            return self._maybe_attach_la(sample, idx)
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# Wan causal VAE temporal downsample (latent frames = (video_frames - 1)//4 + 1).
_WAN_VAE_TEMPORAL_DOWNSAMPLE = 4

# yaml keys the readers forward to _init_offline_la (extend the reader CONFIG_KEYS).
OFFLINE_LA_CONFIG_KEYS: Tuple[str, ...] = (
    "load_la",
    "la_root",
    "la_file_template",
    "la_key",
    "la_dim",
    "la_per_frame",
    "la_first_frame_is_pad",
    "temporal_compression",
)


# Process-global LRU for decoded per-episode latent-action tensors, shared across
# all bucket instances — mirrors LeRobotV3Reader._read_data_table_cached. A
# per-instance dict is a memory bomb under shuffled training: ~1.6M episodes,
# one copy per bucket per DataLoader worker, grows unbounded while the random
# access pattern gives ~0 hit rate. A small global cache keeps the sequential
# windows WITHIN one episode fast at a fixed ceiling (~64 x <1 MB tensors/worker).
@functools.lru_cache(maxsize=64)
def load_episode_la_cached(path: str, la_key: str, la_dim: int) -> torch.Tensor:
    """Load + validate a per-episode latent-action tensor ``[N, la_dim]`` (cached).

    Keyed by (path, la_key, la_dim) so the cache is content-identity based and
    shared process-wide. The returned tensor is treated as READ-ONLY by callers
    (never mutated in place), so sharing it across windows/instances is safe.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if la_key not in obj:
            raise KeyError(
                f"offline_la: la file {path} is a dict without key {la_key!r}. Keys: {list(obj.keys())}."
            )
        la = obj[la_key]
    else:
        la = obj
    if not torch.is_tensor(la):
        la = torch.from_numpy(np.asarray(la))
    la = la.float()
    if la.ndim != 2 or la.shape[-1] != la_dim:
        raise ValueError(
            f"offline_la: la file {path} must hold [N, la_dim={la_dim}], got shape {tuple(la.shape)}."
        )
    return la


class OfflineLaMixin:
    """Offline la loading for a LeRobotV3Reader subclass.

    Depends on base-reader instance attributes: ``_dataset_dir``, ``_cum_n_starts``,
    ``_window_stride``, ``_eps_df``, ``_num_frames``, ``_video_sample_indices``.
    """

    def _init_offline_la(
        self,
        *,
        load_la: bool = False,
        la_root=None,
        la_file_template: str = "episode_{episode:06d}.pt",
        la_key: str = "latent_action",
        la_dim: int = 512,
        la_per_frame: int = 4,
        la_first_frame_is_pad: bool = False,
        temporal_compression: int = _WAN_VAE_TEMPORAL_DOWNSAMPLE,
    ) -> None:
        self._load_la = bool(load_la)
        self._la_root = Path(la_root) if la_root else (self._dataset_dir / "LatentAction_v10")
        self._la_file_template = str(la_file_template)
        self._la_key = str(la_key)
        self._la_dim = int(la_dim)
        self._la_per_frame = int(la_per_frame)
        self._la_first_frame_is_pad = bool(la_first_frame_is_pad)
        self._la_temporal_compression = int(temporal_compression)
        if self._la_temporal_compression < 1:
            raise ValueError(f"temporal_compression must be >= 1, got {self._la_temporal_compression}")

    # --- window bookkeeping (mirror of base _getitem_impl) --------------------
    def _window_bounds(self, idx: int) -> Tuple[int, int, int]:
        ep_local = int(np.searchsorted(self._cum_n_starts, idx, side="right") - 1)
        offset = (idx - int(self._cum_n_starts[ep_local])) * self._window_stride
        ep_len = int(self._eps_df.iloc[ep_local]["length"])
        actual_raw_len = min(self._num_frames, ep_len - offset)
        return ep_local, offset, actual_raw_len

    def _episode_index_for(self, ep_local: int) -> int:
        row = self._eps_df.iloc[ep_local]
        if "episode_index" in self._eps_df.columns:
            return int(row["episode_index"])
        return int(ep_local)

    def _load_episode_la(self, ep_index: int) -> torch.Tensor:
        path = self._la_root / self._la_file_template.format(episode=ep_index)
        if not path.exists():
            raise FileNotFoundError(
                f"offline_la: latent-action file not found: {path} "
                f"(la_root={self._la_root}, template={self._la_file_template!r}). "
                "Set load_la=false for the online-LAM mode, or fix la_root/la_file_template."
            )
        return load_episode_la_cached(str(path), self._la_key, self._la_dim)

    def _load_la_window(self, ep_local: int, offset: int, actual_raw_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(la_gt [la_dim, n_pred, k_la, 1], mask [n_pred*k_la])`` for this
        window, matching the online-LAM layout (``T_video-1`` tokens).

        The video window has ``fv_latent_full`` video-latent frames; latent frame 0
        is the conditioning frame (given, no la), so only ``n_pred = fv_latent_full-1``
        frames are predicted → ``n_pred * k_la`` la tokens. We drop ``la[offset]``
        (the transition INTO the conditioning frame) and take the within-window
        transitions ``la[offset+1 : offset+num_video_frames]``. NO zero prefix — this
        mirrors the online LAM output and the model's ``num_la_tokens``.
        """
        ep_index = self._episode_index_for(ep_local)
        la_ep = self._load_episode_la(ep_index)  # [N, la_dim]

        tc = self._la_temporal_compression
        k = self._la_per_frame
        fv_video_full = int(len(self._video_sample_indices))
        fv_video_real = int((self._video_sample_indices < actual_raw_len).sum())
        n_pred = max((fv_video_full - 1) // tc + 1 - 1, 0)  # predicted latent frames
        n_pred_real = max((fv_video_real - 1) // tc + 1 - 1, 0)
        target = n_pred * k

        # Within-window transitions: drop la[offset] (into the conditioning frame).
        start = offset + 1 if self._la_first_frame_is_pad else offset
        la = la_ep[start : offset + actual_raw_len]
        n_real = int(la.shape[0])

        if n_real < target:
            # Short window: repeat the last real transition (mirrors the video reader
            # repeating the last real frame); the repeated tail is masked below.
            fill = la[-1:] if n_real > 0 else la_ep.new_zeros(1, la_ep.shape[-1])
            la = torch.cat([la, fill.repeat(target - n_real, 1)], dim=0)
        else:
            la = la[:target]
        la_gt = la.reshape(n_pred, k, la.shape[-1]).permute(2, 0, 1).unsqueeze(-1).contiguous()
        mask = torch.ones(target, dtype=torch.bool)
        n_valid = min(n_pred_real * k, target)
        if n_valid < target:  # short window: mask the repeated (non-real) tail
            mask[n_valid:] = False
        return la_gt, mask

    def _maybe_attach_la(self, sample: dict, idx: int) -> dict:
        """Attach ``la_gt`` / ``la_gt_mask`` to a sample when offline la is enabled."""
        if getattr(self, "_load_la", False):
            ep_local, offset, actual_raw_len = self._window_bounds(idx)
            la_gt, la_gt_mask = self._load_la_window(ep_local, offset, actual_raw_len)
            sample["la_gt"] = la_gt
            sample["la_gt_mask"] = la_gt_mask
        return sample


__all__ = ["OfflineLaMixin", "load_episode_la_cached", "OFFLINE_LA_CONFIG_KEYS", "_WAN_VAE_TEMPORAL_DOWNSAMPLE"]
