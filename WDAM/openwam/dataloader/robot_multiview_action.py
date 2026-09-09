"""Midtrain robot reader: multi-camera (top + 2 wrists) L-shape video + REAL
action/proprio in a unified action space + offline latent action.

Stage-2 (midtrain) reader. Where single-view video pretraining feeds
a single TOP camera and no action, midtrain assembles a 3-view L-shape canvas,
emits REAL action + proprioception scattered into a shared unified space, and
still loads OFFLINE latent action (video + la experts keep training warm).

Everything is CONFIG-DRIVEN (no per-embodiment code). Per source, the YAML says:
  * ``action_cols`` / ``state_cols`` — which parquet columns (optionally sliced)
    concatenate, IN ORDER, into the NATIVE raw action / proprio vector. Each entry
    is either ``"col"`` (whole column) or ``["col", "a-b"]`` / ``["col", "n"]``
    (a closed-range / single-index slice). This is the ONLY place the raw layout
    is defined — agiworld's state(48) re-slice, dropping joint grips, etc. are all
    expressed here.
  * ``unify_action_map`` — paired ``[src, dst]`` map (base reader mechanism) that
    scatters the native raw into the unified space. It does the SCATTER + REORDER
    + rm1's rot6d convention fix (column-concat -> interleaved is just a dst
    permutation). ``state_cols`` must produce the SAME raw layout so the one map
    serves both action and proprio.
  * ``camera_layout`` — ``[top, left|__missing_left__, right|__missing_right__]``
    (base reader L-shape canvas; a ``__missing_*__`` slot stays black).
  * ``prompt_prefix`` — embodiment claim prepended to the task text.
  * ``load_la`` + ``la_*`` — offline LatentAction_v10 (8-token, same as pretrain).

Unified 34-D semantic layout the yaml maps target (one arm per half):
  L: pos[0:3] rot6d[3:9] grip[9] arm[10:17]  |  R: pos[17:20] rot6d[20:26] grip[26] arm[27:34]

Normalization (``normalize_mode=quantile``) runs on the native raw BEFORE the
scatter (base contract), using per-dataset ``meta/norm_stats.json``
(``midtrain_stats_computation`` — reads the SAME action_cols/state_cols). rot6d
dims are pinned to identity at stats time.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, List, Optional, Tuple

import numpy as np
import pandas as pd

from openwam.dataloader.bases.lerobot_v3_reader import LeRobotV3Reader
from openwam.dataloader.utils.normalization import apply_normalization, materialize_eef_stats
from openwam.dataloader.utils.offline_la import OFFLINE_LA_CONFIG_KEYS, OfflineLaMixin
from openwam.dataloader.utils.unify_action import parse_unify_spec

# Width of the midtrain unified action space (our own layout, NOT the base's global
# UNIFY_DIM=80). EEF + gripper only (joints dropped — all midtrain datasets have EEF):
#   L: pos[0:3] rot6d[3:9] grip[9]  |  R: pos[10:13] rot6d[13:19] grip[19]
# The reader rebuilds its unify scatter at this width after super().__init__ from the
# YAML unify_action_map; every dst must be < this.
MIDTRAIN_UNIFY_DIM = 20

# A parsed column-spec entry: (column_name, lo, hi) — lo/hi None -> whole column,
# else the CLOSED range [lo, hi] (a single index n -> lo == hi == n).
ColSlice = Tuple[str, Optional[int], Optional[int]]


def _coerce_list(spec):
    """OmegaConf ListConfig -> plain (possibly nested) list; no-op otherwise."""
    try:
        from omegaconf import ListConfig, OmegaConf

        if isinstance(spec, ListConfig):
            return OmegaConf.to_container(spec, resolve=True)
    except Exception:
        pass
    return spec


def parse_col_spec(spec) -> List[ColSlice]:
    """Parse an ``action_cols`` / ``state_cols`` yaml value into ``[(col, lo, hi)]``.

    Each entry is ``"col"`` (whole column) or ``["col", "a-b"]`` / ``["col", "n"]``
    (closed range / single index). Order is preserved (it defines the raw layout)."""
    spec = _coerce_list(spec)
    if not isinstance(spec, (list, tuple)) or len(spec) == 0:
        raise ValueError(f"col spec must be a non-empty list, got {spec!r}")
    out: List[ColSlice] = []
    for entry in spec:
        entry = _coerce_list(entry)
        if isinstance(entry, str):
            out.append((entry, None, None))
            continue
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"col spec entry must be 'col' or ['col','a-b'], got {entry!r}")
        col, rng = str(entry[0]), str(entry[1]).strip()
        if "-" in rng:
            a, b = rng.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                raise ValueError(f"col spec range {rng!r} has end < start")
        else:
            lo = hi = int(rng)
        out.append((col, lo, hi))
    return out


def col_spec_needed_cols(parsed: List[ColSlice]) -> Tuple[str, ...]:
    """Unique column names referenced by a parsed spec (for NEEDED_COLS)."""
    seen = []
    for col, _, _ in parsed:
        if col not in seen:
            seen.append(col)
    return tuple(seen)


def col_spec_width(parsed: List[ColSlice], feature_dims: dict) -> int:
    """Total width of the concatenated raw vector. ``feature_dims[col]`` = that
    column's full width (from info.json), needed for whole-column entries."""
    w = 0
    for col, lo, hi in parsed:
        if lo is None:
            if col not in feature_dims:
                raise KeyError(f"col spec: column {col!r} not in info.features")
            w += int(feature_dims[col])
        else:
            w += hi - lo + 1
    return w


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    """DataFrame column -> ``(T, D)`` float32 (stacks the per-row vectors)."""
    return np.stack(df[name].to_numpy()).astype(np.float32)


def assemble_from_spec(df: pd.DataFrame, parsed: List[ColSlice]) -> np.ndarray:
    """Concatenate the parsed column slices into a ``(T, width)`` native raw vector.
    Shared by the reader (per window) and the stats tool (per episode)."""
    parts = []
    for col, lo, hi in parsed:
        arr = _col(df, col)
        parts.append(arr if lo is None else arr[:, lo : hi + 1])
    return np.concatenate(parts, axis=-1)


class RobotMultiviewActionReader(OfflineLaMixin, LeRobotV3Reader):
    """Multi-camera robot reader with config-driven unified action/proprio + offline la."""

    DATASET_NAME: ClassVar[str] = "RobotMultiviewAction"
    PROMPT_SOURCE: ClassVar[str] = "episode_tasks"  # per-episode `tasks` column (like pretrain)
    STATS_FILENAME: ClassVar[str] = "norm_stats.json"
    DEFAULT_NORMALIZE_MODE: ClassVar[str] = "quantile"
    # A wrist mp4 that is absent / corrupt degrades to a black slot (missing wrist).
    WRIST_DECODE_TOLERATED: ClassVar[tuple] = (Exception,)

    # action_cols / state_cols (raw layout) + prompt_prefix + la knobs. camera_layout
    # / unify_action / unify_action_map / multiview come from the base CONFIG_KEYS
    # (all written in the YAML — the map, incl. rm1's rot6d permutation, lives there).
    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = (
        LeRobotV3Reader.CONFIG_KEYS + ("action_cols", "state_cols", "prompt_prefix") + OFFLINE_LA_CONFIG_KEYS
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        action_cols,
        state_cols,
        prompt_prefix: Optional[str] = None,
        load_la: bool = False,
        la_root: Optional[str] = None,
        la_file_template: str = "episode_{episode:06d}.pt",
        la_key: str = "latent_action",
        la_dim: int = 512,
        la_per_frame: int = 4,
        la_first_frame_is_pad: bool = False,
        temporal_compression: int = 4,
        **kwargs: Any,
    ):
        self._action_spec = parse_col_spec(action_cols)
        self._state_spec = parse_col_spec(state_cols)
        self._prompt_prefix = str(prompt_prefix) if prompt_prefix else ""

        # Feature widths from info.json (needed for whole-column entries), read here
        # because the base needs ACTION_DIM during super().__init__.
        info_path = f"{dataset_dir}/meta/info.json"
        with open(info_path) as f:
            feats = json.load(f).get("features", {})
        feat_dims = {c: int(v["shape"][0]) for c, v in feats.items() if v.get("shape")}
        a_w = col_spec_width(self._action_spec, feat_dims)
        s_w = col_spec_width(self._state_spec, feat_dims)
        if a_w != s_w:
            raise ValueError(
                f"RobotMultiviewAction({dataset_dir}): action_cols width {a_w} != state_cols width {s_w}; "
                f"both must produce the same native raw layout (one unify_action_map serves both)."
            )
        self.ACTION_DIM = a_w
        self._stats_dim = a_w
        # Parquet columns to load = union of action + state columns.
        self.NEEDED_COLS = tuple(
            dict.fromkeys(col_spec_needed_cols(self._action_spec) + col_spec_needed_cols(self._state_spec))
        )

        # unify_action + map come from the YAML (explicit, like robocoin/behavior).
        if not kwargs.get("unify_action"):
            raise ValueError(
                f"RobotMultiviewAction({dataset_dir}): set `unify_action: true` in the dataloader yaml."
            )
        self._unify_map_spec = kwargs.get("unify_action_map")
        if self._unify_map_spec is None:
            raise ValueError(
                f"RobotMultiviewAction({dataset_dir}): unify_action_map is required in the dataloader yaml "
                f"(paired src->dst map into the {MIDTRAIN_UNIFY_DIM}-D unified space)."
            )
        super().__init__(dataset_dir, **kwargs)

        # Rebuild the unify scatter at OUR width (34) instead of the base's global
        # UNIFY_DIM (80) — incremental override, base untouched. Built from the SAME
        # yaml map; nothing has consumed the base's 80-D version yet (construction only).
        self._unify_dim = MIDTRAIN_UNIFY_DIM
        self.ACTION_DIM = MIDTRAIN_UNIFY_DIM
        self._unify_dst_index = parse_unify_spec(self._unify_map_spec, MIDTRAIN_UNIFY_DIM)
        self._unify_dim_mask = np.zeros(MIDTRAIN_UNIFY_DIM, dtype=bool)
        self._unify_dim_mask[self._unify_dst_index] = True

        self._init_offline_la(
            load_la=load_la,
            la_root=la_root,
            la_file_template=la_file_template,
            la_key=la_key,
            la_dim=la_dim,
            la_per_frame=la_per_frame,
            la_first_frame_is_pad=la_first_frame_is_pad,
            temporal_compression=temporal_compression,
        )

    # ── cameras: parse the base's camera_layout [top, left|__missing__, right|__missing__] ──
    def _resolve_cameras(self, info: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        layout = self._camera_layout_param
        if not layout:
            raise ValueError(
                f"RobotMultiviewAction({self._dataset_id}): camera_layout is required, e.g. "
                f"[observation.images.camera_top, observation.images.camera_wrist_left, "
                f"observation.images.camera_wrist_right] (use __missing_left__/__missing_right__ for absent wrists)."
            )
        features = info.get("features", {})

        def _real(name):
            s = str(name)
            return None if s.startswith("__missing") else s

        head = _real(layout[0])
        left = _real(layout[1]) if len(layout) > 1 else None
        right = _real(layout[2]) if len(layout) > 2 else None
        if head is None or head not in features:
            raise ValueError(
                f"RobotMultiviewAction({self._dataset_id}): top camera {head!r} not in info.features "
                f"({[k for k in features if 'image' in k]})."
            )
        # A named-but-absent wrist degrades to a black slot (tolerated).
        left = left if (left is None or left in features) else None
        right = right if (right is None or right in features) else None
        return head, left, right

    # ── stats: flat norm_stats.json at the native raw width ───────────────────
    def _load_stats(self, info: dict) -> Optional[dict]:
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        stats_path = self._dataset_dir / "meta" / self.STATS_FILENAME
        if not stats_path.exists():
            raise FileNotFoundError(
                f"RobotMultiviewAction({self._dataset_id}): normalize_mode={self._normalize_mode!r} but "
                f"{stats_path} is missing. Run: python -m "
                f"openwam.dataloader.utils.stats_computation.midtrain_stats_computation "
                f"--dataloader_config <this yaml>  (or set normalize_mode=null)."
            )
        with open(stats_path) as f:
            raw = json.load(f)
        return materialize_eef_stats(
            raw, self._normalize_mode, dim=self._stats_dim, strict_minmax=False, source_hint=str(stats_path)
        )

    # ── prompt: per-episode `tasks` column + embodiment prefix ─────────────────
    def _load_prompts(self) -> None:
        if self.PROMPT_SOURCE == "episode_tasks":
            return
        super()._load_prompts()

    def _resolve_prompt(self, row, win) -> str:
        if self.PROMPT_SOURCE != "episode_tasks":
            return super()._resolve_prompt(row, win)
        tasks = row["tasks"] if "tasks" in row.index else None
        if tasks is None:
            raise KeyError(f"{self.DATASET_NAME}({self._dataset_id}): episodes parquet has no 'tasks' column.")
        text = tasks if isinstance(tasks, str) else (str(list(tasks)[0]) if len(tasks) else "")
        text = text.strip()
        if not text:
            raise ValueError(
                f"{self.DATASET_NAME}({self._dataset_id}): empty task text for "
                f"episode_index={int(row['episode_index'])}."
            )
        return self._prompt_prefix + text

    # ── native raw action / proprio (normalized, pre-scatter) ─────────────────
    def _action_20d(self, win: pd.DataFrame) -> Optional[np.ndarray]:
        raw = assemble_from_spec(win, self._action_spec)  # (T, native_dim)
        return apply_normalization(raw, self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win: pd.DataFrame) -> Optional[np.ndarray]:
        raw = assemble_from_spec(win, self._state_spec)[0:1]  # proprio at window start
        return apply_normalization(raw, self._normalization_stats, self._normalize_mode)

    # ── offline la attach ─────────────────────────────────────────────────────
    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        return self._maybe_attach_la(sample, idx)

    @classmethod
    def from_config(cls, config, split: str = "train") -> "LeRobotV3Reader":
        return super().from_config(config, split)


__all__ = [
    "RobotMultiviewActionReader",
    "MIDTRAIN_UNIFY_DIM",
    "parse_col_spec",
    "assemble_from_spec",
    "col_spec_needed_cols",
    "col_spec_width",
]
