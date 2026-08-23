import json
import logging
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder
from torchvision.transforms import v2 as T

from ldm.utils import pair

logger = logging.getLogger(__name__)


class LeRobotMixedWindowDataset(Dataset):
    

    def __init__(
        self,
        dataset_cfg: Dict[str, Any],
        image_size: int = 224,
        max_frames: int = 8,
        seed: int = 2026,
        action_dim: int = 14,
        split: str = "train",
        epoch_size: int = 100_000,
        action_sample_ratio: float = 0.3,
        action_zero_eps: float = 1e-6,
        deterministic_val: bool = False,
        val_windows_per_dataset: int = 256,
        batch_level_same_episode: bool = False,
        local_batch_size: int = 1,
        batch_start_jitter: int = 0,
        frame_strides: Optional[List[int]] = None,
        frame_stride_weights: Optional[List[float]] = None,
        rotation_accumulate: str = "sum",
        skip_build: bool = False,
    ):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.image_size = pair(image_size)
        self.max_frames = int(max_frames)
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.action_dim = int(action_dim)
        self.split = split
        self.epoch_size = int(epoch_size)
        self.action_sample_ratio = float(action_sample_ratio)
        self.action_zero_eps = float(action_zero_eps)
        frame_strides = list(frame_strides) if frame_strides else [1]
        self.frame_strides = [int(s) for s in frame_strides]
        if any(s < 1 for s in self.frame_strides):
            raise ValueError(f"frame_strides must all be >= 1, got {self.frame_strides}")
        if frame_stride_weights is None:
            frame_stride_weights = [1.0] * len(self.frame_strides)
        if len(frame_stride_weights) != len(self.frame_strides):
            raise ValueError(
                f"frame_stride_weights length {len(frame_stride_weights)} "
                f"must match frame_strides length {len(self.frame_strides)}"
            )
        self.frame_stride_weights = self._norm([float(w) for w in frame_stride_weights])
        if rotation_accumulate not in ("sum", "chain"):
            raise ValueError("rotation_accumulate must be 'sum' or 'chain'")
        if rotation_accumulate == "chain":
            raise NotImplementedError(
                "rotation_accumulate='chain' is reserved for future strict rotation "
                "composition; use 'sum' for now."
            )
        self.rotation_accumulate = rotation_accumulate
        self._trans_dims = [0, 1, 2, 7, 8, 9]
        self._rot_dims = [3, 4, 5, 10, 11, 12]
        self._grip_dims = [6, 13]
        self.deterministic_val = bool(deterministic_val)
        self.val_windows_per_dataset = int(val_windows_per_dataset)
        self.batch_level_same_episode = bool(batch_level_same_episode) and not self.deterministic_val
        self.local_batch_size = max(1, int(local_batch_size))
        self.batch_start_jitter = max(0, int(batch_start_jitter))
        self._effective_epoch_size = max(1, self.epoch_size // self.local_batch_size) if self.batch_level_same_episode else self.epoch_size
        self._dist_rank = int(
            os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or "0"
        )
        self._parquet_cache: Dict[str, Any] = {}
        self._video_decoder_cache: Dict[str, Any] = {}
        self._video_decoder_cache_size = int(
            self.dataset_cfg.get("video_decoder_cache_size", 4)
        )

        self.transform = T.Compose(
            [
                T.Lambda(
                    lambda img: (
                        img.convert("RGB")
                        if isinstance(img, Image.Image) and img.mode != "RGB"
                        else img
                    )
                ),
                T.ToImage(),
                T.Resize(self.image_size),
                T.ToDtype(torch.float32, scale=True),
            ]
        )

        self.recon_eps: Dict[str, List[Dict[str, Any]]] = {}
        self.align_eps: Dict[str, List[Dict[str, Any]]] = {}
        self.recon_ep_weights: Dict[str, List[float]] = {}
        self.align_ep_weights: Dict[str, List[float]] = {}
        self.recon_names, self.align_names = [], []
        self.recon_weights, self.align_weights = [], []
        self.val_records: List[Tuple[str, Dict[str, Any], int, int]] = []

        if skip_build:
            return

        for name, cfg in dataset_cfg.items():
            if cfg.get("format") != "lerobot":
                continue
            eps = self._load_episodes(name, cfg)
            self.recon_eps[name] = eps
            self.recon_ep_weights[name] = [max(1, int(e.get("num_starts", 1))) for e in eps]
            align_eps = [e for e in eps if e.get("use_align")]
            self.align_eps[name] = align_eps
            self.align_ep_weights[name] = [max(1, int(e.get("num_align_starts", e.get("num_starts", 1)))) for e in align_eps]
            if eps and cfg.get("use_for_reconstruction", True):
                self.recon_names.append(name)
                self.recon_weights.append(float(cfg.get("sample_weight", 1.0)))
            if align_eps and cfg.get("use_for_action_alignment", False):
                self.align_names.append(name)
                self.align_weights.append(
                    float(
                        cfg.get(
                            "alignment_sample_weight", cfg.get("sample_weight", 1.0)
                        )
                    )
                )
            if self.deterministic_val:
                self._append_val_records(name, eps)
            recon_windows = int(sum(self.recon_ep_weights[name]))
            align_windows = int(sum(self.align_ep_weights[name])) if align_eps else 0
            logger.info(
                "LeRobot %s: episodes=%d recon_windows=%d align_episodes=%d align_windows=%d sample_weight=%.3f alignment_sample_weight=%.3f",
                name,
                len(eps),
                recon_windows,
                len(align_eps),
                align_windows,
                float(cfg.get("sample_weight", 1.0)),
                float(cfg.get("alignment_sample_weight", cfg.get("sample_weight", 1.0))),
            )

        if not self.recon_names:
            raise ValueError("No LeRobot reconstruction episodes found.")
        self.recon_weights = self._norm(self.recon_weights)
        self.align_weights = (
            self._norm(self.align_weights) if self.align_weights else []
        )
        if self.deterministic_val and self.val_records:
            self.epoch_size = len(self.val_records)
            self._effective_epoch_size = self.epoch_size
        logger.info(
            "LeRobot split=%s batch_level_same_episode=%s local_batch_size=%d batch_start_jitter=%d epoch_size=%d effective_len=%d action_sample_ratio=%.3f frame_strides=%s frame_stride_weights=%s rotation_accumulate=%s",
            self.split,
            self.batch_level_same_episode,
            self.local_batch_size,
            self.batch_start_jitter,
            self.epoch_size,
            self._effective_epoch_size,
            self.action_sample_ratio,
            self.frame_strides,
            [round(w, 3) for w in self.frame_stride_weights],
            self.rotation_accumulate,
        )

    def __len__(self):
        return self._effective_epoch_size

    @staticmethod
    def _norm(xs: List[float]) -> List[float]:
        s = sum(xs)
        return [x / s for x in xs] if s > 0 else [1.0 / len(xs)] * len(xs)

    def _clip_span(self, stride: int) -> int:
        
        return (self.max_frames - 1) * int(stride) + 1

    def _sample_stride(self) -> int:
        return int(random.choices(self.frame_strides, weights=self.frame_stride_weights, k=1)[0])

    def _wait_for_cache(
        self, path: Path, what: str, timeout: float = 21_600.0, poll: float = 15.0
    ) -> bool:
        
        if self._dist_rank == 0 or path.exists():
            return path.exists()
        waited = 0.0
        logger.info(
            "rank %d waiting for %s cache built by rank 0: %s",
            self._dist_rank,
            what,
            path,
        )
        while not path.exists():
            time.sleep(poll)
            waited += poll
            if waited >= timeout:
                logger.warning(
                    "rank %d timed out (%.0fs) waiting for %s cache %s; building locally",
                    self._dist_rank,
                    timeout,
                    what,
                    path,
                )
                return False
        logger.info("rank %d found %s cache after %.0fs: %s", self._dist_rank, what, waited, path)
        return True

    def _load_episodes(
        self, name: str, cfg: Dict[str, Any], defer_valid_starts: bool = False
    ) -> List[Dict[str, Any]]:
        import pandas as pd

        root = Path(cfg["root_dir"])
        video_key = cfg["video_key"]
        action_key = cfg.get("action_key")
        data_type = cfg.get("data_type", "human")
        stride = int(cfg.get("window_stride", 1))
        use_align = (
            bool(cfg.get("use_for_action_alignment", False)) and action_key is not None
        )
        human_all_dims = bool(cfg.get("human_require_all_action_dims_nonzero", False))
        repair_data_file_index = bool(cfg.get("repair_data_file_index", False))
        repair_cache_path = cfg.get("repair_data_file_index_cache")
        info_path = root / "meta" / "info.json"
        if info_path.exists():
            with open(info_path, "r") as f:
                fps = float(json.load(f).get("fps", cfg.get("fps", 30)))
        else:
            fps = float(cfg.get("fps", 30))
        episodes: List[Dict[str, Any]] = []
        missing_data_files = Counter()
        missing_video_files = Counter()
        repaired_data_files = Counter()
        length_mismatches = Counter()
        if repair_data_file_index:
            map_cache = (
                Path(repair_cache_path)
                if repair_cache_path
                else root / "meta" / "ldm_episode_data_file_map.json"
            )
            if self._dist_rank > 0 and not map_cache.exists():
                self._wait_for_cache(map_cache, f"{name} data-file-map")
            episode_data_file_map, episode_frame_count_map = (
                self._build_episode_data_file_map(root, repair_cache_path)
            )
        else:
            episode_data_file_map, episode_frame_count_map = ({}, {})

        build_valid_cache = use_align and data_type == "human"
        valid_starts_cache = None
        if build_valid_cache:
            valid_starts_cache = self._load_valid_starts_cache(
                root, action_key, human_all_dims
            )
            if not valid_starts_cache and self._dist_rank > 0:
                vs_path = self._valid_starts_cache_path(root, action_key)
                if self._wait_for_cache(vs_path, f"{name} valid-starts"):
                    valid_starts_cache = self._load_valid_starts_cache(
                        root, action_key, human_all_dims
                    )
            if valid_starts_cache is None:
                valid_starts_cache = {}
        valid_starts_cache_dirty = False
        for ep_file in sorted(
            (root / "meta" / "episodes").glob("chunk-*/file-*.parquet")
        ):
            ep_df = pd.read_parquet(ep_file)
            for _, row in ep_df.iterrows():
                length = int(row["length"])
                if length < self._clip_span(min(self.frame_strides)):
                    continue
                data_chunk = int(
                    row.get("data/chunk_index", row.get("meta/episodes/chunk_index", 0))
                )
                data_file_idx = int(row.get("data/file_index", 0))
                video_chunk = int(
                    row.get(f"videos/{video_key}/chunk_index", data_chunk)
                )
                video_file_idx = int(
                    row.get(f"videos/{video_key}/file_index", data_file_idx)
                )
                data_start = int(row.get("dataset_from_index", 0))
                ts0 = float(row.get(f"videos/{video_key}/from_timestamp", 0.0))
                data_path = (
                    root
                    / f"data/chunk-{data_chunk:03d}/file-{data_file_idx:03d}.parquet"
                )
                video_path = (
                    root
                    / f"videos/{video_key}/chunk-{video_chunk:03d}/file-{video_file_idx:03d}.mp4"
                )
                ep_index = int(row["episode_index"])
                mapped_data_path = episode_data_file_map.get(ep_index)
                data_file_repaired = False
                parquet_length = episode_frame_count_map.get(ep_index)
                if parquet_length is not None and parquet_length != length:
                    length_mismatches[f"meta={length},parquet={parquet_length}"] += 1
                    length = min(length, parquet_length)
                    if length < self._clip_span(min(self.frame_strides)):
                        continue
                if mapped_data_path is not None:
                    if mapped_data_path != data_path:
                        repaired_data_files[str(mapped_data_path)] += 1
                    data_path = mapped_data_path
                    data_start = 0
                    data_file_repaired = True
                elif not data_path.exists():
                    missing_data_files[str(data_path)] += 1
                    continue
                if not video_path.exists():
                    missing_video_files[str(video_path)] += 1
                    continue
                num_starts_by_stride: Dict[int, int] = {}
                for s in self.frame_strides:
                    max_start_s = length - self._clip_span(s)
                    if max_start_s < 0:
                        num_starts_by_stride[s] = 0
                    else:
                        num_starts_by_stride[s] = max(1, max_start_s // max(1, stride) + 1)
                num_starts = max(
                    1,
                    int(
                        round(
                            sum(
                                num_starts_by_stride[s] * w
                                for s, w in zip(self.frame_strides, self.frame_stride_weights)
                            )
                        )
                    ),
                )
                ep = {
                    "dataset": name,
                    "data_type": data_type,
                    "length": length,
                    "stride": stride,
                    "num_starts": num_starts,
                    "num_starts_by_stride": num_starts_by_stride,
                    "num_align_starts": 0,
                    "data_start": data_start,
                    "video_start": int(round(ts0 * fps)),
                    "episode_index": ep_index,
                    "data_file": str(data_path),
                    "data_file_repaired": data_file_repaired,
                    "video_file": str(video_path),
                    "action_key": action_key,
                    "use_align": False,
                    "valid_starts_by_stride": None,
                }
                if use_align and not (
                    defer_valid_starts and not (data_type == "robot" and not human_all_dims)
                ):
                    if data_type == "robot" and not human_all_dims:
                        ep["use_align"] = True
                        ep["num_align_starts"] = num_starts
                    else:
                        valid_by_stride: Dict[int, np.ndarray] = {}
                        for s in self.frame_strides:
                            cache_s = valid_starts_cache.get(s) if valid_starts_cache is not None else None
                            if cache_s is not None and ep_index in cache_s:
                                starts = np.asarray(cache_s[ep_index], dtype=np.int32)
                            else:
                                starts = self._valid_action_starts(
                                    ep, action_key, human_all_dims, s
                                )
                                if valid_starts_cache is not None:
                                    valid_starts_cache.setdefault(s, {})[ep_index] = (
                                        starts.astype(np.int32).tolist()
                                    )
                                    valid_starts_cache_dirty = True
                            if len(starts) > 0:
                                valid_by_stride[s] = starts
                        if valid_by_stride:
                            ep["use_align"] = True
                            ep["valid_starts_by_stride"] = valid_by_stride
                            ep["num_align_starts"] = max(
                                1,
                                int(
                                    round(
                                        sum(
                                            len(valid_by_stride.get(s, ()))
                                            * w
                                            for s, w in zip(
                                                self.frame_strides, self.frame_stride_weights
                                            )
                                        )
                                    )
                                ),
                            )
                episodes.append(ep)
        if (
            valid_starts_cache
            and valid_starts_cache_dirty
            and self._dist_rank == 0
        ):
            self._save_valid_starts_cache(
                root, action_key, human_all_dims, valid_starts_cache
            )
        self._log_missing_file_summary(name, "data", missing_data_files)
        self._log_missing_file_summary(name, "video", missing_video_files)
        if repaired_data_files:
            total_repaired = sum(repaired_data_files.values())
            preview = "; ".join(
                [
                    f"{path} ({count} eps)"
                    for path, count in repaired_data_files.most_common(5)
                ]
            )
            logger.warning(
                "LeRobot %s remapped %d episode data-file references "
                "using parquet episode_index scan. Top mapped targets: %s",
                name,
                total_repaired,
                preview,
            )
        if length_mismatches:
            preview = "; ".join(
                [
                    f"{kind} ({count} eps)"
                    for kind, count in length_mismatches.most_common(5)
                ]
            )
            logger.warning(
                "LeRobot %s found %d episodes with meta/parquet length mismatch. Top mismatch types: %s",
                name,
                sum(length_mismatches.values()),
                preview,
            )
        return episodes

    def _valid_starts_cache_path(self, root: Path, action_key: Optional[str]) -> Path:
        safe_action = (action_key or "none").replace("/", "_").replace(".", "_")
        eps = f"{self.action_zero_eps:g}".replace(".", "p")
        strides_tag = "s" + "_".join(str(s) for s in self.frame_strides)
        return (
            root
            / "meta"
            / f"ldm_valid_starts_{self.max_frames}f_action{self.action_dim}_{safe_action}_eps{eps}_{strides_tag}.json"
        )

    def _load_valid_starts_cache(
        self, root: Path, action_key: Optional[str], human_all_dims: bool = False
    ) -> Dict[int, Dict[int, List[int]]]:
        
        path = self._valid_starts_cache_path(root, action_key)
        if not path.exists():
            return {}
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            meta = payload.get("meta", {})
            if (
                meta.get("max_frames") != self.max_frames
                or meta.get("action_dim") != self.action_dim
            ):
                return {}
            if (
                meta.get("action_zero_eps") != self.action_zero_eps
                or meta.get("action_key") != action_key
            ):
                return {}
            if meta.get("frame_strides") != list(self.frame_strides):
                return {}
            if bool(meta.get("human_require_all_action_dims_nonzero", False)) != bool(
                human_all_dims
            ):
                return {}
            logger.info("Loaded LeRobot valid starts cache: %s", path)
            return {
                int(s): {int(k): v for k, v in per_ep.items()}
                for s, per_ep in payload.get("valid_starts_by_stride", {}).items()
            }
        except Exception as exc:
            logger.warning(
                "Ignoring invalid LeRobot valid starts cache %s: %s", path, exc
            )
            return {}

    def _save_valid_starts_cache(
        self,
        root: Path,
        action_key: Optional[str],
        human_all_dims: bool,
        cache: Dict[int, Dict[int, List[int]]],
    ):
        path = self._valid_starts_cache_path(root, action_key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(f".tmp.{random.randint(0, 1_000_000_000)}")
            payload = {
                "meta": {
                    "max_frames": self.max_frames,
                    "action_dim": self.action_dim,
                    "action_zero_eps": self.action_zero_eps,
                    "action_key": action_key,
                    "frame_strides": list(self.frame_strides),
                    "human_require_all_action_dims_nonzero": bool(human_all_dims),
                },
                "valid_starts_by_stride": {
                    str(s): {str(k): v for k, v in per_ep.items()}
                    for s, per_ep in cache.items()
                },
            }
            with open(tmp_path, "w") as f:
                json.dump(payload, f)
            tmp_path.replace(path)
            logger.info("Saved LeRobot valid starts cache: %s", path)
        except Exception as exc:
            logger.warning(
                "Failed to save LeRobot valid starts cache %s: %s", path, exc
            )

    @staticmethod
    def _build_episode_data_file_map(
        root: Path, cache_path: Optional[str] = None
    ) -> Tuple[Dict[int, Path], Dict[int, int]]:
        import pandas as pd

        data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
        cache = (
            Path(cache_path)
            if cache_path
            else root / "meta" / "ldm_episode_data_file_map.json"
        )
        signature = [
            [
                str(path.relative_to(root)),
                path.stat().st_size,
                int(path.stat().st_mtime_ns),
            ]
            for path in data_files
        ]
        if cache.exists():
            try:
                with open(cache, "r") as f:
                    payload = json.load(f)
                if payload.get("signature") == signature:
                    mapping = {
                        int(k): root / v for k, v in payload["episode_to_file"].items()
                    }
                    frame_counts = {
                        int(k): int(v)
                        for k, v in payload["episode_frame_counts"].items()
                    }
                    logger.info("Loaded LeRobot episode data-file map cache: %s", cache)
                    return mapping, frame_counts
            except Exception as exc:
                logger.warning(
                    "Ignoring invalid LeRobot episode data-file map cache %s: %s",
                    cache,
                    exc,
                )

        logger.info(
            "Building LeRobot episode data-file map from %d parquet files under %s",
            len(data_files),
            root,
        )
        mapping: Dict[int, Path] = {}
        frame_counts: Dict[int, int] = {}
        for data_file in data_files:
            try:
                df = pd.read_parquet(data_file, columns=["episode_index"])
            except Exception:
                continue
            if "episode_index" not in df.columns:
                continue
            counts = df["episode_index"].value_counts()
            for ep_idx, count in counts.items():
                ep_idx = int(ep_idx)
                mapping[ep_idx] = data_file
                frame_counts[ep_idx] = int(count)

        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp_cache = cache.with_suffix(f".tmp.{random.randint(0, 1_000_000_000)}")
            payload = {
                "signature": signature,
                "episode_to_file": {
                    str(k): str(v.relative_to(root)) for k, v in mapping.items()
                },
                "episode_frame_counts": {
                    str(k): int(v) for k, v in frame_counts.items()
                },
            }
            with open(tmp_cache, "w") as f:
                json.dump(payload, f)
            tmp_cache.replace(cache)
            logger.info("Saved LeRobot episode data-file map cache: %s", cache)
        except Exception as exc:
            logger.warning(
                "Failed to save LeRobot episode data-file map cache %s: %s", cache, exc
            )
        return mapping, frame_counts

    @staticmethod
    def _log_missing_file_summary(name: str, file_type: str, counts: Counter):
        if not counts:
            return
        total = sum(counts.values())
        preview = "; ".join(
            [f"{path} ({count} eps)" for path, count in counts.most_common(5)]
        )
        logger.warning(
            "LeRobot %s skipped %d episodes due to %d missing %s files. Top missing: %s",
            name,
            total,
            len(counts),
            file_type,
            preview,
        )

    def _get_df(self, data_file: str):
        import pandas as pd

        df = self._parquet_cache.get(data_file)
        if df is None:
            df = pd.read_parquet(data_file)
            self._parquet_cache[data_file] = df
            if len(self._parquet_cache) > 4:
                self._parquet_cache.pop(next(iter(self._parquet_cache)))
        return df

    def _read_rows(self, ep: Dict[str, Any], start: int, span: int):
        
        df = self._get_df(ep["data_file"])
        lo = ep["data_start"] + start
        hi = lo + span
        if "index" in df.columns and not ep.get("data_file_repaired", False):
            rows = df[(df["index"] >= lo) & (df["index"] < hi)]
            if len(rows) == span:
                return rows
        if {"episode_index", "frame_index"}.issubset(df.columns):
            rows = df[
                (df["episode_index"] == ep["episode_index"])
                & (df["frame_index"] >= start)
                & (df["frame_index"] < start + span)
            ]
            if len(rows) == span:
                return rows.sort_values("frame_index")
        rows = df.iloc[start : start + span]
        if len(rows) != span:
            raise ValueError(
                f"expected {span} rows, got {len(rows)} from {ep['data_file']} start={start}"
            )
        return rows

    def _aggregate_actions(self, raw_actions: np.ndarray, stride: int) -> np.ndarray:
        
        agg = np.zeros((self.max_frames, self.action_dim), dtype=np.float32)
        for t in range(1, self.max_frames):
            seg = raw_actions[(t - 1) * stride + 1 : t * stride + 1]  
            if self.action_dim >= 14:
                agg[t, self._trans_dims] = seg[:, self._trans_dims].sum(axis=0)
                agg[t, self._rot_dims] = seg[:, self._rot_dims].sum(axis=0)
                agg[t, self._grip_dims] = seg[-1, self._grip_dims]
            else:
                agg[t] = seg.sum(axis=0)
        return agg

    def _read_actions(self, ep: Dict[str, Any], start: int, stride: int) -> np.ndarray:
        span = self._clip_span(stride)
        rows = self._read_rows(ep, start, span)
        key = ep["action_key"]
        raw = np.stack(
            [np.asarray(x, dtype=np.float32).reshape(-1) for x in rows[key].tolist()], 0
        )
        if raw.shape != (span, self.action_dim):
            raise ValueError(f"bad raw action shape {raw.shape}, expected {(span, self.action_dim)}")
        arr = self._aggregate_actions(raw, stride)
        if arr.shape != (self.max_frames, self.action_dim):
            raise ValueError(f"bad aggregated action shape {arr.shape}")
        return arr

    def _episode_rows(self, ep: Dict[str, Any]):
        df = self._get_df(ep["data_file"])
        lo = ep["data_start"]
        hi = lo + ep["length"]
        if "index" in df.columns and not ep.get("data_file_repaired", False):
            rows = df[(df["index"] >= lo) & (df["index"] < hi)]
            if len(rows) == ep["length"]:
                return rows
        if {"episode_index", "frame_index"}.issubset(df.columns):
            rows = df[df["episode_index"] == ep["episode_index"]].sort_values(
                "frame_index"
            )
            if len(rows) == ep["length"]:
                return rows
        rows = df.iloc[: ep["length"]]
        if len(rows) != ep["length"]:
            raise ValueError(
                f"expected episode length {ep['length']}, got {len(rows)} from {ep['data_file']}"
            )
        return rows

    def _valid_action_starts(
        self, ep: Dict[str, Any], action_key: str, human_all_dims: bool, stride: int
    ) -> np.ndarray:
        starts = []
        try:
            rows = self._episode_rows(ep)
            action_arr = np.stack(
                [
                    np.asarray(x, dtype=np.float32).reshape(-1)
                    for x in rows[action_key].tolist()
                ],
                0,
            )
        except Exception:
            return np.asarray(starts, dtype=np.int32)
        if action_arr.shape != (ep["length"], self.action_dim):
            return np.asarray(starts, dtype=np.int32)
        span = self._clip_span(stride)
        for st in range(0, ep["length"] - span + 1, ep["stride"]):
            raw = action_arr[st : st + span]
            agg = self._aggregate_actions(raw, stride)
            trans = agg[1:]  
            if not np.isfinite(trans).all():
                continue
            if np.all(np.abs(trans) <= self.action_zero_eps):
                continue
            if human_all_dims and not np.all(
                np.any(np.abs(trans) > self.action_zero_eps, axis=0)
            ):
                continue
            starts.append(st)
        return np.asarray(starts, dtype=np.int32)

    def _append_val_records(self, name: str, eps: List[Dict[str, Any]]):
        if not eps:
            return
        n_strides = len(self.frame_strides)
        per_ep = max(1, self.val_windows_per_dataset // max(1, len(eps) * n_strides))
        added = 0
        for ep in eps:
            for stride in self.frame_strides:
                max_start = ep["length"] - self._clip_span(stride)
                if max_start < 0:
                    continue
                starts = np.linspace(
                    0, max_start, num=min(per_ep, max_start + 1), dtype=np.int32
                ).tolist()
                for st in starts:
                    self.val_records.append((name, ep, int(st), int(stride)))
                    added += 1
                    if added >= self.val_windows_per_dataset:
                        return

    def _sample_episode(self, want_action: bool) -> Tuple[Dict[str, Any], bool]:
        if want_action and self.align_names:
            ds = random.choices(self.align_names, weights=self.align_weights, k=1)[0]
            ep = random.choices(self.align_eps[ds], weights=self.align_ep_weights[ds], k=1)[0]
            return ep, True
        ds = random.choices(self.recon_names, weights=self.recon_weights, k=1)[0]
        ep = random.choices(self.recon_eps[ds], weights=self.recon_ep_weights[ds], k=1)[0]
        return ep, False

    def _available_strides(self, ep: Dict[str, Any], want_action: bool) -> List[int]:
        if want_action and ep.get("valid_starts_by_stride") is not None:
            return [int(s) for s in self.frame_strides if len(ep["valid_starts_by_stride"].get(s, ())) > 0]
        num_starts_by_stride = ep.get("num_starts_by_stride", {})
        return [int(s) for s in self.frame_strides if int(num_starts_by_stride.get(s, 0)) > 0]

    def _sample_stride_for_episode(self, ep: Dict[str, Any], want_action: bool) -> int:
        available = self._available_strides(ep, want_action)
        if not available:
            return min(self.frame_strides)
        weight_by_stride = dict(zip(self.frame_strides, self.frame_stride_weights))
        weights = [float(weight_by_stride.get(s, 0.0)) for s in available]
        if sum(weights) <= 0:
            weights = [1.0] * len(available)
        return int(random.choices(available, weights=weights, k=1)[0])

    def _sample_start(self, ep: Dict[str, Any], want_action: bool, frame_stride: int) -> int:
        if want_action and ep.get("valid_starts_by_stride") is not None:
            starts = ep["valid_starts_by_stride"].get(frame_stride, [])
            if len(starts) > 0:
                return int(random.choice(starts))
        max_start = ep["length"] - self._clip_span(frame_stride)
        if max_start < 0:
            raise ValueError(
                f"episode {ep.get('episode_index')} length={ep['length']} cannot fit "
                f"max_frames={self.max_frames} with frame_stride={frame_stride}"
            )
        n = max_start // ep["stride"]
        return int(random.randint(0, n) * ep["stride"])

    def _candidate_starts(self, ep: Dict[str, Any], want_action: bool, frame_stride: int) -> List[int]:
        if want_action and ep.get("valid_starts_by_stride") is not None:
            return [int(x) for x in ep["valid_starts_by_stride"].get(frame_stride, [])]
        max_start = ep["length"] - self._clip_span(frame_stride)
        if max_start < 0:
            return []
        return list(range(0, max_start + 1, ep["stride"]))

    def _sample_batch_starts(self, ep: Dict[str, Any], want_action: bool, frame_stride: int) -> List[int]:
        candidates = self._candidate_starts(ep, want_action, frame_stride)
        if not candidates:
            return [self._sample_start(ep, want_action, frame_stride) for _ in range(self.local_batch_size)]
        if self.batch_start_jitter <= 0:
            return random.choices(candidates, k=self.local_batch_size)

        base_start = random.choice(candidates)
        hi = base_start + self.batch_start_jitter
        local_candidates = [st for st in candidates if base_start <= st <= hi]
        if not local_candidates:
            local_candidates = [base_start]
        if len(local_candidates) >= self.local_batch_size:
            return random.sample(local_candidates, k=self.local_batch_size)
        return random.choices(local_candidates, k=self.local_batch_size)

    def _get_video_decoder(self, video_file: str):
        dec = self._video_decoder_cache.get(video_file)
        if dec is None:
            dec = VideoDecoder(
                video_file,
                dimension_order="NCHW",
                num_ffmpeg_threads=1,
                device="cpu",
                seek_mode="exact",
            )
            self._video_decoder_cache[video_file] = dec
            if len(self._video_decoder_cache) > self._video_decoder_cache_size:
                self._video_decoder_cache.pop(next(iter(self._video_decoder_cache)))
        return dec

    def _load_video(self, ep: Dict[str, Any], start: int, frame_stride: int) -> torch.Tensor:
        idxs = [
            ep["video_start"] + start + i * frame_stride
            for i in range(self.max_frames)
        ]
        dec = self._get_video_decoder(ep["video_file"])
        return self.transform(dec.get_frames_at(indices=idxs).data)

    def _load_sample(self, ep: Dict[str, Any], start: int, want_action: bool, frame_stride: int):
        video = self._load_video(ep, start, frame_stride)
        actions = torch.zeros((self.max_frames, self.action_dim), dtype=torch.float32)
        action_valid = torch.tensor(False)
        if want_action and ep.get("action_key"):
            actions = torch.from_numpy(self._read_actions(ep, start, frame_stride).astype(np.float32))
            action_valid = torch.tensor(True)
        mask = torch.ones((self.max_frames,), dtype=torch.bool)
        return video, mask, actions, action_valid

    def _sample_training_item(self):
        want_action = random.random() < self.action_sample_ratio
        ep, want_action = self._sample_episode(want_action)
        frame_stride = self._sample_stride_for_episode(ep, want_action)
        start = self._sample_start(ep, want_action, frame_stride)
        return self._load_sample(ep, start, want_action, frame_stride)

    def _sample_training_batch(self):
        want_action = random.random() < self.action_sample_ratio
        ep, want_action = self._sample_episode(want_action)
        frame_stride = self._sample_stride_for_episode(ep, want_action)
        videos, masks, actions, action_valid = [], [], [], []
        starts = self._sample_batch_starts(ep, want_action, frame_stride)
        for start in starts:
            video, mask, action, valid = self._load_sample(ep, start, want_action, frame_stride)
            videos.append(video)
            masks.append(mask)
            actions.append(action)
            action_valid.append(valid)
        return (
            torch.stack(videos, dim=0),
            torch.stack(masks, dim=0),
            torch.stack(actions, dim=0),
            torch.stack(action_valid, dim=0),
        )

    def __getitem__(self, idx: int):
        try:
            if self.deterministic_val and self.val_records:
                _, ep, start, frame_stride = self.val_records[idx % len(self.val_records)]
                return self._load_sample(ep, start, False, frame_stride)
            if self.batch_level_same_episode:
                return self._sample_training_batch()
            return self._sample_training_item()
        except Exception as e:
            logger.warning("LeRobot sample failed at idx=%s: %s", idx, e)
            if self.deterministic_val:
                raise
            return self.__getitem__((idx + 1) % max(1, len(self)))
