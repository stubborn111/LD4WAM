"""MixtureDataset for multi-dataset co-training.

Combines multiple BaseDataset instances with configurable sampling
weights, enabling cross-dataset training (currently RoboCOIN + EgoDex).

Weight mechanism
----------------
Each sub-dataset is assigned a normalized weight w_i (sum = 1). The index
map is built once at __init__ by giving every sub-dataset a *virtual*
sample count:

    total_real  = sum of all sub-dataset real lengths
    virtual_n_i = round(total_real * w_norm_i)

Samples are drawn by cycling through the real dataset with ``vi % ds_len``.
``cycles_i = virtual_n_i / N_i``:
- cycles > 1 → oversampled (repeated visits)
- cycles < 1 → only a prefix of the source is visited at all per virtual
               epoch; the rest are NEVER seen unless ``set_epoch`` is
               called to re-shuffle.

Concrete numbers for the current configuration (RoboCOIN 72.8M + EgoDex
87.1M, total 159.9M virtual samples) under each strategy:

    manual w=[1.0, 1.0]   w_norm=[0.500, 0.500]  cycles=[1.10×, 0.92×]
    uniform               w_norm=[0.500, 0.500]  cycles=[1.10×, 0.92×]
    inverse_size          w_norm=[0.545, 0.455]  cycles=[1.20×, 0.84×]
    proportional          w_norm=[N_R, N_E]/sum  cycles=[1.00×, 1.00×]

Weight design strategies (see ``MixtureDataset.from_config`` for the
authoritative spec):

1. ``manual``: each sub-entry specifies its own ``weight`` field.
2. ``uniform``: every enabled source contributes equal virtual samples;
   small sources are oversampled, large ones undersampled.
3. ``inverse_size``: w_i ∝ 1 / (N_i × num_frames_i). Favors small sources.
4. ``proportional`` (default): w_i ∝ N_i. Every real sample visited
   exactly once per virtual epoch — no oversampling, no waste.

Per-epoch reshuffle
-------------------
The index_map is built ONCE at __init__. With non-proportional strategies,
some real samples may NEVER appear in any virtual epoch (those whose
source has cycles < 1, so virtual_n < N). Call ``mixture.set_epoch(epoch)``
at each epoch boundary to re-shuffle with seed=base_seed+epoch*7919, which
gives different sub-samples per epoch. With ``proportional`` this is
mostly cosmetic (full coverage every epoch regardless).

Per-source normalization
------------------------
MixtureDataset does NOT aggregate or apply any normalization. ``normalization_stats``
is hard-coded None. Each sub-source's reader handles its own normalization
in ``__getitem__`` (e.g. RoboCOINDataset applies per-bucket min-max with
its robot_type stats), so no downstream re-normalization happens.

Adjust weights at launch without editing the yaml:
    dataloader.datasets.robocoin.weight=3.0 dataloader.datasets.egodex.weight=0.3

Example config (configs/dataloader/mixture.yaml):
    defaults:
      - robocoin@datasets.robocoin
      - egodex@datasets.egodex
      - _self_

    type: mixture
    weight_strategy: proportional
    datasets:
      robocoin: {enabled: true, weight: 1.0}
      egodex:   {enabled: true, weight: 1.0}
"""

import copy
import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from openwam.dataloader.bases import BaseDataset

logger = logging.getLogger(__name__)


class MixtureDataset(BaseDataset):
    """Weighted mixture of multiple action datasets.

    Samples are drawn from sub-datasets according to normalized weights.
    Each sub-dataset contributes ``round(total_real * weight_i)`` virtual
    samples; samples cycle if the virtual count exceeds the real dataset size.
    The virtual index list is shuffled once with a fixed seed.

    Args:
        datasets:             List of BaseDataset instances.
        weights:              Sampling weight per dataset (normalized to sum=1).
                              If None, defaults to dataset sizes — equivalent to
                              ``weight_strategy=proportional``: every real sample
                              visited exactly once per virtual epoch.
        seed:                 Random seed for reproducible index shuffling.
        names:                Optional human-readable name per sub-dataset (for
                              logging and ``get_dataset`` lookup). Defaults to
                              ``source_<i>``.
        strict_action_dim:    When True, mismatched sub-dataset ``action_dim``
                              raises ValueError. When False (legacy auto-pad path,
                              opt-in via ``action_dim_override``), the max dim is
                              used and smaller actions are zero-padded with a warning.
        action_dim_override:  Only consulted when ``strict_action_dim=False``.
                              Forces the output action_dim regardless of
                              sub-source dims; smaller actions are zero-padded.
    """

    def __init__(
        self,
        datasets: Sequence[BaseDataset],
        weights: Optional[Sequence[float]] = None,
        seed: int = 42,
        action_dim_override: Optional[int] = None,
        names: Optional[Sequence[str]] = None,
        strict_action_dim: bool = False,
    ):
        if not datasets:
            raise ValueError("MixtureDataset requires at least one sub-dataset")

        self._datasets = list(datasets)
        self._base_seed = seed
        self._seed = seed
        self._strict_action_dim = strict_action_dim

        if names is None:
            self._names = [f"source_{i}" for i in range(len(self._datasets))]
        else:
            names = list(names)
            if len(names) != len(self._datasets):
                raise ValueError(f"names length ({len(names)}) != datasets length ({len(self._datasets)})")
            self._names = [str(n) for n in names]

        dims = [d.action_dim for d in self._datasets]
        if strict_action_dim:
            if len(set(dims)) != 1:
                raise ValueError(
                    "MixtureDataset: all enabled sub-datasets must share the same action_dim, "
                    f"got {dict(zip(self._names, dims))}. Align action_dim across sub-sources "
                    "before mixing (e.g. pad in the per-source reader). To restore the legacy "
                    "auto-pad behavior, re-add ``action_dim_override`` to the yaml."
                )
            self._action_dim = dims[0]
        else:
            # Legacy auto-pad branch — only reachable from configs that keep an
            # `action_dim_override: <value>` field. New configs (mixture.yaml +
            # anything composed via the modern from_config path) hit the strict
            # branch above.
            if action_dim_override is not None:
                self._action_dim = action_dim_override
            elif len(set(dims)) == 1:
                self._action_dim = dims[0]
            else:
                self._action_dim = max(dims)
                logger.warning(
                    "Sub-datasets have different action dims %s; padding to max=%d. "
                    "Set action_dim_override explicitly to suppress this warning.",
                    dims,
                    self._action_dim,
                )

        if weights is None:
            weights = [float(len(d)) for d in self._datasets]
        if len(weights) != len(self._datasets):
            raise ValueError(f"weights length ({len(weights)}) != datasets length ({len(self._datasets)})")
        if any(w < 0 for w in weights):
            raise ValueError(f"MixtureDataset: negative weights are not allowed: {weights}")
        total_w = sum(weights)
        if total_w <= 0:
            raise ValueError("MixtureDataset: all sub-sources are empty or zero-weight; nothing to sample.")
        self._weights = [w / total_w for w in weights]

        self._build_index_map()

        logger.info(
            "MixtureDataset: %d sub-datasets, total %d virtual samples, weights={%s}",
            len(self._datasets),
            len(self),
            ", ".join(f"{n}: {w:.3f}" for n, w in zip(self._names, self._weights)),
        )

    def set_epoch(self, epoch: int) -> None:
        """Re-shuffle the virtual index_map for this epoch.

        ===================================================================
        ⚠️  CALLER REQUIREMENT — READ BEFORE USE  ⚠️

        This method mutates ``self._index_map`` in place. Under
        ``torch.utils.data.DataLoader`` with ``num_workers > 0``, each
        worker holds a forked copy of the dataset object; calling
        ``set_epoch`` on the parent only updates the parent's copy. To make
        the new shuffle visible inside the workers, the caller MUST:

          1. Use ``persistent_workers=False`` (workers are re-forked every
             epoch, so they pick up the parent's updated state on each
             epoch boundary), AND
          2. Call ``set_epoch(epoch)`` BEFORE constructing / iterating the
             DataLoader for that epoch (typically at the top of the
             per-epoch loop in the trainer).

        With ``persistent_workers=True`` the workers never see the new
        index_map — set_epoch becomes a silent no-op for sampling, and
        every epoch reuses the init-time shuffle. Use a custom Sampler
        instead if persistent workers are required.
        ===================================================================

        Mirrors ``torch.utils.data.distributed.DistributedSampler.set_epoch``:
        without this call the index_map keeps its init-time shuffle for the
        entire run — fine under ``proportional`` (full coverage every epoch
        regardless) but ``inverse_size`` / ``manual`` strategies will
        systematically miss any samples whose virtual_n < N.

        Safe to skip; default behavior is identical to the previous releases.
        """
        self._seed = self._base_seed + int(epoch) * 7919
        self._build_index_map()

    def _build_index_map(self):
        """Build (dataset_idx, sample_idx) virtual-index map.

        Stored as a 2D int32 array. int32 is sufficient: a single sub-source
        with > 2.1B samples would already be unworkable for other reasons,
        and source idx maxes out at a small handful. int32 vs int64 halves
        the memory footprint of the index_map — on the production mixture
        that's ~1.3 GB instead of ~2.5 GB per DataLoader worker process.
        """
        total_real = sum(len(d) for d in self._datasets)
        parts: List[np.ndarray] = []
        for di, (ds, w) in enumerate(zip(self._datasets, self._weights)):
            n_real = len(ds)
            if n_real == 0 or w <= 0.0:
                # Skip empty / zero-weight sources. A zero-length ds would make
                # ``arange(virtual_n) % 0`` emit index 0 into an empty dataset
                # (numpy only warns), then __getitem__ raises an opaque IndexError
                # inside a worker. Surface it here with a clear log instead.
                logger.warning(
                    "MixtureDataset: skipping source '%s' (len=%d, weight=%.4f) — not sampled.",
                    self._names[di],
                    n_real,
                    w,
                )
                continue
            virtual_n = max(1, round(total_real * w))
            sample_idx = (np.arange(virtual_n, dtype=np.int64) % n_real).astype(np.int32)
            di_arr = np.full(virtual_n, di, dtype=np.int32)
            parts.append(np.stack([di_arr, sample_idx], axis=1))
        if not parts:
            raise RuntimeError("MixtureDataset: all sub-sources are empty or zero-weight; nothing to sample.")
        combined = np.concatenate(parts, axis=0)
        rng = np.random.RandomState(self._seed)
        perm = rng.permutation(len(combined))
        self._index_map = combined[perm]

    def __getitem__(self, idx: int) -> dict:
        di, si = self._index_map[idx]
        di, si = int(di), int(si)
        sample = self._datasets[di][si]

        # Legacy auto-pad path (strict_action_dim=False, where sub-source
        # action_dims may differ). Under strict mode all
        # sub-sources share action_dim by construction and this branch never
        # triggers.
        if not self._strict_action_dim:
            action_traj = sample.get("action")
            if (
                action_traj is not None
                and isinstance(action_traj, torch.Tensor)
                and action_traj.shape[-1] < self._action_dim
            ):
                pad_size = self._action_dim - action_traj.shape[-1]
                sample["action"] = torch.nn.functional.pad(action_traj, (0, pad_size))

        sample["_dataset_index"] = di
        sample["_dataset_name"] = self._names[di]
        return sample

    def __len__(self) -> int:
        return len(self._index_map)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def normalization_stats(self) -> Optional[dict]:
        # Always None: normalization is the responsibility of each sub-source's
        # reader (see e.g. RoboCOINDataset._normalize_array). MixtureDataset
        # never aggregates or re-normalizes per-source stats — doing so would
        # smear out per-source scales (e.g. RoboCOIN's per-robot_type stats)
        # into a meaningless global average.
        return None

    @property
    def datasets(self) -> List[BaseDataset]:
        return self._datasets

    @property
    def weights(self) -> List[float]:
        return self._weights

    @property
    def names(self) -> List[str]:
        return list(self._names)

    def get_dataset(self, name: str) -> BaseDataset:
        """Return the sub-dataset registered under ``name``.

        Raises ``KeyError`` with a helpful list when ``name`` is unknown.
        """
        try:
            idx = self._names.index(name)
        except ValueError:
            raise KeyError(f"MixtureDataset: no sub-dataset named '{name}'. Known: {self._names}") from None
        return self._datasets[idx]

    def dataset_sample_counts(self) -> Dict[str, int]:
        """Per-source virtual-sample counts, keyed by source name."""
        di_col = self._index_map[:, 0]
        counts = np.bincount(di_col, minlength=len(self._datasets))
        return {self._names[i]: int(c) for i, c in enumerate(counts)}

    @classmethod
    def from_config(cls, config, split: str = "train") -> "MixtureDataset":
        """Build from a Hydra/OmegaConf config.

        Sub-sources live under the ``datasets:`` key. Two layouts:

        **dict** (recommended, used by ``configs/dataloader/mixture.yaml``).
        Sub-source config is composed in via Hydra ``defaults``
        (``<name>@datasets.<name>``); the per-source block under ``datasets:``
        only carries ``enabled`` / ``weight`` overrides. The dict key is the
        source name (used in logging, ``sample['_dataset_name']``,
        ``get_dataset(name)`` lookup, CLI override path
        ``dataloader.datasets.<name>.<field>``).

        .. code-block:: yaml

            defaults:
              - robocoin@datasets.robocoin
              - egodex@datasets.egodex
              - _self_
            type: mixture
            weight_strategy: manual
            datasets:
              robocoin: {enabled: true, weight: 1.0}
              egodex:   {enabled: true, weight: 0.3}

        **list** (legacy list form).
        ``datasets:`` is a list of inline configs; the source name is derived
        from the ``type`` field (with ``source_<i>`` fallback and a numeric
        suffix to disambiguate duplicate types).

        .. code-block:: yaml

            datasets:
              - type: robotwin
                enabled: true
                weight: 1.0
                dataset_dir: /path/to/data
                ...

        Weight strategy (controlled by ``config.weight_strategy``):
          manual         — use the ``weight`` field on each sub-dataset entry.
          uniform        — ignore ``weight`` fields; each enabled source contributes
                           equal virtual samples regardless of size.
          inverse_size   — weight_i ∝ 1 / (len(ds_i) × num_frames_i): small sources
                           are oversampled, large sources undersampled. (Renamed
                           from the misleading legacy name ``token``.)
          proportional   — weight_i ∝ len(ds_i): virtual_n_i == len(ds_i) exactly,
                           every real sample is visited exactly once per virtual
                           epoch (full coverage, no waste, no duplication).
        """
        from concurrent.futures import ThreadPoolExecutor

        from openwam.dataloader.registry import build_dataset
        from openwam.dataloader.utils import get_cfg as _get

        def _copy_with_default(cfg, key, value):
            if value is None or _get(cfg, key) is not None:
                return cfg
            if hasattr(cfg, "items"):
                copied = {k: v for k, v in cfg.items()}
                copied[key] = value
                return copied
            copied = copy.copy(cfg)
            setattr(copied, key, value)
            return copied

        def _normalize_entries(datasets_cfg):
            """Return ``[(name, cfg), ...]`` preserving insertion order.

            Detects dict-style (Hydra defaults composition) vs list-style
            (legacy inline). For list-style, derives the name from the
            ``type`` field with index fallback.
            """
            if datasets_cfg is None:
                return []
            # OmegaConf DictConfig and plain dict both expose .items() AND .keys().
            # ListConfig and plain list only support iteration.
            if hasattr(datasets_cfg, "items") and hasattr(datasets_cfg, "keys") and not isinstance(datasets_cfg, list):
                return [(str(name), c) for name, c in datasets_cfg.items()]
            entries = []
            seen = {}
            for i, c in enumerate(datasets_cfg):
                base = str(_get(c, "type", f"source_{i}"))
                # Disambiguate duplicate types (e.g. two robotwin entries in one mixture)
                n = seen.get(base, 0)
                seen[base] = n + 1
                name = base if n == 0 else f"{base}_{n}"
                entries.append((name, c))
            return entries

        weight_strategy = _get(config, "weight_strategy", "proportional")
        split_manifest = _get(config, "split_manifest")
        # Authoritative run seed (project.seed, injected into cfg.dataloader.seed
        # by train.py's _inject_project_seed; else the ctor default 42). Push it
        # DOWN into each sub-source cfg so the per-reader subsample — which seeds
        # episode selection off cfg.seed — tracks project.seed too, not just the
        # mixture-level index_map shuffle. A sub-cfg that declares its own ``seed``
        # still wins (see _copy_with_default), so per-dataset pinning for ablations
        # keeps working. Default project.seed=42 leaves subsample byte-identical.
        mixture_seed = int(_get(config, "seed", 42))

        # Sub-source discovery under the ``datasets:`` key. Dict form for new
        # mixture.yaml (name → cfg), legacy list form.
        all_entries = _normalize_entries(_get(config, "datasets"))
        enabled_entries = []
        for name, c in all_entries:
            if _get(c, "enabled", True):
                c = _copy_with_default(c, "split_manifest", split_manifest)
                c = _copy_with_default(c, "seed", mixture_seed)
                enabled_entries.append((name, c))
            else:
                logger.info(
                    "MixtureDataset: skipping disabled sub-dataset '%s' (type=%s)",
                    name,
                    _get(c, "type", "?"),
                )

        enabled_names = [n for n, _ in enabled_entries]
        enabled_cfgs = [c for _, c in enabled_entries]

        # Sanity check: all enabled sub-sources must have matching window /
        # image shape; otherwise collate will fail at the first batch. Catching
        # it here gives a clear error pointing at the offending sub-cfg.
        shape_fields = ("num_frames", "video_stride", "height", "width")
        for key in shape_fields:
            present = [(n, _get(c, key)) for n, c in zip(enabled_names, enabled_cfgs) if _get(c, key) is not None]
            uniq = set(v for _, v in present)
            if len(uniq) > 1:
                raise ValueError(f"MixtureDataset: enabled sub-sources must share the same '{key}', got {present}")

        # Build sub-datasets in parallel via threads. Each sub init is mostly
        # pd.read_parquet / pyarrow / numpy — all release the GIL during the
        # heavy work. shared network filesystems like parallel IO. ProcessPool would need to pickle
        # the resulting datasets back, which is expensive (eps_df is MBs).
        if not enabled_cfgs:
            raise RuntimeError("MixtureDataset: all sub-datasets are disabled or failed to load")
        n_workers = min(len(enabled_cfgs), 16)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            sub_datasets = list(pool.map(lambda c: build_dataset(c, split=split), enabled_cfgs))

        valid_strategies = ("manual", "uniform", "inverse_size", "proportional")
        if weight_strategy not in valid_strategies:
            raise ValueError(
                f"MixtureDataset: unknown weight_strategy={weight_strategy!r}. Valid options: {valid_strategies}."
            )

        if weight_strategy == "uniform":
            # Each enabled source contributes equal virtual samples regardless of size.
            weights = [1.0] * len(sub_datasets)
            logger.info("MixtureDataset: weight_strategy=uniform, all weights set to 1.0")
        elif weight_strategy == "inverse_size":
            # w_i ∝ 1 / (N_i × num_frames_i): favors small sources, large sources get
            # undersampled. Use when you want a small dataset to be oversampled relative
            # to a large one. (num_frames is per-source; effective when sources have
            # different num_frames, otherwise reduces to w ∝ 1/N.)
            weights = []
            for ds, sub_cfg in zip(sub_datasets, enabled_cfgs):
                nf = float(_get(sub_cfg, "num_frames", 49))
                weights.append(1.0 / max(len(ds) * nf, 1.0))
            logger.info(
                "MixtureDataset: weight_strategy=inverse_size, raw weights=%s",
                [f"{w:.2e}" for w in weights],
            )
        elif weight_strategy == "proportional":
            # w_i ∝ N_i: virtual_n_i == N_i exactly, every real sample is visited
            # exactly once per virtual epoch. No oversampling, no waste.
            weights = [float(len(ds)) for ds in sub_datasets]
            logger.info("MixtureDataset: weight_strategy=proportional, raw weights=%s (= sub-source sizes)", weights)
        else:  # manual
            weights = [float(_get(sub_cfg, "weight", 1.0)) for sub_cfg in enabled_cfgs]
            logger.info("MixtureDataset: weight_strategy=manual, weights=%s", weights)

        # action_dim policy:
        # - If the cfg declares ``action_dim_override`` (even as null), it's an
        #   opt-in to the legacy "auto-pad to max with warning" path. The
        #   field value (None or int) is forwarded.
        # - If the field is absent (e.g. the new mixture.yaml), strict mode kicks
        #   in: mismatched sub-source action_dim raises in MixtureDataset.__init__.
        try:
            has_override_field = "action_dim_override" in config
        except TypeError:
            has_override_field = hasattr(config, "action_dim_override")

        return cls(
            datasets=sub_datasets,
            weights=weights,
            seed=int(_get(config, "seed", 42)),
            action_dim_override=_get(config, "action_dim_override") if has_override_field else None,
            names=enabled_names,
            strict_action_dim=not has_override_field,
        )
