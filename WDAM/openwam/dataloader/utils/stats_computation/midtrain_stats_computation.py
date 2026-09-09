#!/usr/bin/env python3
"""Compute action/proprio normalization stats for the midtrain robot datasets.

CONFIG-DRIVEN: reads the SAME dataloader yaml the reader uses, so the stats are
computed over byte-identical raw vectors (no layout drift). For each
``robot_multiview_action`` source it writes a flat ``meta/norm_stats.json`` at the
native raw width (the concatenation defined by that source's ``action_cols`` /
``state_cols``). The reader loads it and normalizes the native raw
(``normalize_mode=quantile``) BEFORE the unify scatter.

Design (mirrors behavior/robocoin stats, NOT robotwin):
  * Reuse RoboCOIN's streaming ``Accumulator`` (exact mean/std/min/max + reservoir
    q01/q99) and the shared ``pin_rot6d_identity``.
  * Build the raw vectors with the reader's own ``assemble_from_spec`` (parsed from
    the yaml ``action_cols`` / ``state_cols``) — zero layout drift.
  * Pool the ACTION stream + the PROPRIO stream into ONE accumulator (grip/joint
    marginals differ between command and measurement; the union serves both).
  * Pin the source's ``rot6d_dims`` (in the native raw layout) to identity so
    rotation stays a pass-through on the manifold; pos/joint/grip keep real stats.

Output schema (``meta/norm_stats.json``):
    {"mean":[..N],"std":[..N],"min":[..N],"max":[..N],"q01":[..N],"q99":[..N],
     "num_timesteps":T,"num_files":M,"action_dim":N,"rot6d_dims":[...],
     "rot6d_identity":bool,"pool":"action+proprio"}

Usage:
    python -m openwam.dataloader.utils.stats_computation.midtrain_stats_computation \
        --dataloader_config configs/dataloader/midtrain_robot.yaml
    # optionally restrict to a subset by dataset_dir substring:
    #   --only Robomind1.0_franka_merged
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from openwam.dataloader.robot_multiview_action import (
    assemble_from_spec,
    col_spec_needed_cols,
    parse_col_spec,
)
from openwam.dataloader.utils.normalization import pin_rot6d_identity

RESERVOIR_CAP = 1_000_000


class Accumulator:
    """Online mean/std/min/max accumulator + reservoir for q01/q99."""

    def __init__(self, dim: int = 20, reservoir_cap: int = RESERVOIR_CAP, seed: int = 0):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.min_val = np.full(dim, np.inf, dtype=np.float64)
        self.max_val = np.full(dim, -np.inf, dtype=np.float64)
        # Reservoir sample (Algorithm R) for quantile estimation.
        self.cap = int(reservoir_cap)
        self.rng = np.random.RandomState(seed)
        self._res = np.empty((self.cap, dim), dtype=np.float32)
        self._res_n = 0  # rows currently in the reservoir
        self._res_seen = 0  # rows offered to the reservoir so far

    def update(self, batch: np.ndarray):
        """Update with (N, dim) array using Welford's online algorithm."""
        for i in range(len(batch)):
            x = batch[i].astype(np.float64)
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.m2 += delta * delta2
            self.min_val = np.minimum(self.min_val, x)
            self.max_val = np.maximum(self.max_val, x)
        self._reservoir_add(np.asarray(batch, dtype=np.float32))

    def update_batch(self, batch: np.ndarray):
        """Batch update (more efficient for large arrays)."""
        n = len(batch)
        if n == 0:
            return
        self._reservoir_add(np.asarray(batch, dtype=np.float32))
        batch = batch.astype(np.float64)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_min = batch.min(axis=0)
        batch_max = batch.max(axis=0)

        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_var * n
            self.min_val = batch_min
            self.max_val = batch_max
            self.count = n
        else:
            total = self.count + n
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * n / total
            self.m2 = self.m2 + batch_var * n + delta**2 * self.count * n / total
            self.mean = new_mean
            self.count = total
            self.min_val = np.minimum(self.min_val, batch_min)
            self.max_val = np.maximum(self.max_val, batch_max)

    def _reservoir_add(self, batch: np.ndarray):
        """Feed (N, dim) rows into the reservoir (vectorized Algorithm R)."""
        n = len(batch)
        if n == 0:
            return
        # Phase 1: fill until the reservoir is at capacity.
        if self._res_n < self.cap:
            take = min(self.cap - self._res_n, n)
            self._res[self._res_n : self._res_n + take] = batch[:take]
            self._res_n += take
            self._res_seen += take
            batch = batch[take:]
            if len(batch) == 0:
                return
        # Phase 2: each further row replaces a random slot with prob cap/(seen+1).
        m = len(batch)
        t = self._res_seen + np.arange(m)  # 0-indexed global position (>= cap)
        p = self.rng.randint(0, t + 1)  # random int in [0, t] per row
        keep = p < self.cap
        self._res[p[keep]] = batch[keep]
        self._res_seen += m

    def finalize(self):
        std = np.sqrt(self.m2 / max(self.count, 1))
        std = np.where(std < 1e-8, 1.0, std)
        if self._res_n > 0:
            res = self._res[: self._res_n]
            q01 = np.quantile(res, 0.01, axis=0)
            q99 = np.quantile(res, 0.99, axis=0)
        else:  # no data — degenerate fallback (keeps the schema complete)
            q01 = self.min_val
            q99 = self.max_val
        return {
            "mean": self.mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "min": self.min_val.astype(np.float32).tolist(),
            "max": self.max_val.astype(np.float32).tolist(),
            "q01": q01.astype(np.float32).tolist(),
            "q99": q99.astype(np.float32).tolist(),
        }

_ROBOT_TYPE = "robot_multiview_action"


def _iter_data_parquets(dataset_dir: Path):
    """Yield every ``data/**/*.parquet`` on disk (sorted; partial-download safe)."""
    data_dir = dataset_dir / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{data_dir} does not exist.")
    yield from sorted(data_dir.rglob("*.parquet"))


def compute_stats(dataset_dir: Path, action_spec, state_spec, rot6d_dims) -> dict:
    a_parsed = parse_col_spec(action_spec)
    s_parsed = parse_col_spec(state_spec)
    needed = tuple(dict.fromkeys(col_spec_needed_cols(a_parsed) + col_spec_needed_cols(s_parsed)))
    files = list(_iter_data_parquets(dataset_dir))  # materialize for a total (progress bar)
    acc = None
    n_files = 0
    try:
        from tqdm import tqdm

        pbar = tqdm(files, desc=f"  {dataset_dir.name}", unit="file")
    except Exception:  # tqdm absent → plain iterator (periodic prints below)
        pbar = files
    for i, fpath in enumerate(pbar):
        try:
            df = pq.read_table(fpath, columns=list(needed)).to_pandas()
            if len(df) == 0:
                continue
            a = assemble_from_spec(df, a_parsed)  # (T, N)
            p = assemble_from_spec(df, s_parsed)  # (T, N)
            if acc is None:
                acc = Accumulator(dim=a.shape[1])
            acc.update_batch(np.concatenate([a, p], axis=0).astype(np.float32))
            n_files += 1
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(timesteps=f"{acc.count:,}")
            elif (i + 1) % 20 == 0:  # no tqdm → periodic print
                print(f"    {dataset_dir.name}: {i + 1}/{len(files)} files, {acc.count:,} timesteps", flush=True)
        except Exception as e:  # noqa: BLE001 — skip a corrupt/partial shard, keep going
            print(f"  Warning: skipping {fpath}: {e}")

    if acc is None or acc.count == 0:
        raise RuntimeError(f"no usable parquet under {dataset_dir}/data — nothing to compute stats from.")

    out = acc.finalize()
    dims = [int(d) for d in (rot6d_dims or [])]
    if dims:
        pin_rot6d_identity(out, dims)  # finalize() returns lists; pins stats[key][i]
    out.update(
        {
            "num_timesteps": int(acc.count),
            "num_files": n_files,
            "action_dim": int(acc.dim),
            "rot6d_dims": dims,
            "rot6d_identity": bool(dims),
            "pool": "action+proprio",
        }
    )
    return out


def _process_source(src) -> None:
    dataset_dir = Path(str(src["dataset_dir"]))
    result = compute_stats(dataset_dir, src["action_cols"], src["state_cols"], src.get("rot6d_dims"))
    out_dir = dataset_dir / "meta"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "norm_stats.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"== {dataset_dir.name} ==\n"
        f"  timesteps={result['num_timesteps']:,} files={result['num_files']} dim={result['action_dim']}\n"
        f"  q01[:6]={[round(x, 3) for x in result['q01'][:6]]}\n"
        f"  q99[:6]={[round(x, 3) for x in result['q99'][:6]]}\n"
        f"  rot6d pinned identity: {result['rot6d_identity']} at {result['rot6d_dims']}\n"
        f"  Saved: {out_path}"
    )


def _finalize_out(acc, n_files: int, rot6d_dims) -> dict:
    """Finalize an Accumulator into the norm_stats schema (shared by per-source and pooled)."""
    out = acc.finalize()
    dims = [int(d) for d in (rot6d_dims or [])]
    if dims:
        pin_rot6d_identity(out, dims)  # finalize() returns lists; pins stats[key][i]
    out.update(
        {
            "num_timesteps": int(acc.count),
            "num_files": n_files,
            "action_dim": int(acc.dim),
            "rot6d_dims": dims,
            "rot6d_identity": bool(dims),
            "pool": "action+proprio",
        }
    )
    return out


def _process_pooled(sources) -> None:
    """Pool action+proprio across ALL sources into ONE Accumulator, then write the
    SAME norm_stats.json into every source's meta/ (one shared stat for one robot).

    Every source must declare the SAME native raw layout (action_cols / state_cols /
    rot6d_dims) — pooling only makes sense when the concatenated vectors are aligned.
    """
    first = sources[0]
    a_parsed = parse_col_spec(first["action_cols"])
    s_parsed = parse_col_spec(first["state_cols"])
    rot6d_dims = first.get("rot6d_dims")
    needed = tuple(dict.fromkeys(col_spec_needed_cols(a_parsed) + col_spec_needed_cols(s_parsed)))
    for src in sources[1:]:  # guard against pooling misaligned layouts
        if (
            parse_col_spec(src["action_cols"]) != a_parsed
            or parse_col_spec(src["state_cols"]) != s_parsed
            or [int(d) for d in (src.get("rot6d_dims") or [])] != [int(d) for d in (rot6d_dims or [])]
        ):
            raise SystemExit(
                f"--pool_all requires identical action_cols/state_cols/rot6d_dims across sources; "
                f"{src.get('dataset_dir')} differs from {first.get('dataset_dir')}."
            )

    acc = None
    n_files = 0
    for src in sources:
        dataset_dir = Path(str(src["dataset_dir"]))
        files = list(_iter_data_parquets(dataset_dir))
        try:
            from tqdm import tqdm

            pbar = tqdm(files, desc=f"  {dataset_dir.name}", unit="file")
        except Exception:
            pbar = files
        for i, fpath in enumerate(pbar):
            try:
                df = pq.read_table(fpath, columns=list(needed)).to_pandas()
                if len(df) == 0:
                    continue
                a = assemble_from_spec(df, a_parsed)  # (T, N)
                p = assemble_from_spec(df, s_parsed)  # (T, N)
                if acc is None:
                    acc = Accumulator(dim=a.shape[1])
                acc.update_batch(np.concatenate([a, p], axis=0).astype(np.float32))
                n_files += 1
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(timesteps=f"{acc.count:,}")
                elif (i + 1) % 20 == 0:
                    print(f"    {dataset_dir.name}: {i + 1}/{len(files)} files, {acc.count:,} timesteps", flush=True)
            except Exception as e:  # noqa: BLE001 — skip a corrupt/partial shard, keep going
                print(f"  Warning: skipping {fpath}: {e}")

    if acc is None or acc.count == 0:
        raise RuntimeError("no usable parquet across pooled sources — nothing to compute stats from.")

    result = _finalize_out(acc, n_files, rot6d_dims)
    for src in sources:  # write the SAME shared stat into every source's meta/
        out_dir = Path(str(src["dataset_dir"])) / "meta"
        os.makedirs(out_dir, exist_ok=True)
        with open(out_dir / "norm_stats.json", "w") as f:
            json.dump(result, f, indent=2)
    print(
        f"== POOLED ({len(sources)} sources) ==\n"
        f"  timesteps={result['num_timesteps']:,} files={result['num_files']} dim={result['action_dim']}\n"
        f"  q01[:14]={[round(x, 3) for x in result['q01'][:14]]}\n"
        f"  q99[:14]={[round(x, 3) for x in result['q99'][:14]]}\n"
        f"  rot6d pinned identity: {result['rot6d_identity']} at {result['rot6d_dims']}\n"
        f"  Saved shared norm_stats.json into: {', '.join(Path(str(s['dataset_dir'])).name for s in sources)}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataloader_config", required=True, help="dataloader yaml (mixture of robot_multiview_action sources)")
    ap.add_argument("--only", default=None, help="only process sources whose dataset_dir contains this substring")
    ap.add_argument(
        "--pool_all",
        action="store_true",
        help="pool action+proprio across ALL sources into ONE shared norm_stats.json written to every "
        "source's meta/ (requires identical action_cols/state_cols/rot6d_dims). Use for one-robot multi-task "
        "data like Piper.",
    )
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.dataloader_config), resolve=True)
    sources = [s for s in cfg.get("datasets", []) if s.get("type") == _ROBOT_TYPE and s.get("enabled", True)]
    if args.only:
        sources = [s for s in sources if args.only in str(s.get("dataset_dir", ""))]
    if not sources:
        raise SystemExit(f"no {_ROBOT_TYPE} sources in {args.dataloader_config} (filter --only={args.only!r}).")

    for src in sources:
        for key in ("dataset_dir", "action_cols", "state_cols"):
            if key not in src:
                raise SystemExit(f"source missing required key {key!r}: {src.get('dataset_dir')}")

    if args.pool_all:
        print(f"Computing POOLED stats over {len(sources)} source(s) from {args.dataloader_config}\n")
        _process_pooled(sources)
    else:
        print(f"Computing stats for {len(sources)} source(s) from {args.dataloader_config}\n")
        for src in sources:
            _process_source(src)


if __name__ == "__main__":
    main()
