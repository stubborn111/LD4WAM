
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2 as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import (
    DATA_CFG,
    DATASET_CFG,
    MODEL_CFG,
    OUTPUT_ROOT,
)  

LOG = logging.getLogger("export_latent_dynamics")
RUN_NAME = "ldm"
FORMAT_VERSION = "ldm_latent_dynamics_v1"
CAMERA_NAMES = {
    "observation.images.head": "imagehead",
    "observation.images.hand_left": "imagehandleft",
    "observation.images.hand_right": "imagehandright",
    "observation.images.ego": "imagego",
    "observation.images.stereo_left": "imagestereoleft",
    "observation.images.stereo_right": "imagestereoright",
}


def camera_name(video_key: str) -> str:
    if video_key in CAMERA_NAMES:
        return CAMERA_NAMES[video_key]
    if video_key.startswith("observation.images."):
        return "image" + "".join(
            c for c in video_key.split("observation.images.", 1)[1] if c.isalnum()
        )
    return "".join(c if c.isalnum() else "_" for c in video_key).strip("_")


def transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(
                lambda img: (
                    img.convert("RGB")
                    if isinstance(img, Image.Image) and img.mode != "RGB"
                    else img
                )
            ),
            T.ToImage(),
            T.Resize((image_size, image_size)),
            T.ToDtype(torch.float32, scale=True),
        ]
    )


def info(root: Path) -> Dict[str, Any]:
    with open(root / "meta" / "info.json", "r") as f:
        return json.load(f)


WRIST_KEY_PATTERNS = ("hand_left", "hand_right", "wrist_left", "wrist_right")


def is_wrist_key(key: str) -> bool:
    lk = key.lower()
    return any(p in lk for p in WRIST_KEY_PATTERNS)


def video_keys(
    root: Path,
    selected: Optional[List[str]],
    skip_wrist: bool = False,
) -> List[str]:
    features = info(root).get("features", {})
    keys = sorted(
        k
        for k, v in features.items()
        if isinstance(v, dict) and v.get("dtype") == "video"
    )
    if selected:
        keys = [k for k in keys if k in selected]
    if skip_wrist:
        filtered = [k for k in keys if not is_wrist_key(k)]
        removed = [k for k in keys if is_wrist_key(k)]
        if removed:
            LOG.info("skip_wrist_views: skipping keys %s", removed)
        keys = filtered
    if not keys:
        raise ValueError(f"No video features found in {root}")
    return keys


def dataset_roots(names: Optional[List[str]]) -> Dict[str, Path]:
    out = {
        n: Path(c["root_dir"])
        for n, c in DATASET_CFG.items()
        if c.get("format") == "lerobot"
    }
    if names:
        missing = sorted(set(names) - set(out))
        if missing:
            raise ValueError(f"Unknown datasets: {missing}")
        out = {n: out[n] for n in names}
    return out


def rows(root: Path, limit: Optional[int]) -> Iterable[Tuple[int, pd.Series]]:
    n = 0
    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(root / "meta" / "episodes")
    for p in files:
        df = pd.read_parquet(p)
        for _, r in df.iterrows():
            yield int(r["episode_index"]), r
            n += 1
            if limit is not None and n >= limit:
                return


def video_ref(root: Path, r: pd.Series, key: str) -> Tuple[Path, float, float]:
    ck, fk = f"videos/{key}/chunk_index", f"videos/{key}/file_index"
    ft, tt = f"videos/{key}/from_timestamp", f"videos/{key}/to_timestamp"
    chunk, file_idx = int(r[ck]), int(r[fk])
    path = root / f"videos/{key}/chunk-{chunk:03d}/file-{file_idx:03d}.mp4"
    return path, float(r[ft]), float(r[tt])


def to_pil(frame: torch.Tensor) -> Image.Image:
    arr = (
        frame.cpu().permute(1, 2, 0).numpy()
        if frame.shape[0] in (1, 3)
        else frame.cpu().numpy()
    )
    arr = np.clip(arr, 0, 255).astype(np.uint8) if arr.dtype != np.uint8 else arr
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, -1)
    return Image.fromarray(arr[..., :3], mode="RGB")


def read_clip(dec: Any, indices: List[int], tfm: T.Compose) -> torch.Tensor:
    try:
        data = dec.get_frames_at(indices=indices).data
    except Exception:
        data = torch.stack([dec.get_frames_at(indices=[i]).data[0] for i in indices])
    return torch.stack([tfm(to_pil(x)) for x in data])


def dims(model: Any) -> Tuple[int, int]:
    idx_dim = int(np.prod(getattr(model, "action_size", (4, 4))))
    qdim = int(
        getattr(getattr(model, "vq", None), "embedding_dim", MODEL_CFG["quant_dim"])
    )
    return idx_dim * qdim, idx_dim


def validate(payload: Dict[str, Any], n: int, latent_dim: int, idx_dim: int) -> None:
    la, li = payload["latent_dynamics"], payload["latent_dynamics_idxs"]
    assert tuple(la.shape) == (n, latent_dim), (la.shape, n, latent_dim)
    assert tuple(li.shape) == (n, idx_dim), (li.shape, n, idx_dim)
    assert la.dtype == torch.bfloat16, la.dtype
    assert li.dtype in (torch.int16, torch.int32, torch.int64), li.dtype
    if n:
        assert torch.all(la[0].float() == 0), "first latent row must be zero"
        assert torch.all(li[0] == 0), "first idx row must be zero"
    assert torch.isfinite(la.float()).all(), "latent has NaN/Inf"


def atomic_save(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}.{random.randint(0, 10**9)}"
    )
    torch.save(obj, tmp)
    tmp.replace(path)


def load_model(ckpt: Path, device: torch.device) -> Any:
    from ldm.latent_dynamics_model import LDM

    model = LDM(**MODEL_CFG)
    ok = model.load_weights(ckpt, map_location="cpu", strict=False, verbose=True)
    if ok is None:
        raise RuntimeError(f"failed to load {ckpt}")
    return model.eval().to(device)


@torch.no_grad()
def export_one(
    model: Any,
    root: Path,
    dname: str,
    ep: int,
    r: pd.Series,
    key: str,
    out: Path,
    device: torch.device,
    tfm: T.Compose,
    fps: float,
    mframes: int,
    batch_windows: int,
    latent_dim: int,
    idx_dim: int,
    overwrite: bool,
    decode_threads: int = 4,
) -> str:
    if out.exists() and not overwrite:
        return "skipped"
    n = int(r["length"])
    latent = torch.zeros((n, latent_dim), dtype=torch.float32)
    idxs = torch.zeros((n, idx_dim), dtype=torch.int16)
    vpath, from_ts, to_ts = video_ref(root, r, key)
    if not vpath.exists():
        raise FileNotFoundError(vpath)
    start_frame = int(round(from_ts * fps))
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(
        str(vpath),
        dimension_order="NCHW",
        num_ffmpeg_threads=decode_threads,
        device="cpu",
        seek_mode="exact",
    )
    if n and start_frame + n - 1 >= len(dec):
        raise ValueError(
            f"episode {ep} exceeds video length: start={start_frame} n={n} len={len(dec)} path={vpath}"
        )

    starts = list(range(0, max(n - 1, 0), mframes - 1))
    needed: List[int] = []
    for s in starts:
        for i in range(mframes):
            needed.append(start_frame + min(s + i, n - 1))
    unique_abs = sorted(set(needed))
    try:
        raw_batch = dec.get_frames_at(indices=unique_abs).data
        frame_map = {abs_idx: tfm(to_pil(raw_batch[k])) for k, abs_idx in enumerate(unique_abs)}
    except Exception:
        frame_map = {}
        for abs_idx in unique_abs:
            frame_map[abs_idx] = tfm(to_pil(dec.get_frames_at(indices=[abs_idx]).data[0]))

    for bs in range(0, len(starts), batch_windows):
        cur = starts[bs : bs + batch_windows]
        clips = []
        for s in cur:
            clip_indices = [start_frame + min(s + i, n - 1) for i in range(mframes)]
            clips.append(torch.stack([frame_map[idx] for idx in clip_indices]))
        batch = torch.stack(clips).to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            res = model.inference(
                batch,
                return_reconstructions=False,
                return_quantized_actions=True,
                return_quantized_actions_idxs=True,
            )
        qa = (
            res["quantized_actions"]
            .reshape(len(cur), mframes - 1, latent_dim)
            .cpu()
            .float()
        )
        qi = (
            res["quantized_actions_idxs"]
            .reshape(len(cur), mframes - 1, idx_dim)
            .cpu()
            .to(torch.int16)
        )
        for bi, s in enumerate(cur):
            for j in range(mframes - 1):
                t = s + j + 1
                if t >= n:
                    break
                latent[t], idxs[t] = qa[bi, j], qi[bi, j]
    payload = {
        "latent_dynamics": latent.to(torch.bfloat16),
        "latent_dynamics_idxs": idxs,
        "quantization": "soft",
        "episode_index": ep,
        "num_frames": n,
        "latent_dim": latent_dim,
        "idx_dim": idx_dim,
        "dataset_name": dname,
        "video_key": key,
        "camera": camera_name(key),
        "fps": fps,
        "video_path": str(vpath.relative_to(root)),
        "video_from_timestamp": from_ts,
        "video_to_timestamp": to_ts,
        "video_start_frame": start_frame,
        "window_size": mframes,
        "window_stride": mframes - 1,
        "first_frame_is_zero_padding": True,
        "dtype": "bfloat16",
        "format_version": FORMAT_VERSION,
    }
    validate(payload, n, latent_dim, idx_dim)
    atomic_save(payload, out)
    return "written"


def write_manifest(
    root: Path,
    dname: str,
    keys: List[str],
    ckpt: Path,
    latent_dim: int,
    idx_dim: int,
    fps: float,
    image_size: int,
    mframes: int,
) -> None:
    manifest = {
        "format_version": FORMAT_VERSION,
        "dataset_name": dname,
        "dataset_root": str(root),
        "checkpoint": str(ckpt),
        "video_keys": keys,
        "camera_files": {k: camera_name(k) + ".pth" for k in keys},
        "latent_dynamics": {
            "shape": "[N, 512]",
            "dtype": "torch.bfloat16",
            "quantization": "soft",
            "first_row": "zero",
        },
        "latent_dynamics_idxs": {
            "shape": "[N, 16]",
            "dtype": "torch.int16",
            "first_row": "zero",
        },
        "latent_dim": latent_dim,
        "idx_dim": idx_dim,
        "fps": fps,
        "image_size": image_size,
        "window_size": mframes,
        "window_stride": mframes - 1,
    }
    out = root / "LatentDynamics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(OUTPUT_ROOT) / RUN_NAME / "ldm_model_latest.pt",
    )
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--video-keys", nargs="*", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-windows", type=int, default=16)
    p.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of export shards. Defaults to WORLD_SIZE.",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Current shard index. Defaults to RANK.",
    )
    p.add_argument("--limit-episodes", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-wrist-views", action="store_true",
                   help="Skip camera keys containing hand_left/hand_right/wrist_* patterns.")
    p.add_argument("--decode-threads", type=int, default=4,
                   help="Number of ffmpeg threads per VideoDecoder (default 4).")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    num_shards = args.num_shards if args.num_shards is not None else world_size
    shard_index = args.shard_index if args.shard_index is not None else rank
    if not (0 <= shard_index < num_shards):
        raise ValueError(
            f"Invalid shard_index={shard_index} for num_shards={num_shards}"
        )
    if args.device == "auto":
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    LOG.info(
        "export shard %d/%d rank=%d local_rank=%d world_size=%d device=%s",
        shard_index,
        num_shards,
        rank,
        local_rank,
        world_size,
        device,
    )
    model = load_model(args.checkpoint, device)
    latent_dim, idx_dim = dims(model)
    image_size, mframes = int(DATA_CFG["image_size"]), int(DATA_CFG["max_frames"])
    tfm = transform(image_size)
    written = skipped = failed = 0
    for dname, root in dataset_roots(args.datasets).items():
        fps = float(info(root).get("fps", 30.0))
        keys = video_keys(root, args.video_keys, skip_wrist=args.skip_wrist_views)
        if shard_index == 0:
            write_manifest(
                root,
                dname,
                keys,
                args.checkpoint,
                latent_dim,
                idx_dim,
                fps,
                image_size,
                mframes,
            )
        items = list(rows(root, args.limit_episodes))
        items = [(ep, r) for ep, r in items if ep % num_shards == shard_index]
        bar = tqdm(
            items, desc=f"{dname} shard {shard_index}/{num_shards}", dynamic_ncols=True
        )
        for ep, r in bar:
            for key in keys:
                out = (
                    root
                    / "LatentDynamics"
                    / f"episode_{ep:06d}"
                    / f"{camera_name(key)}.pth"
                )
                try:
                    status = export_one(
                        model,
                        root,
                        dname,
                        ep,
                        r,
                        key,
                        out,
                        device,
                        tfm,
                        fps,
                        mframes,
                        args.batch_windows,
                        latent_dim,
                        idx_dim,
                        args.overwrite,
                        decode_threads=args.decode_threads,
                    )
                    written += status == "written"
                    skipped += status == "skipped"
                except Exception as exc:
                    failed += 1
                    LOG.exception(
                        "failed dataset=%s episode=%s key=%s: %s", dname, ep, key, exc
                    )
            bar.set_postfix(written=written, skipped=skipped, failed=failed)
    LOG.info("done written=%d skipped=%d failed=%d", written, skipped, failed)
    LOG.info("quantization=soft; latent_dynamics contains continuous probability-weighted codebook vectors")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
