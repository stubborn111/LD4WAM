# Third-party notices

This directory contains code derived from the following projects. Their
license terms apply to the corresponding parts.

| Component | Where | Upstream | License |
|---|---|---|---|
| OpenWAM framework (`openwam/` package, training / deployment entrypoints, RoboTwin evaluation client) | `openwam/`, `scripts/`, `benchmarks/` | https://github.com/OpenWAM-Official/OpenWAM | MIT (see `LICENSE`) |
| Wan2.2 video DiT / VAE implementation | `openwam/model/video_backbone/wan/` | https://github.com/Wan-Video/Wan2.2 | Apache-2.0 |
| DiffSynth-Studio (Wan pipeline and model code the backbone is adapted from) | `openwam/model/video_backbone/wan/` | https://github.com/modelscope/DiffSynth-Studio | Apache-2.0 |
| RoboTwin 2.0 benchmark (evaluation interface targets its `eval_policy.py`) | `benchmarks/robotwin/` | https://github.com/RoboTwin-Platform/RoboTwin | MIT |

Model weights (Wan2.2-TI2V-5B, LAQ checkpoints) are not distributed with this
code and are subject to their own licenses.
