# WDAM — World Dynamics Action Model

Detailed guide for the `WDAM/` component of LD4WAM. The repository-level README covers
the one-command launch scripts, model releases and acknowledgements; this file
documents what is inside this directory and how to train, deploy and evaluate it.

## 1. What it is

LD4WAM is a **latent-action world-action model**: one mixture-of-transformers
forward pass over three token streams, `[video | latent action | action]`:

| Expert | Module | Role |
|---|---|---|
| Video | Wan2.2-TI2V-5B DiT (`openwam/model/video_backbone/`) | predicts future frames from the current frame + prompt |
| Latent action | `LatentActionExpert` (`openwam/model/latent_action_backbone/`) | learnable queries that regress the per-transition latent dynamics produced by the frozen LDM (`../LDM`, run online during training) |
| Action | `ActionDiT` (`openwam/model/action_backbone/`) | flow-matched robot actions; inverse-dynamics training and two-stage inference (video first, then action conditioned on the clean video + latent-action K/V) |

The architecture is registered as `la_tri_system` / variant `idm`
(`openwam/model/architectures/la_tri_system/idm.py`, driver in `mot_driver.py`).
Training loss: `lambda_video * L_video + lambda_action * L_action + lambda_la * L_la`.

## 2. Layout

```text
WDAM/
├── openwam/
│   ├── dataloader/          # RoboTwin reader (+ generic LeRobot-v3 readers used for pretraining)
│   ├── model/
│   │   ├── architectures/   # base.py (losses, prepare_inputs, ckpt I/O) + la_tri_system/
│   │   ├── action_backbone/ # ActionDiT + flow-matching scheduler
│   │   ├── latent_action_backbone/   # latent-action expert
│   │   ├── latent_action_model/      # frozen LDM adapter (online latent-dynamics targets)
│   │   └── video_backbone/  # Wan2.2 backbone + Wan VAE encoder
│   ├── train/               # OpenWAMTrainer (torchrun + DeepSpeed ZeRO), checkpointing
│   └── deploy/              # WebSocket policy server, engine, sync/async executors
├── configs/                 # Hydra: train.yaml (RoboTwin post-training), deploy.yaml, model/, dataloader/
├── scripts/                 # train.sh / train.py, deploy.sh / deploy.py, deploy_multi.sh, inference_single_test.py
└── benchmarks/robotwin/     # RoboTwin 2.0 evaluation client, parallel eval, web dashboard
```

## 3. Setup

```bash
conda create -n wdam python=3.12 && conda activate wdam
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e .            # run inside WDAM/
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir /path/to/Wan2.2-TI2V-5B
```

Every path in `configs/` is a `/path/to/...` placeholder. Either edit the yaml or
pass Hydra overrides on the command line, e.g.
`model.video_backbone.model_path=/data/Wan2.2-TI2V-5B`.

## 4. Data

The shipped config trains on **RoboTwin 2.0** (`configs/dataloader/robotwin.yaml`):
`dataset/<task>/<robot>_<variant>/` HDF5 episodes, all 50 tasks by default
(`dataloader.task_name=<task>` for one), `variant: clean_50 | randomized_500 | both`,
20-D EEF actions (`action_mode: eef`) or 14-D joints (`action_mode: joint`) with
proprio. Normalization stats are computed and cached on first use, or explicitly with

```bash
python -m openwam.dataloader.utils.stats_computation.robotwin_stats_computation --dataset_dir /path/to/RoboTwin2.0/dataset
```

The package also contains the generic LeRobot-v3 readers used for the
multi-view robot mid-training stage (`robot_multiview_action`, `mixture`);
their configs are not part of this release.

## 5. Training

Launcher: `bash scripts/train.sh key=value ...` (config: `configs/train.yaml`)
(torchrun + DeepSpeed; `NPROC_PER_NODE`, `CUDA_VISIBLE_DEVICES`,
`NNODES`/`NODE_RANK`/`MASTER_ADDR` select the topology). `training.debug=true`
runs a 20-step smoke and saves at step 10 / 20.

| Config | Model | Trains | Notes |
|---|---|---|---|
| `train.yaml` | `la_tri_system_idm` | video + latent-action + action experts | RoboTwin 2.0 post-training. The latent-dynamics targets come from the frozen LDM run online: `model.architecture.latent_action_model.repo_path` already points at `../LDM`; set `.ckpt` to the released LDM checkpoint and export `DINOV3_MODEL_PATH`. `finetune_ckpt_path` warm-starts from the released WDAM checkpoint ([Hugging Face](https://huggingface.co/Jaber628/LD4WAM_WDAM_pretrain) / [ModelScope](https://www.modelscope.cn/models/Jaber628/LD4WAM_WDAM_pretrain)); null = start from the Wan weights. |

Key knobs (all under `training.`): `lambda_video / lambda_action / lambda_la`,
`batch_size`, `use_gradient_checkpointing`, `zero_stage`, `save_steps`,
`finetune_ckpt_path` (weights only; add `finetune_skip_modules=[action_backbone]` to rebuild the action expert at a new `action_dim`) vs `resume_ckpt_path` (full state).
`allow_experimental: true` must stay on — the architecture is registered as
experimental. The model yaml's top-level `freeze:` list names the frozen
pretrained parts (text encoder, VAE, image encoder).

A run directory (`training.output_path/<timestamp>/`) is self-contained:
`config.yaml`, `tokenizer/`, `checkpoint_step_*.safetensors`, normalization
stats, plus `accel_state_step_*/` for resuming.

## 6. Deployment

```bash
bash scripts/deploy.sh /path/to/run_dir [--port 8848] [--denoise-steps 10] [--execution-mode sync|async]
python scripts/inference_single_test.py --test --server ws://127.0.0.1:8848   # random-image smoke
bash scripts/deploy_multi.sh /path/to/run_dir                                 # one server per GPU, ports 8848+i
```

`configs/deploy.yaml` holds the defaults; every `inference.*` field has a
same-name CLI flag. The server loads the latest `checkpoint_step_*.safetensors`
(`--ckpt-name` pins one), derives the window geometry from the saved training
config, and runs the two-stage IDM sampler with `denoise_steps` steps.
The WebSocket protocol (obs / action / reset / ping) is described in
[benchmarks/robotwin/README.md](benchmarks/robotwin/README.md#client-protocol-for-other-benchmarks--robots).

## 7. Evaluation

RoboTwin 2.0 single-task, multi-task and parallel evaluation, plus a web
dashboard and CSV export: see [benchmarks/robotwin/README.md](benchmarks/robotwin/README.md).

The LD4WAM RoboTwin 2.0 weights are released as
[`Jaber628/LD4WAM_robotwin`](https://huggingface.co/Jaber628/LD4WAM_robotwin)
(also on [ModelScope](https://www.modelscope.cn/models/Jaber628/LD4WAM_robotwin));
serve it with `scripts/deploy.sh` and run the evaluation scripts against it.

## 8. Checks

```bash
make lint      # ruff
make check     # compileall + ruff
```

## 9. Codebase

This codebase is built on [OpenWAM](https://github.com/OpenWAM-Official/OpenWAM);
the `openwam/` package, the Hydra training / deployment entrypoints and the
RoboTwin evaluation client come from it, with the LD4WAM architecture added as
the `la_tri_system` family.
OpenWAM is licensed under Apache-2.0 (`LICENSE`, `NOTICE`); other upstream components are listed in
`THIRD_PARTY_NOTICES.md`.
