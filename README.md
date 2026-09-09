# LD4WAM: Learning Latent Dynamics from Human Videos for World Action Models

<p align="center">
  <a href="https://arxiv.org/abs/2608.22403"><img src="https://img.shields.io/badge/arXiv-placeholder-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://stubborn111.github.io/LD4WAM/"><img src="https://img.shields.io/badge/Project%20Page-online-2563eb?logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="Assets/LD4WAM.pdf"><img src="https://img.shields.io/badge/Paper-PDF-dc2626?logo=adobeacrobatreader&logoColor=white" alt="Paper PDF"></a>
</p>

LD4WAM is a framework for learning robot manipulation from large-scale human and robot videos. It introduces motion-aligned latent dynamics: an embodiment-agnostic representation that connects visual dynamics learned from video with executable robot actions.

## Overview

<p align="center">
  <img src="Assets/ld4wam-overview.png" alt="Overview of LD4WAM" width="100%">
</p>

LD4WAM learns motion-aligned latent dynamics from unified human and robot data. The resulting representation serves as a bridge between the video expert and the action expert in a World Dynamics Action Model.

## Latent Dynamics Model (LDM)

LDM encodes video clips with a frozen DINOv3 and models temporal
changes with a spatio-temporal transformer. Each transition is represented by
16 soft-quantized tokens of 32 dimensions, forming a 512-dimensional latent
dynamics representation. The training objective combines semantic feature
reconstruction with motion alignment.

### Environment

The release uses the conda environment `ldm`. Install dependencies with:

```bash
cd LDM
conda env create -f environment.yml
conda activate ldm
```

The tested environment uses CUDA 12.8 PyTorch wheels. For another CUDA
runtime, install the matching PyTorch and torchvision builds, then install the
remaining packages from `requirements.txt`.

Place the local DINOv3 checkpoint at
`LDM/pretrained/dinov3-vitl16-local`, or set `DINOV3_MODEL_PATH` to another
path.

### Data

The loader reads LeRobot v3 datasets. By default, datasets are expected under
the top-level `data/` directory; set `LDM_DATA_ROOT` to use
another root. Dataset names, roots, camera keys, and action keys are defined in
`LDM/configs/config.py`.

The expected layout is:

```text
data/
├── meta/info.json
├── meta/episodes/**/*.parquet
├── data/**/*.parquet
└── videos/<video-key>/**/*.mp4
```

### Training

Run training from `LDM/`. Use `bash scripts/train.sh 1` for a single GPU, or
pass the number of processes and a rendezvous port for multi-GPU training:

```bash
cd LDM
bash scripts/train.sh 1
bash scripts/train.sh 8 27563
```

Edit `LDM/configs/config.py` to change training hyperparameters, dataset roots,
camera keys, action keys, and sampling settings.

Data, DINOv3, and output paths can be overridden with environment variables:

```bash
LDM_DATA_ROOT=/path/to/data \
DINOV3_MODEL_PATH=/path/to/dinov3-vitl16-local \
OUTPUT_ROOT=./outputs \
bash scripts/train.sh 8
```

### Inference

Export frame-aligned latent dynamics with:

```bash
cd LDM
CHECKPOINT=./outputs/ldm/checkpoints/ldm_model_final.pt \
DATASETS="agiworld egodex" \
bash scripts/infer.sh 1
```

Edit `LDM/configs/config.py` to change the registered datasets and their camera
or action settings. Set `CHECKPOINT`, `DATASETS`, `LIMIT_EPISODES`,
`VIDEO_KEYS`, and `OVERWRITE` when calling `LDM/scripts/infer.sh` to control
the checkpoint and export behavior.

### 🚀 Model Release

Our trained LDM model release is available from [ModelScope](https://www.modelscope.cn/models/Jaber628/LD4WAM_LDM) and [Hugging Face](https://huggingface.co/Jaber628/LD4WAM_LDM).

## World Dynamics Action Model (WDAM)

WDAM couples the Wan2.2-TI2V-5B video expert, a latent-dynamics expert
supervised by the frozen LDM, and an action expert in one mixture-of-transformers
forward pass. The code lives in [`WDAM/`](WDAM/) and is built on
[OpenWAM](https://github.com/OpenWAM-Official/OpenWAM); the detailed component
guide is [`WDAM/README.md`](WDAM/README.md). All commands below run inside `WDAM/`.

### Environment

```bash
cd WDAM
conda create -n wdam python=3.12 && conda activate wdam
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install -r ../LDM/requirements.txt          # the LDM is imported at training time
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir /path/to/Wan2.2-TI2V-5B
export DINOV3_MODEL_PATH=/path/to/dinov3-vitl16-local   # DINOv3 used by the LDM (see the LDM section)
```

Download the released LDM checkpoint (ModelScope / Hugging Face links above)
and the RoboTwin 2.0 dataset:

```bash
huggingface-cli download TianxingChen/RoboTwin2.0 --repo-type dataset --local-dir /path/to/RoboTwin2.0
cd /path/to/RoboTwin2.0/dataset/<task> && unzip aloha-agilex_clean_50.zip      # per task / variant
```

### Training (RoboTwin 2.0 post-training)

`configs/train.yaml` composes `model=la_tri_system_idm` + `dataloader=robotwin`.
Fill in the `/path/to/...` placeholders in the yamls or pass them as Hydra
overrides; `latent_action_model.repo_path` already points at `../LDM`.

```bash
# 20-step debug run, one task, 1 GPU
NPROC_PER_NODE=1 bash scripts/train.sh \
    model.video_backbone.model_path=/path/to/Wan2.2-TI2V-5B \
    model.architecture.latent_action_model.ckpt=/path/to/ldm_model_final.pt \
    dataloader.dataset_dir=/path/to/RoboTwin2.0/dataset \
    dataloader.task_name=adjust_bottle dataloader.variant=clean_50 \
    training.debug=true training.batch_size=1 \
    training.output_path=/path/to/ckpts

# full run, all 50 tasks, 8 GPUs (torchrun + DeepSpeed ZeRO-2)
NPROC_PER_NODE=8 bash scripts/train.sh \
    model.video_backbone.model_path=/path/to/Wan2.2-TI2V-5B \
    model.architecture.latent_action_model.ckpt=/path/to/ldm_model_final.pt \
    dataloader.dataset_dir=/path/to/RoboTwin2.0/dataset \
    training.batch_size=24 training.num_epochs=5 training.save_steps=2000 \
    training.output_path=/path/to/ckpts

# warm-start from the released WDAM pretrained checkpoint (see Model Release below)
NPROC_PER_NODE=8 bash scripts/train.sh ... training.finetune_ckpt_path=/path/to/LD4WAM_WDAM_pretrain
```

Multi-node: run the same command on every node with
`NNODES=<n> NODE_RANK=<r> MASTER_ADDR=<ip>`. Each run writes a self-contained
directory `training.output_path/<timestamp>/` (`config.yaml`, `tokenizer/`,
`checkpoint_step_*.safetensors`, normalization stats) that is used for deployment.

### Deployment

```bash
bash scripts/deploy.sh /path/to/ckpts/<run> --port 8848 --denoise-steps 10   # WebSocket policy server
python scripts/inference_single_test.py --test --server ws://127.0.0.1:8848   # smoke test with random images
bash scripts/deploy_multi.sh /path/to/ckpts/<run>                              # one server per GPU, ports 8848+i
```

### Evaluation on RoboTwin 2.0

Requires a RoboTwin checkout and conda env (`ROBOTWIN_PATH`, `ROBOTWIN_ENV`)
and a running policy server; see [`WDAM/benchmarks/robotwin/README.md`](WDAM/benchmarks/robotwin/README.md).

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin ROBOTWIN_ENV=robotwin
cd benchmarks/robotwin
bash single_eval.sh adjust_bottle demo_clean wdam 0 8848 127.0.0.1            # one task
bash multi_eval.sh -m demo_clean -n run1 -d /path/to/ckpts/<run> all          # all 50 tasks, sequential
bash parallel_eval.sh -m all -n run1 -w 8 --port 8848 all                     # N workers against deploy_multi servers
python export_results_csv.py /path/to/log_dir -o results.csv                  # collect success rates
```

`python benchmarks/web_control.py <log_dir> --benchmark robotwin --host 0.0.0.0 --port 8765` opens a live dashboard.

### 🚀 Model Release

| Checkpoint | Description | Hugging Face | ModelScope |
|---|---|---|---|
| `LD4WAM_WDAM_pretrain` | pretrained WDAM (video + latent-dynamics + action experts); the starting point for post-training | [link](https://huggingface.co/Jaber628/LD4WAM_WDAM_pretrain) | [link](https://www.modelscope.cn/models/Jaber628/LD4WAM_WDAM_pretrain) |
| `LD4WAM_robotwin` | LD4WAM (WDAM) weights for RoboTwin 2.0 (all 50 tasks), ready to serve and evaluate | [link](https://huggingface.co/Jaber628/LD4WAM_robotwin) | [link](https://www.modelscope.cn/models/Jaber628/LD4WAM_robotwin) |

Download a checkpoint directory and pass it as `training.finetune_ckpt_path`
(post-training) or to `scripts/deploy.sh` (serving); each directory ships its
`config.yaml`, tokenizer and (for the RoboTwin model) normalization statistics.

## Acknowledgements

The LDM is built upon [ViPRA](https://github.com/sroutray/vipra). The WDAM code is derived from [OpenWAM](https://github.com/OpenWAM-Official/OpenWAM) (MIT, see [`WDAM/LICENSE`](WDAM/LICENSE)); its Wan video backbone derives from [Wan2.2](https://github.com/Wan-Video/Wan2.2) and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (Apache-2.0), and evaluation uses [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin); see [`WDAM/THIRD_PARTY_NOTICES.md`](WDAM/THIRD_PARTY_NOTICES.md). We thank the authors of these projects for making their work available to the community.

## Citation

If you find LD4WAM useful, please consider citing:

```bibtex
@misc{shen2026ld4wamlearninglatentdynamics,
      title={LD4WAM: Learning Latent Dynamics from Human Videos for World Action Models}, 
      author={Zhenhao Shen and Jiaqi Liang and Jasper Lu and Feng Jiang and Yuran Wang and Chuanbo Wei and Jiayi Liu and Jianchun Yang and Qize Yu and Jiadi You and Ce Hao and Guanqi He and Chen Xie and Ruihai Wu},
      year={2026},
      eprint={2608.22403},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2608.22403}, 
}
```
