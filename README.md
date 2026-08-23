# LD4WAM: Learning Latent Dynamics from Human Videos for World Action Models

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-placeholder-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
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
the top-level `data/` directory, next to `LDM/`; set `LDM_DATA_ROOT` to use
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

WDAM is built on top of an upstream project. To align with that project's release schedule, the WDAM implementation will be released together with the corresponding upstream release. Stay tuned.

## Acknowledgements

This project is built upon [ViPRA](https://github.com/sroutray/vipra). We thank the authors for their work and for making the project available to the community.

## Citation

If you find LD4WAM useful, please consider citing:

```bibtex

```
