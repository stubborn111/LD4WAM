import logging
import os

from accelerate import Accelerator, DistributedDataParallelKwargs
from configs.config import (
    DATA_CFG,
    DATASET_CFG,
    LEROBOT_DATALOADER_CFG,
    MODEL_CFG,
    TRAIN_CFG,
    VAL_DATASET_CFG,
    OUTPUT_ROOT,
)
from ldm.lerobot_dataset import LeRobotMixedWindowDataset
from ldm.latent_dynamics_model import LDM
from ldm.trainer import LDMTrainer
from ldm.utils import setup_logging

accelerator_kwargs = {"log_with": "wandb", "find_unused_parameters": True}
find_unused_params = accelerator_kwargs.pop("find_unused_parameters", True)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_params)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerator_kwargs)

setup_logging(accelerator)
logger = logging.getLogger(__name__)

model = LDM(**MODEL_CFG)

train_dataset = LeRobotMixedWindowDataset(
    DATASET_CFG,
    image_size=DATA_CFG["image_size"],
    max_frames=DATA_CFG["max_frames"],
    seed=2026,
    action_dim=MODEL_CFG.get("action_dim", 0),
    split="train",
    epoch_size=LEROBOT_DATALOADER_CFG["train_epoch_size"],
    action_sample_ratio=LEROBOT_DATALOADER_CFG["action_sample_ratio"],
    action_zero_eps=LEROBOT_DATALOADER_CFG["action_zero_eps"],
    deterministic_val=False,
    batch_level_same_episode=LEROBOT_DATALOADER_CFG.get("batch_level_same_episode", False),
    local_batch_size=TRAIN_CFG["batch_size"],
    batch_start_jitter=LEROBOT_DATALOADER_CFG.get("batch_start_jitter", 0),
    frame_strides=LEROBOT_DATALOADER_CFG.get("frame_strides", [1]),
    frame_stride_weights=LEROBOT_DATALOADER_CFG.get("frame_stride_weights"),
    rotation_accumulate=LEROBOT_DATALOADER_CFG.get("rotation_accumulate", "sum"),
)
val_dataset = LeRobotMixedWindowDataset(
    VAL_DATASET_CFG,
    image_size=DATA_CFG["image_size"],
    max_frames=DATA_CFG["max_frames"],
    seed=2026,
    action_dim=MODEL_CFG.get("action_dim", 0),
    split="val",
    epoch_size=LEROBOT_DATALOADER_CFG["val_epoch_size"],
    action_sample_ratio=0.0,
    action_zero_eps=LEROBOT_DATALOADER_CFG["action_zero_eps"],
    deterministic_val=True,
    val_windows_per_dataset=LEROBOT_DATALOADER_CFG["val_windows_per_dataset"],
    batch_level_same_episode=False,
    local_batch_size=1,
    batch_start_jitter=0,
    frame_strides=LEROBOT_DATALOADER_CFG.get("frame_strides", [1]),
    frame_stride_weights=LEROBOT_DATALOADER_CFG.get("frame_stride_weights"),
    rotation_accumulate=LEROBOT_DATALOADER_CFG.get("rotation_accumulate", "sum"),
)

save_folder = str(OUTPUT_ROOT)
run_name = "ldm"
run_id = ""
save_postfix = "_" + run_id if run_id else ""
results_folder = os.path.join(save_folder, run_name + save_postfix)
checkpoint_folder = os.path.join(results_folder, "checkpoints")
os.makedirs(results_folder, exist_ok=True)
os.makedirs(checkpoint_folder, exist_ok=True)
wandb_dir = os.environ.get("WANDB_DIR", save_folder)
os.makedirs(wandb_dir, exist_ok=True)
wandb_mode = os.environ.get("WANDB_MODE", "offline")

wandb_kwargs = {
    "wandb": {
        "mode": wandb_mode,
        "dir": wandb_dir,
        "name": results_folder.split("/")[-1],
        "config": None,
        "id": run_name if not run_id else f"{run_name}_{run_id}",
        "resume": "allow",
    }
}

ckpt_path = os.path.join(checkpoint_folder, "ldm_model_latest.pt")

trainer = LDMTrainer(
    model,
    accelerator,
    train_dataset,
    results_folder=results_folder,
    checkpoint_folder=checkpoint_folder,
    val_dataset=val_dataset,
    resume_checkpoint_path=ckpt_path if os.path.exists(ckpt_path) else None,
    wandb_kwargs=wandb_kwargs,
    **TRAIN_CFG,
)

trainer.train()
