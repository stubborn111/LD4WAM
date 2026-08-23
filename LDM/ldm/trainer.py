import json
import logging
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from ema_pytorch import EMA
from ldm.latent_dynamics_model import LDM
from ldm.utils import default, exists
from torch import nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def cycle(dl: DataLoader, skipped_dl: Optional[DataLoader] = None):
    if skipped_dl is not None:
        for data in skipped_dl:
            yield data
    while True:
        for data in dl:
            yield data


class LDMTrainer(nn.Module):
    def __init__(
        self,
        model: LDM,
        accelerator: Accelerator,
        dataset: Dataset,
        num_train_steps: int,
        results_folder: str,
        checkpoint_folder: Optional[str] = None,
        batch_size: int = 32,
        val_dataset: Optional[Dataset] = None,
        val_batch_size: Optional[int] = 4,
        lr: float = 1e-4,
        pretrained_init_lr_mult_factor: float = 0.1,
        weight_decay: float = 0.0,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        min_lr: float = 0.0,
        grad_accum_every: int = 1,
        max_grad_norm: float = 1.0,
        use_ema: bool = False,
        ema_update_every: int = 10,
        ema_beta: float = 0.9999,
        save_model_every: int = 1000,
        save_milestone_every: int = 10000,
        val_every_n_steps: int = 1000,
        num_val_batches_to_log: int = 5,
        num_workers: int = 4,
        prefetch_factor: int = 4,
        pin_memory: bool = True,
        log_every_n_steps: int = 50,
        metrics_shard_steps: int = 1000,
        resume_checkpoint_path: Optional[str] = None,
        milestone_optim_state: bool = True,
        wandb_kwargs: dict = {},
    ):
        super().__init__()
        self.accelerator = accelerator

        config = {}
        arguments = locals()
        for key in arguments.keys():
            if key not in [
                "self",
                "config",
                "__class__",
                "model",
                "wandb_kwargs",
                "val_dataset",
                "dataset",
                "accelerator",
            ]:
                config[key] = arguments[key]

        if hasattr(model, "__dict__"):
            model_config = {
                k: v
                for k, v in vars(model).items()
                if isinstance(
                    v, (int, float, str, bool)
                )  
            }
            config.update(model_config)

        wandb_kwargs["wandb"]["config"] = config
        self.accelerator.init_trackers(
            project_name="ldm", config=config, init_kwargs=wandb_kwargs
        )
        if self.accelerator.is_main_process:
            logger.info(f"Config:\n{config}")

        self.model = model
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.checkpoint_folder = Path(default(checkpoint_folder, self.results_folder / "checkpoints"))
        self.checkpoint_folder.mkdir(parents=True, exist_ok=True)

        self.train_dataset_returns_batches = bool(getattr(dataset, "batch_level_same_episode", False))
        train_loader_batch_size = 1 if self.train_dataset_returns_batches else batch_size
        train_drop_last = not self.train_dataset_returns_batches
        if self.train_dataset_returns_batches:
            logger.info(
                "Using dataset-batched training: dataset returns local batches of %d samples; DataLoader batch_size is set to 1.",
                getattr(dataset, "local_batch_size", batch_size),
            )
        self.dataloader = DataLoader(
            dataset,
            batch_size=train_loader_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=train_drop_last,
            prefetch_factor=prefetch_factor,
            persistent_workers=num_workers > 0,
        )

        self.val_dataloader = None
        if exists(val_dataset):
            effective_val_batch_size = default(val_batch_size, batch_size)
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=effective_val_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
                persistent_workers=num_workers > 0,
            )

        self.lr = lr
        self.pretrained_init_lr_mult_factor = pretrained_init_lr_mult_factor
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = int(lr_warmup_steps)
        self.min_lr = float(min_lr)
        self.optimizer = torch.optim.AdamW(
            model.get_trainable_parameters(
                lr,
                pretrained_init_keywords=["enc_spatial_transformer"],
                pretrained_init_lr_mult_factor=pretrained_init_lr_mult_factor,
            ),
            lr=lr,
            weight_decay=weight_decay,
        )
        for group in self.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.optimizer_base_lrs = [group["initial_lr"] for group in self.optimizer.param_groups]

        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )

        self.grad_accum_every = grad_accum_every
        self.max_grad_norm = max_grad_norm
        self.save_model_every = save_model_every
        self.save_milestone_every = save_milestone_every
        self.milestone_optim_state = milestone_optim_state
        self.val_every_n_steps = val_every_n_steps
        self.num_val_batches_to_log = num_val_batches_to_log

        self.num_train_steps = num_train_steps
        self.current_step = 0
        self.current_val_step = 0
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.metrics_shard_steps = max(1, int(metrics_shard_steps))
        self.metrics_folder = self.results_folder / "metrics"
        self.metrics_folder.mkdir(parents=True, exist_ok=True)
        self.metrics_jsonl_path = self.metrics_folder / "train_metrics_00000000_00000999.jsonl"
        self.latest_metrics_path = self.results_folder / "latest_metrics.json"

        self.use_ema = use_ema
        if self.use_ema:
            model_to_ema = self.accelerator.unwrap_model(self.model)
            self.ema_model = EMA(
                model_to_ema, beta=ema_beta, update_every=ema_update_every
            )
        else:
            self.ema_model = None

        self.resume_checkpoint_path = resume_checkpoint_path
        skipped_dl_for_resume = None
        skipped_val_dl_for_resume = None
        if (
            self.resume_checkpoint_path is not None
            and Path(self.resume_checkpoint_path).exists()
        ):
            self.load(self.resume_checkpoint_path)
            if self.current_step > 0:
                logger.info(
                    f"Map style resuming training dataloader from step {self.current_step}..."
                )
                skipped_dl_for_resume = self.maybe_skip_batches_for_resume(
                    self.current_step, self.dataloader
                )
            if self.val_dataloader is not None and self.current_val_step > 0:
                logger.info(
                    f"Map style resuming validation dataloader from step {self.current_val_step}..."
                )
                skipped_val_dl_for_resume = self.maybe_skip_batches_for_resume(
                    self.current_val_step, self.val_dataloader
                )

        self.dl_iter = cycle(self.dataloader, skipped_dl_for_resume)
        if self.val_dataloader is not None:
            self.val_dl_iter = cycle(self.val_dataloader, skipped_val_dl_for_resume)
        else:
            self.val_dl_iter = None

        self.accelerator.wait_for_everyone()

    def maybe_skip_batches_for_resume(
        self, step: int, dataloader: DataLoader
    ) -> Optional[DataLoader]:
        
        num_batches_processed = step * self.grad_accum_every
        num_batches_to_skip = num_batches_processed % len(dataloader)

        if num_batches_to_skip > 0:
            skipped_loader = self.accelerator.skip_first_batches(
                dataloader, num_batches_to_skip
            )
            logger.info(
                f"Resuming: Skipping {num_batches_to_skip} batches from dataloader."
            )
            return skipped_loader
        else:
            return None

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (
            self.accelerator.distributed_type == DistributedType.NO
            and self.accelerator.num_processes == 1
        )

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def _lr_scale_for_step(self, step: int) -> float:
        if self.lr_scheduler == "constant":
            return 1.0
        if self.lr_scheduler != "cosine":
            raise ValueError(f"Unsupported lr_scheduler: {self.lr_scheduler}")
        if self.lr_warmup_steps > 0 and step < self.lr_warmup_steps:
            return max(float(step + 1) / float(self.lr_warmup_steps), 1e-8)
        decay_steps = max(1, self.num_train_steps - self.lr_warmup_steps)
        progress = min(max((step - self.lr_warmup_steps) / decay_steps, 0.0), 1.0)
        min_scale = self.min_lr / self.lr if self.lr > 0 else 0.0
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_scale + (1.0 - min_scale) * cosine

    def _set_lr_for_step(self, step: int) -> float:
        scale = self._lr_scale_for_step(step)
        base_lr = self.lr
        for group, base_lr in zip(self.optimizer.param_groups, self.optimizer_base_lrs):
            initial_lr = group.get("initial_lr", base_lr)
            group["lr"] = initial_lr * scale
        return self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else base_lr * scale

    def _write_metrics_jsonl(self, payload: Dict[str, Any]) -> None:
        if not self.is_main:
            return
        serializable = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    serializable[key] = value.item()
            elif isinstance(value, (int, float, str, bool)) or value is None:
                serializable[key] = value
        shard_start = (int(serializable.get("step", 0)) // self.metrics_shard_steps) * self.metrics_shard_steps
        shard_end = shard_start + self.metrics_shard_steps - 1
        self.metrics_jsonl_path = self.metrics_folder / f"train_metrics_{shard_start:08d}_{shard_end:08d}.jsonl"
        with open(self.metrics_jsonl_path, "a") as f:
            f.write(json.dumps(serializable, ensure_ascii=False) + "\n")
        tmp_latest = self.latest_metrics_path.with_suffix(".json.tmp")
        with open(tmp_latest, "w") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        tmp_latest.replace(self.latest_metrics_path)

    def load(self, path: str):
        p = Path(path)
        if not p.exists():
            logger.info(f"Checkpoint not found at {str(p)}, starting from scratch.")
            return

        try:
            logger.info(f"Loading checkpoint from {str(p)}...")
            data = torch.load(p, map_location="cpu")

            model_to_load = self.accelerator.unwrap_model(self.model)

            if "model" in data:
                model_state = data["model"]
            elif "module" in data:  
                model_state = data["module"]
            else:
                model_state = (
                    data  
                )

            msg = model_to_load.load_state_dict(model_state, strict=False)
            logger.info(f"Model loaded with message: {msg}")

            if "optimizer" in data:
                self.optimizer.load_state_dict(data["optimizer"])
                for group, base_lr in zip(self.optimizer.param_groups, self.optimizer_base_lrs):
                    group["initial_lr"] = base_lr
            else:
                logger.info("Warning: Optimizer state not found in checkpoint.")

            self.current_step = int(data.get("steps", data.get("step", 0))) + 1
            self.current_val_step = int(
                data.get(
                    "val_steps",
                    data.get("val_step", self.current_step // self.val_every_n_steps),
                )
            )

            if self.use_ema and self.ema_model is not None and "ema_model" in data:
                self.ema_model.load_state_dict(data["ema_model"])

            logger.info(
                f"Resumed training from checkpoint {str(p)} at step {self.current_step}"
            )

        except Exception as e:
            logger.error(
                f"Failed to load checkpoint from {str(p)}: {e}. Starting from scratch."
            )
            self.current_step = 0

    def save(self, path: str, is_milestone: bool = False):
        if not self.is_main:
            return

        p = Path(path)
        logger.info(f"Saving checkpoint to {str(p)} at step {self.current_step}...")

        save_data = {
            "model": self.accelerator.get_state_dict(self.model),
            "steps": self.current_step,
            "val_steps": self.current_val_step,
        }

        if not is_milestone or (is_milestone and self.milestone_optim_state):
            save_data["optimizer"] = self.optimizer.state_dict()

        if self.use_ema and self.ema_model is not None:
            save_data["ema_model"] = self.ema_model.state_dict()

        try:
            tmp_path = p.with_suffix(p.suffix + ".tmp")

            self.accelerator.save(save_data, tmp_path)  
            tmp_path.replace(p)  

            logger.info(f"Checkpoint saved successfully to {str(p)}.")
        except Exception as e:
            logger.error(f"Failed to save checkpoint to {str(p)}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()  

    def train_step(self):
        step_start_time = time.perf_counter()
        self.model.train()
        total_loss_value_accum = 0.0
        total_data_time = 0.0
        current_lr = self._set_lr_for_step(self.current_step)

        for i in range(self.grad_accum_every):
            is_last_accum_step = i == self.grad_accum_every - 1

            with self.accelerator.accumulate(self.model):
                data_start_time = time.perf_counter()
                batch_data = next(self.dl_iter)
                total_data_time += time.perf_counter() - data_start_time
                videos, mask = batch_data[0], batch_data[1]
                if self.train_dataset_returns_batches:
                    videos = videos.squeeze(0)
                    mask = mask.squeeze(0)
                actions = batch_data[2] if len(batch_data) > 2 else None
                action_mask = batch_data[3] if len(batch_data) > 3 else None
                if self.train_dataset_returns_batches:
                    if actions is not None:
                        actions = actions.squeeze(0)
                    if action_mask is not None:
                        action_mask = action_mask.squeeze(0)

                loss, logs_dict = self.model(
                    videos,
                    mask,
                    step=self.current_step,
                    actions=actions,
                    action_mask=action_mask,
                )

                if torch.isnan(loss).any():
                    logger.warning(
                        f"NaN loss detected at step {self.current_step}. Skipping gradient update for this batch."
                    )
                    self.accelerator.skip_gradient_allreduce = (
                        True  
                    )
                    if (
                        is_last_accum_step
                    ):  
                        self.optimizer.zero_grad()  
                    continue  

                loss_to_backward = loss / self.grad_accum_every
                self.accelerator.backward(loss_to_backward)

                total_loss_value_accum += loss_to_backward.item()

        grad_norm_val = 0.0
        if self.max_grad_norm is not None:
            grad_norm_val = self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            grad_norm_val = grad_norm_val.item()

        self.optimizer.step()

        log_payload = {"step": self.current_step}
        for key, value in logs_dict.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                log_payload[key] = value.item()
            elif isinstance(value, (int, float)):
                log_payload[key] = value

        if "total_loss_accumulated_step" not in log_payload:  
            log_payload["total_loss_accumulated_step"] = total_loss_value_accum

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        param_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(p.detach().float(), 2.0)
                    for p in unwrapped_model.parameters()
                    if p.requires_grad
                ]
            )
        ).item()

        log_payload["param_norm"] = param_norm
        log_payload["grad_norm"] = grad_norm_val
        log_payload["lr"] = current_lr
        step_time = time.perf_counter() - step_start_time
        log_payload["data_time"] = total_data_time
        log_payload["step_time"] = step_time
        log_payload["data_time_frac"] = total_data_time / max(step_time, 1e-8)

        self.optimizer.zero_grad()

        if self.use_ema and self.ema_model is not None:
            if self.is_main:
                self.ema_model.update()

        return total_loss_value_accum, log_payload

    @torch.no_grad()
    def run_validation_and_log(self, train_step: int):
        if self.val_dl_iter is None:
            return {}

        logger.info(
            f"Running validation step {self.current_val_step} at training step {train_step}..."
        )
        model_for_eval = (
            self.ema_model if self.use_ema and self.ema_model else self.model
        )
        model_for_eval = self.accelerator.unwrap_model(model_for_eval)
        model_for_eval.eval()

        total_val_loss = 0.0
        all_val_logs = {}  

        batch_idx = 0
        while batch_idx < self.num_val_batches_to_log:
            val_batch_data = next(self.val_dl_iter)
            videos, mask = val_batch_data[0], val_batch_data[1]
            val_actions = val_batch_data[2] if len(val_batch_data) > 2 else None
            val_action_mask = val_batch_data[3] if len(val_batch_data) > 3 else None

            videos = videos.to(self.device)
            mask = mask.to(self.device)
            if val_actions is not None:
                val_actions = val_actions.to(self.device)
            if val_action_mask is not None:
                val_action_mask = val_action_mask.to(self.device)

            val_loss, val_logs_dict = model_for_eval(
                videos,
                mask,
                step=train_step,
                actions=val_actions,
                action_mask=val_action_mask,
            )

            for key, value in val_logs_dict.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    all_val_logs[key] = all_val_logs.get(key, 0.0) + value.item()
                elif isinstance(value, (int, float)):
                    all_val_logs[key] = all_val_logs.get(key, 0.0) + value
            total_val_loss += val_loss.item()

            batch_idx += 1

        avg_val_logs = {}
        if batch_idx > 0:  
            for key, value in all_val_logs.items():
                avg_val_logs[f"val/{key}"] = value / batch_idx
            avg_val_logs["val/total_loss_avg"] = total_val_loss / batch_idx
        avg_val_logs["val/train_step"] = train_step
        avg_val_logs["val/val_step"] = self.current_val_step

        logger.info(f"Validation at step {train_step}: {avg_val_logs}")

        self.current_val_step += 1
        return avg_val_logs

    def train(self):
        logger.info(
            f"Starting training from step {self.current_step} up to {self.num_train_steps} steps."
        )

        while self.current_step < self.num_train_steps:
            avg_loss_this_step, logs_train = self.train_step()

            if self.current_step % self.log_every_n_steps == 0:
                logger.info(
                    f"Step {self.current_step}/{self.num_train_steps}: {logs_train}"
                )

            logs_val = {}
            if (
                self.val_dataloader is not None
                and self.current_step % self.val_every_n_steps == 0
            ):
                logs_val = self.run_validation_and_log(self.current_step)

            if self.is_main:
                if self.current_step % self.save_model_every == 0:
                    self.save(self.checkpoint_folder / "ldm_model_latest.pt")

                if self.current_step % self.save_milestone_every == 0:
                    self.save(
                        self.checkpoint_folder
                        / f"ldm_model_milestone_{self.current_step}.pt",
                        is_milestone=True,
                    )
            if len(logs_val) > 0:
                logs = {**logs_train, **logs_val}
            else:
                logs = logs_train
            self._write_metrics_jsonl(logs)
            self.accelerator.log(logs)

            self.current_step += 1
            self.accelerator.wait_for_everyone()

        if self.is_main:
            self.save(self.checkpoint_folder / "ldm_model_final.pt", is_milestone=True)
            logger.info("Training complete.")
            if self.accelerator.trackers:
                self.accelerator.end_training()
