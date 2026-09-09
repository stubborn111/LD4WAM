"""la_tri_system IDM architecture: video + latent_action + action, two-stage.

A three-expert MoT — Wan video DiT + a learnable-query Latent Action Expert +
the shared ActionDiT — trained and run with FastWAM-IDM two-stage semantics
("solve action from the predicted video"). Non-autoregressive: actions are
flow-matched in parallel; the latent-action stream is produced in one shot
during the clean/conditioning pass (zero extra inference steps).

The IDM forward / compute_loss / generate logic below was adapted from
OpenWAM's dual-system IDM architecture and the 3-stream wiring from its
tri-system, but neither is a dependency — only the neutral
:class:`BaseWAMArchitecture`, :class:`ActionDiT`, and
:class:`LatentActionExpert` / :class:`LaTriSystemIDMMoTDriver`.

Loss = ``lambda_video * L_video + lambda_action * L_action + lambda_la * L_la``,
where ``L_la`` is a plain masked MSE against an offline-precomputed LAM target
(NOT flow matching).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.la_tri_system.mot_driver import LaTriSystemIDMMoTDriver
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.utils.common import resolve_bridge_layers
from openwam.model.latent_action_backbone import LatentActionExpert, LatentActionExpertConfig

logger = logging.getLogger(__name__)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@register_architecture(
    "la_tri_system_idm",
    status="experimental",
    note="la_tri_system IDM: video + latent_action (learnable query, LAM-supervised) + action, "
    "two-stage denoising (video first, then action with frozen video+la KV).",
    framework="la_tri_system",
    variant="idm",
)
class LaTriSystemIDMArchitecture(BaseWAMArchitecture):
    """Video + latent-action + action MoT with IDM two-stage train/inference."""

    # The latent-action expert is supervised but also feeds the action stream
    # through joint attention; it must stay trainable.
    _NEVER_FREEZE = frozenset({"latent_action_expert"})

    # Probability of noising the cond-video branch during training (IDM
    # robustness, as in OpenWAM's IDM).
    video_cond_noise_prob: float = 0.5

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.latent_action_expert: Optional[LatentActionExpert] = None
        # The frozen online LAM is held OUTSIDE the nn.Module tree (via __dict__
        # + the `latent_action_model` property below) and built LAZILY on first
        # training use, so nothing is created here. `_lam_cfg` holds the config
        # block until then; None means the offline la_gt fallback.
        self._lam_cfg = None
        self._num_la_tokens: Optional[int] = None  # la tokens to emit (online-LAM layout)
        self._mot_driver: LaTriSystemIDMMoTDriver | None = None
        self._mot_driver_kwargs: dict = {}
        self._last_la_pred: Optional[Tensor] = None
        if cfg is None:
            return
        if self.video_backbone is None:
            return

        cfg = dict(cfg) if isinstance(cfg, dict) else {k: v for k, v in cfg.items()}
        cfg.setdefault("num_dit_layers", self.video_backbone.num_layers)
        cfg.setdefault("video_dim", self.video_backbone.dim)
        cfg.setdefault("num_heads", self.video_backbone.num_heads)
        cfg.setdefault("attn_head_dim", self.video_backbone.head_dim)
        cfg.setdefault("bridge_interval", 1)

        bl = resolve_bridge_layers(cfg, num_layers=self.video_backbone.num_layers)
        video_dim = int(self.video_backbone.dim)
        num_heads = int(_cfg_get(cfg, "num_heads", self.video_backbone.num_heads))
        attn_head_dim = int(_cfg_get(cfg, "attn_head_dim", self.video_backbone.head_dim))
        text_dim = int(_cfg_get(cfg, "text_dim", 4096))
        action_dim_hidden = int(_cfg_get(cfg, "dim", 1024))

        self._init_proprio_context(cfg, text_dim=text_dim)

        attention_mask_mode = str(_cfg_get(cfg, "attention_mask_mode", "joint"))
        if attention_mask_mode != "joint":
            raise ValueError(
                "la_tri_system IDM fixes attention_mask_mode='joint' to preserve the IDM "
                "teacher-forcing / two-stage mask semantics. Do not override it."
            )

        # --- Action expert (shared ActionDiT, idm variant) ---
        self.action_backbone = ActionDiT(
            action_dim=int(_cfg_get(cfg, "action_dim", 20)),
            dim=action_dim_hidden,
            ffn_dim=int(_cfg_get(cfg, "ffn_dim", 4 * action_dim_hidden)),
            num_heads=num_heads,
            num_layers=len(bl),
            video_dim=video_dim,
            bridge_layers=bl,
            variant="idm",
            attn_head_dim=attn_head_dim,
            text_dim=text_dim,
        )

        # --- Latent action expert (learnable query, no scheduler) ---
        la_cfg_dict = _cfg_get(cfg, "latent_action", {}) or {}
        la_cfg = LatentActionExpertConfig(
            la_dim=int(_cfg_get(la_cfg_dict, "la_dim", 512)),
            la_hidden_dim=int(_cfg_get(la_cfg_dict, "la_hidden_dim", 512)),
            la_ffn_dim=int(_cfg_get(la_cfg_dict, "la_ffn_dim", 2048)),
            num_layers=self.video_backbone.num_layers,
            k_la=int(_cfg_get(la_cfg_dict, "k_la", 4)),
            max_la_len=int(_cfg_get(la_cfg_dict, "max_la_len", 1024)),
            eps=float(_cfg_get(la_cfg_dict, "eps", 1e-6)),
        )
        self.latent_action_expert = LatentActionExpert(
            la_cfg, wan_dim=self.video_backbone.dim, wan_num_heads=self.video_backbone.num_heads
        )
        self.la_dim = la_cfg.la_dim
        self.k_la = la_cfg.k_la

        # --- Online LAM (optional): compute the la supervision target on the fly
        # (like the VAE encodes RGB->latents) instead of loading it offline. When
        # a ``latent_action_model`` block is present, prepare_inputs runs this
        # frozen LAM on the raw obs frames each step to produce ``la_gt``; when
        # absent, prepare_inputs falls back to the offline la_gt in the batch.
        #
        # Held OUTSIDE the nn.Module tree (via __dict__ / the `latent_action_model`
        # property) so accelerate/DeepSpeed's bf16 `prepare` never casts it
        # (LAQ/DINOv3 need fp32) and it stays out of training checkpoints. Built
        # LAZILY on first training use (see ``_ensure_latent_action_model``), NOT
        # here: the LAM loads from an external repo/ckpt path and inference never
        # runs it, so eager construction would crash deploy on any box without
        # those paths for a model it never uses. Deploy never reaches the online
        # path, so the LAM stays unbuilt there.
        self._lam_cfg = _cfg_get(cfg, "latent_action_model", None)
        # Number of la tokens the expert emits when there is no target to size
        # from (inference/generate, or the pretrain-without-la path). Online-LAM
        # layout = one action per adjacent RGB-frame pair = (video frames - 1).
        # None -> fall back to the per-latent-frame Fv*k_la layout.
        _n_la = _cfg_get(la_cfg_dict, "num_la_tokens", None)
        self._num_la_tokens = int(_n_la) if _n_la else None

        self.video_cond_noise_prob = float(
            _cfg_get(cfg, "idm_video_cond_noise_prob", _cfg_get(cfg, "video_cond_noise_prob", 0.5))
        )
        if not (0.0 <= self.video_cond_noise_prob <= 1.0):
            raise ValueError(f"idm_video_cond_noise_prob must be in [0, 1], got {self.video_cond_noise_prob}.")

        self._mot_driver_kwargs = {
            "mot_checkpoint_mixed_attn": bool(_cfg_get(cfg, "mot_checkpoint_mixed_attn", True)),
            "attention_mask_mode": "joint",
            "video_attention_mask_mode": str(_cfg_get(cfg, "video_attention_mask_mode", "first_frame_causal")),
            "la_video_attention_mode": str(_cfg_get(cfg, "la_video_attention_mode", "full")),
        }
        self.build_mot_driver()

    def build_mot_driver(self) -> LaTriSystemIDMMoTDriver:
        if self.video_backbone is None:
            raise RuntimeError("LaTriSystemIDMArchitecture.build_mot_driver: video_backbone is not set.")
        if self.action_backbone is None:
            raise RuntimeError("LaTriSystemIDMArchitecture.build_mot_driver: action_backbone is not set.")
        if self.latent_action_expert is None:
            raise RuntimeError("LaTriSystemIDMArchitecture.build_mot_driver: latent_action_expert is not set.")
        self._mot_driver = LaTriSystemIDMMoTDriver(
            self.video_backbone,
            self.action_backbone,
            self.latent_action_expert,
            **self._mot_driver_kwargs,
        )
        return self._mot_driver

    @property
    def mot_driver(self) -> LaTriSystemIDMMoTDriver | None:
        return self._mot_driver

    @property
    def latent_action_model(self):
        """Frozen online LAM, held outside the nn.Module tree (see __init__).

        Kept off ``_modules`` so accelerate/DeepSpeed's bf16 ``prepare`` never
        casts it (LAQ/DINOv3 require fp32) and it is never written into training
        checkpoints. Returns None when no ``latent_action_model`` block is set
        (offline la_gt fallback) or before the lazy build. Device/dtype handled
        by ``set_dtype_device`` / the lazy builder.
        """
        return self.__dict__.get("_lam", None)

    def _checkpoint_excluded_prefixes(self) -> tuple[str, ...]:
        """Exclude the frozen LAM from the safetensors checkpoint.

        New checkpoints never carry it (held outside the module tree), but an
        OLD checkpoint trained while the LAM was still a registered submodule
        has ``latent_action_model.*`` keys — tolerating them here lets those
        load cleanly against this (LAM-less) module tree. The LAM is rebuilt
        from its own external ckpt when training needs it, and unused at
        inference, so it is neither required nor saved.
        """
        return super()._checkpoint_excluded_prefixes() + ("latent_action_model.",)

    def _ensure_latent_action_model(self):
        """Lazily build the frozen online LAM on first training use.

        No-op when no ``latent_action_model`` block was configured (offline
        fallback) or when it is already built. Kept out of ``__init__`` so
        deploy — which never reaches this path — has no dependency on the
        training-box LAM repo/ckpt paths. See the note in ``__init__``.
        """
        if self.latent_action_model is not None or self._lam_cfg is None:
            return
        from openwam.model.latent_action_model import build_latent_action_model

        lam = build_latent_action_model(self._lam_cfg)
        lam.eval()
        for p in lam.parameters():
            p.requires_grad_(False)
        if int(lam.la_dim) != int(self.la_dim):
            raise ValueError(
                f"latent_action_model.la_dim ({lam.la_dim}) must equal the "
                f"expert latent_action.la_dim ({self.la_dim}). The expert predicts a target of "
                "exactly the LAM's flattened latent-action dimension."
            )
        # Hold OUTSIDE the nn.Module tree (see the property) and PIN fp32 — the
        # bf16 training dtype must not reach LAQ/DINOv3; encode() casts its output
        # back to the caller dtype. Mirrors set_dtype_device.
        lam.to(device=self.device, dtype=torch.float32)
        self.__dict__["_lam"] = lam

    def freeze_modules(self, names: list[str]) -> list[str]:
        rejected = self._NEVER_FREEZE & set(names)
        if rejected:
            raise ValueError(
                f"la_tri_system: refusing to freeze {rejected}. The latent-action expert is "
                "supervised and feeds the action stream through joint attention; it must stay trainable."
            )
        return super().freeze_modules(names)

    def set_dtype_device(self, dtype, device):
        super().set_dtype_device(dtype, device)
        if self.latent_action_expert is not None:
            self.latent_action_expert.to(dtype=dtype, device=device)
        # The frozen LAM lives outside the nn.Module tree, so neither super() nor
        # accelerate `prepare` ever touch it — move it explicitly and PIN it to
        # fp32 (LAQ/DINOv3 need fp32; the bf16 training dtype must NOT reach it).
        # encode() casts its output back to the caller dtype.
        if self.latent_action_model is not None:
            self.latent_action_model.to(device=device, dtype=torch.float32)

    def _iter_zero3_external_params(self):
        """Raw-access leaves read by the MoT driver outside owners' ``__call__``.

        video + action ``block.modulation`` (read in ``pre_attn_at_layer_for_compile``),
        plus the latent-action blocks' ``self_attn`` Q/K/V projection weights are
        accessed through normal ``__call__`` (the la pre-attn reads ``block.self_attn``
        submodules), so only ``modulation`` leaves need registering — same set as
        OpenWAM's joint self-attention MoT.
        """
        vb = self.video_backbone
        dit = getattr(vb, "_dit", None) if vb is not None else None
        if dit is not None:
            for block in getattr(dit, "blocks", ()):
                p = getattr(block, "modulation", None)
                if p is not None:
                    yield p
        ab = self.action_backbone
        if ab is not None:
            for block in getattr(ab, "blocks", ()):
                p = getattr(block, "modulation", None)
                if p is not None:
                    yield p

    @torch.no_grad()
    def prepare_inputs(self, batch: list[dict]) -> dict:
        """Base aggregation + ``la_gt`` / ``la_gt_mask`` collection.

        Two sources for the latent-action target:

        - **Online LAM** (``latent_action_model`` configured): run the frozen LAM
          on the raw obs frames (still present in the batch under ``"video"``),
          exactly like the VAE encodes RGB->latents. Produces one action per
          adjacent frame pair -> ``la_gt`` shape ``(B, T-1, la_dim)``, mask
          all-True. This is the primary path (no offline precompute needed).
        - **Offline fallback** (no LAM): read a precomputed ``la_gt`` per sample,
          shape ``(Fv*k_la, la_dim)`` (or ``(la_dim, Fv, k_la, 1)``). All-or-none
          per batch. When absent, ``compute_loss`` simply skips the la term.
        """
        import numpy as np

        if isinstance(batch, dict):
            batch = [batch]
        inputs = super().prepare_inputs(batch)

        # ---- Online LAM path (preferred when configured) ----
        if self._lam_cfg is not None:
            self._ensure_latent_action_model()  # lazy build on first training use
            la_gt = self._encode_la_targets_online(batch)
            if la_gt is not None:
                inputs["la_gt"] = la_gt
                inputs["la_gt_mask"] = self._la_mask_from_video(batch, la_gt)
            return inputs

        # ---- Offline fallback (original behavior) ----
        la_flags = [s.get("la_gt") is not None for s in batch]
        if not any(la_flags):
            return inputs
        if not all(la_flags):
            raise ValueError("Mixed la_gt in batch: every sample must carry la_gt, or none.")

        _dtype, _device = self.dtype, self.device
        total_la_expected = None
        la_list, la_mask_list = [], []
        for s in batch:
            la = s["la_gt"]
            if isinstance(la, np.ndarray):
                la = torch.from_numpy(la)
            # Accept (la_dim, Fv, k_la, 1) → (Fv*k_la, la_dim).
            if la.ndim == 4:
                la = la.squeeze(-1)  # (la_dim, Fv, k_la)
                la = la.permute(1, 2, 0).reshape(-1, la.shape[0])  # (Fv*k_la, la_dim)
            elif la.ndim != 2:
                raise ValueError(f"la_gt must be (Fv*k_la, la_dim) or (la_dim, Fv, k_la, 1), got {tuple(la.shape)}")
            if la.shape[-1] != self.la_dim:
                raise ValueError(f"la_gt last dim {la.shape[-1]} must equal la_dim={self.la_dim}.")
            if la.shape[0] % self.k_la != 0:
                raise ValueError(f"la_gt token count {la.shape[0]} must be a multiple of k_la={self.k_la}.")
            if total_la_expected is None:
                total_la_expected = la.shape[0]
            elif la.shape[0] != total_la_expected:
                raise ValueError("Inconsistent la_gt token count across batch; uniform length required.")
            la_list.append(la.to(dtype=_dtype, device=_device))

            m = s.get("la_gt_mask")
            if m is None:
                m = torch.ones(la.shape[0], dtype=torch.bool)
            elif isinstance(m, np.ndarray):
                m = torch.from_numpy(m)
            la_mask_list.append(m.to(dtype=torch.bool, device=_device).reshape(-1))

        inputs["la_gt"] = torch.stack(la_list, dim=0)  # (B, Fv*k_la, la_dim)
        inputs["la_gt_mask"] = torch.stack(la_mask_list, dim=0)  # (B, Fv*k_la)
        return inputs

    @torch.no_grad()
    def _encode_la_targets_online(self, batch: list[dict]) -> Optional[Tensor]:
        """Run the frozen LAM on the raw obs frames -> ``(B, T-1, la_dim)``.

        Reads ``sample["video"]`` (the same RGB frames the VAE encodes; present
        in the batch before base aggregation consumes them). Tolerant to the
        common frame containers: a per-sample sequence of PIL images / HxWx3
        arrays / CHW tensors, or a stacked ``(T,3,H,W)`` / ``(T,H,W,3)`` tensor.
        Returns None if a sample has no frames.
        """
        import numpy as np
        from PIL import Image

        def _frame_to_chw(im) -> Tensor:
            if isinstance(im, Image.Image):
                arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0  # (H,W,3)
                return torch.from_numpy(arr).permute(2, 0, 1)
            if isinstance(im, np.ndarray):
                t = torch.from_numpy(im.astype(np.float32))
            elif torch.is_tensor(im):
                t = im.float()
            else:
                raise TypeError(f"Unsupported frame type for online LAM: {type(im)}")
            if t.ndim == 3 and t.shape[0] != 3 and t.shape[-1] == 3:  # HWC -> CHW
                t = t.permute(2, 0, 1)
            if t.max() > 1.5:  # 0..255 -> 0..1
                t = t / 255.0
            return t

        def _sample_to_clip(v) -> Tensor:
            # v: sequence of frames, or a stacked tensor (T,3,H,W)/(T,H,W,3).
            if torch.is_tensor(v) or isinstance(v, np.ndarray):
                t = v.float() if torch.is_tensor(v) else torch.from_numpy(np.asarray(v, dtype=np.float32))
                if t.ndim == 4 and t.shape[1] != 3 and t.shape[-1] == 3:  # (T,H,W,3)->(T,3,H,W)
                    t = t.permute(0, 3, 1, 2)
                if t.max() > 1.5:
                    t = t / 255.0
                return t
            return torch.stack([_frame_to_chw(f) for f in v], dim=0)  # (T,3,H,W)

        clips = []
        for s in batch:
            v = s.get("video")
            if v is None:
                return None
            clips.append(_sample_to_clip(v))
        # Assemble the LAM input in the model dtype, like every other input
        # (base.prepare_inputs builds latents/actions/... in self.dtype). We do
        # NOT convert the LAM output here: the LAM upcasts to fp32 internally for
        # its own forward and hands the target back in this same (model) dtype.
        videos = torch.stack(clips, dim=0).to(device=self.device, dtype=self.dtype)  # (B,T,3,H,W) [0,1]
        return self.latent_action_model.encode(videos)  # (B, T-1, la_dim) in model dtype

    def _la_mask_from_video(self, batch: list[dict], la_gt: Tensor) -> Tensor:
        """Derive the la validity mask from the reader's frame-level ``video_mask``.

        A window shorter than the video length is padded by repeating the last
        real frame, so transition ``t`` (frame t -> t+1) is valid iff BOTH frames
        are real: ``la_mask[t] = video_mask[t] & video_mask[t+1]``. Falls back to
        all-True for samples that carry no ``video_mask``.
        """
        n_la = la_gt.shape[1]
        masks = []
        for s in batch:
            vm = s.get("video_mask")
            if vm is None:
                masks.append(torch.ones(n_la, dtype=torch.bool))
                continue
            vm = vm if torch.is_tensor(vm) else torch.as_tensor(vm)
            vm = vm.to(dtype=torch.bool).reshape(-1)  # (T,)
            tm = vm[:-1] & vm[1:]  # (T-1,) transition valid iff both frames real
            if tm.numel() != n_la:  # frame/la length drift → be safe, don't mask
                tm = torch.ones(n_la, dtype=torch.bool)
            masks.append(tm)
        return torch.stack(masks, dim=0).to(device=la_gt.device)  # (B, T-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio_state: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        cond_video_latents: Optional[Tensor] = None,
        cond_video_timestep: Optional[Tensor] = None,
        **pipeline_inputs,
    ):
        """Dispatch: IDM 3-branch training (``cond_video_latents`` set) vs the
        video-only fallback. Returns ``(video_pred, action_pred)`` and stashes
        ``la_pred`` on ``self._last_la_pred`` (the 2-tuple keeps the base
        ``compute_loss`` contract; ``compute_loss`` here reads the stash)."""
        if (cond_video_latents is None) != (cond_video_timestep is None):
            raise ValueError(
                "LaTriSystemIDMArchitecture.forward: cond_video_latents and cond_video_timestep "
                "must be passed together (both or neither)."
            )
        if cond_video_latents is not None:
            return self._forward_idm_training(
                noisy_actions=noisy_actions,
                action_timestep=action_timestep,
                cond_video_latents=cond_video_latents,
                cond_video_timestep=cond_video_timestep,
                proprio_state=proprio_state,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                **pipeline_inputs,
            )

        # Video-only fallback (no actions / inference stage 1).
        self._last_la_pred = None
        vb = self.video_backbone
        if vb is None:
            raise RuntimeError("video_backbone is None")
        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio_state)
        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )
        for block_id in range(vb.num_layers):
            vstate = vb.run_block(block_id, vstate)
        return vb.finalize(vstate), None

    def _forward_idm_training(
        self,
        *,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        cond_video_latents: Tensor,
        cond_video_timestep: Tensor,
        proprio_state: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ):
        vb = self.video_backbone
        ab = self.action_backbone
        la = self.latent_action_expert
        if vb is None:
            raise RuntimeError("video_backbone is None")

        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio_state)
        action_context = pipeline_inputs.get("context")
        action_context_mask = pipeline_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)

        # IDM needs token-wise (4D) video t_mod so noisy + cond branches concat
        # along the sequence dim with per-branch timesteps.
        pipeline_inputs["force_per_token_t_mod"] = True
        pipeline_inputs["zero_clean_prefix_t_mod"] = True

        vstate_noisy = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )
        cond_inputs = dict(pipeline_inputs)
        cond_inputs["latents"] = cond_video_latents
        cond_inputs["timestep"] = cond_video_timestep
        vstate_cond = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **cond_inputs,
        )

        driver = self._mot_driver or self.build_mot_driver()
        B = int(vstate_cond.hidden_states.shape[0])
        Fv = int(vstate_cond.grid_frames)
        # la token count: prefer the target length stashed by compute_loss (online
        # LAM -> T_video-1) or the configured num_la_tokens; else the per-latent-
        # frame Fv*k_la layout (offline fallback).
        n_la = getattr(self, "_la_tokens_next", None) or self._num_la_tokens
        _la_dtype, _la_device = vstate_cond.hidden_states.dtype, vstate_cond.hidden_states.device
        if n_la is not None:
            lastate = la.prepare_state(B, num_la_tokens=int(n_la), dtype=_la_dtype, device=_la_device)
        else:
            lastate = la.prepare_state(B, Fv, dtype=_la_dtype, device=_la_device)

        if noisy_actions is not None and ab is not None:
            astate = ab.prepare_state(
                noisy_actions,
                action_timestep,
                context=action_context,
                context_mask=action_context_mask,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
            vstate_noisy, vstate_cond, lastate, astate = driver.run_idm_training_loop(
                vstate_noisy,
                vstate_cond,
                lastate,
                astate,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
            action_pred = ab.extract_prediction(astate)
            self._last_la_pred = la.extract_prediction(lastate)
        else:
            # Pretrain (lambda_action == 0 → noisy_actions is None): still run the
            # latent-action stream so la_pred exists and the la loss can supervise
            # it. Without this, pretrain would silently train video only.
            vstate_noisy, vstate_cond, lastate = driver.run_video_and_la_training_loop(
                vstate_noisy,
                vstate_cond,
                lastate,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
            action_pred = None
            self._last_la_pred = la.extract_prediction(lastate)

        return vb.finalize(vstate_noisy), action_pred

    # ------------------------------------------------------------------
    # Training loss: IDM 3-branch + latent-action masked MSE
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        *,
        actions: Optional[Tensor] = None,
        lambda_video: float = 1.0,
        lambda_action: float = 1.0,
        lambda_la: float = 0.0,
        current_step: int = 0,
        decoupled_sampler=None,
        action_timestep_per_token: bool = False,
        la_gt: Optional[Tensor] = None,
        la_gt_mask: Optional[Tensor] = None,
        **inputs,
    ) -> dict:
        vb = self.video_backbone
        ab = self.action_backbone
        action_scheduler = ab.scheduler
        _dtype = self.dtype
        _device = self.device

        if actions is None:
            actions = inputs.pop("actions", None)
        else:
            inputs.pop("actions", None)
        # la_gt / la_gt_mask are loss-side only — they must NOT flow into forward.
        la_gt = la_gt if la_gt is not None else inputs.pop("la_gt", None)
        la_gt_mask = la_gt_mask if la_gt_mask is not None else inputs.pop("la_gt_mask", None)

        if action_timestep_per_token:
            raise ValueError("action_timestep_per_token=True is not supported for la_tri_system IDM.")

        max_tb = int(inputs.pop("max_timestep_boundary", 1) * len(vb.scheduler.timesteps))
        min_tb = int(inputs.pop("min_timestep_boundary", 0) * len(vb.scheduler.timesteps))
        input_latents = inputs["input_latents"]
        B = input_latents.shape[0]

        # ---- Branch A: noisy video ----
        if decoupled_sampler is not None:
            video_t, decoupled_action_t = decoupled_sampler.sample_timesteps(B, current_step=current_step, device="cpu")
            num_ts = len(vb.scheduler.timesteps)
            video_timestep_ids = (
                (video_t / decoupled_sampler.num_train_timesteps * num_ts).long().clamp(min_tb, max_tb - 1)
            )
        else:
            decoupled_action_t = None
            video_timestep_ids = torch.randint(min_tb, max_tb, (B,))
        video_timesteps = vb.scheduler.timesteps[video_timestep_ids].to(dtype=_dtype, device=_device)
        video_sigmas = vb.scheduler.sigmas[video_timestep_ids].to(dtype=_dtype, device=_device)

        video_noise = torch.randn_like(input_latents)
        sigma_bc = video_sigmas.view(B, 1, 1, 1, 1)
        latents_noisy = (1 - sigma_bc) * input_latents + sigma_bc * video_noise
        video_target = video_noise - input_latents
        if inputs.get("first_frame_latents") is not None:
            latents_noisy[:, :, 0:1] = inputs["first_frame_latents"]

        # ---- Branch B: noisy action ----
        noisy_actions = action_target = action_timesteps = action_timestep_ids = None
        if lambda_action > 0 and actions is not None:
            if decoupled_action_t is not None:
                num_ts_a = len(action_scheduler.timesteps)
                action_timestep_ids = (
                    (decoupled_action_t / decoupled_sampler.num_train_timesteps * num_ts_a).long().clamp(0, num_ts_a - 1)
                )
            else:
                action_timestep_ids = torch.randint(0, len(action_scheduler.timesteps), (B,))
            action_timesteps = action_scheduler.timesteps[action_timestep_ids].to(dtype=_dtype, device=_device)
            action_sigmas = action_scheduler.sigmas[action_timestep_ids].to(dtype=_dtype, device=_device)
            actions = actions.to(dtype=_dtype, device=_device)
            if actions.dim() == 2:
                actions = actions.unsqueeze(0)
            action_noise = torch.randn_like(actions)
            a_sigma_bc = action_sigmas.view(B, 1, 1)
            noisy_actions = action_scheduler.add_noise(actions, action_noise, a_sigma_bc)
            action_target = action_scheduler.training_target(actions, action_noise)

        # ---- Branch C: teacher-forcing cond video (optionally noised) ----
        cond_noise_mask = torch.rand((B,), device=_device) < self.video_cond_noise_prob
        latents_cond = input_latents.clone()
        cond_video_timesteps = torch.zeros((B,), dtype=_dtype, device=_device)
        if bool(cond_noise_mask.any()):
            cond_ids = torch.randint(min_tb, max_tb, (B,))
            cond_sigmas = vb.scheduler.sigmas[cond_ids].to(dtype=_dtype, device=_device)
            cond_sampled_ts = vb.scheduler.timesteps[cond_ids].to(dtype=_dtype, device=_device)
            cond_video_timesteps = torch.where(cond_noise_mask, cond_sampled_ts, cond_video_timesteps)
            noise_cond = torch.randn_like(input_latents)
            cond_sigma_bc = cond_sigmas.view(B, 1, 1, 1, 1)
            latents_cond_noisy = (1 - cond_sigma_bc) * input_latents + cond_sigma_bc * noise_cond
            latents_cond = torch.where(cond_noise_mask.view(B, 1, 1, 1, 1), latents_cond_noisy, latents_cond)
        if inputs.get("first_frame_latents") is not None:
            latents_cond[:, :, 0:1] = inputs["first_frame_latents"]

        # ---- Forward ----
        forward_inputs = dict(inputs)
        # base.prepare_inputs emits the aggregated proprio under "proprio" (as in
        # OpenWAM's IDM); "proprio_state" is the forward() kwarg name, not a
        # batch key. Popping the wrong key silently yielded None -> proprio raise.
        proprio_state = forward_inputs.pop("proprio", None)
        forward_inputs.pop("proprio_mask", None)
        use_grad_ckpt = forward_inputs.pop("use_gradient_checkpointing", False)
        use_grad_ckpt_offload = forward_inputs.pop("use_gradient_checkpointing_offload", False)
        forward_inputs.pop("action_is_pad", None)
        forward_inputs.pop("video_is_pad", None)
        forward_inputs["latents"] = latents_noisy

        # ZeRO-3 external-parameter registration: only present on codebases that
        # expose the protocol (the MoT driver reads block.modulation leaves raw
        # under DeepSpeed ZeRO-3). Guarded so this arch also works on codebases
        # without it — on non-ZeRO-3 paths it is a no-op anyway.
        if hasattr(self, "_register_zero3_externals"):
            self._register_zero3_externals()
        # Size the la stream to the target: online LAM gives one action per
        # adjacent frame pair, so la_pred must have la_gt.shape[1] tokens. Falls
        # back to the configured num_la_tokens (or Fv*k_la) when no target.
        self._la_tokens_next = int(la_gt.shape[1]) if la_gt is not None else self._num_la_tokens
        video_noise_pred, action_noise_pred = self(
            noisy_actions if lambda_action > 0 else None,
            action_timesteps if lambda_action > 0 else None,
            proprio_state=proprio_state,
            use_gradient_checkpointing=use_grad_ckpt,
            use_gradient_checkpointing_offload=use_grad_ckpt_offload,
            cond_video_latents=latents_cond,
            cond_video_timestep=cond_video_timesteps,
            timestep=video_timesteps,
            **forward_inputs,
        )
        la_pred = self._last_la_pred

        # ---- Video loss ----
        loss_video = self._compute_video_loss(video_noise_pred, video_target, video_timestep_ids, inputs, _device)

        out = {"loss_video": lambda_video * loss_video.detach()}
        loss = lambda_video * loss_video

        # ---- Action loss ----
        if lambda_action > 0 and action_noise_pred is not None:
            loss_action = self._compute_action_loss(
                action_noise_pred, action_target, action_timestep_ids, action_scheduler, inputs, _device
            )
            loss = loss + lambda_action * loss_action
            out["loss_action"] = lambda_action * loss_action.detach()
        else:
            out["loss_action"] = torch.tensor(0.0, device=loss_video.device)

        # ---- Latent-action loss (plain masked MSE, NOT flow matching) ----
        if lambda_la > 0 and la_gt is not None and la_pred is not None:
            loss_la = self._compute_la_loss(la_pred, la_gt, la_gt_mask, _device)
            loss = loss + lambda_la * loss_la
            out["loss_la"] = lambda_la * loss_la.detach()
        else:
            out["loss_la"] = torch.tensor(0.0, device=loss_video.device)

        out["loss"] = loss
        return out

    def _compute_la_loss(
        self, la_pred: Tensor, la_gt: Tensor, la_gt_mask: Optional[Tensor], device
    ) -> Tensor:
        """Masked MSE between the expert's la_pred and the LAM target.

        la_pred / la_gt: ``(B, N_la, la_dim)`` — online LAM: ``N_la = T_video-1``
        (one action per adjacent frame pair); offline fallback: ``N_la = Fv*k_la``.
        la_gt_mask: ``(B, N_la)`` bool, True = valid (padded-frame transitions are
        masked out; see ``_la_mask_from_video``).
        """
        la_gt = la_gt.to(device=device, dtype=torch.float32)
        if la_pred.shape != la_gt.shape:
            raise ValueError(
                f"la_pred shape {tuple(la_pred.shape)} must match la_gt shape {tuple(la_gt.shape)}. "
                "The expert's la token count must equal the LAM target's (online: T_video-1)."
            )
        per_token = (la_pred.float() - la_gt).pow(2).mean(dim=-1)  # (B, N_la)
        if la_gt_mask is None:
            return per_token.mean()
        m = la_gt_mask.to(device=device, dtype=torch.float32)
        return (per_token * m).sum() / (m.sum() + 1e-6)

    # ------------------------------------------------------------------
    # Inference: two-stage generation
    # ------------------------------------------------------------------

    def _run_video_only_backbone(
        self, pipeline_inputs: dict, *, use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tensor:
        """Run the video backbone once on already-prepared pipeline inputs.

        Caller owns proprio-context augmentation (so stage 1 does not re-enter
        ``forward()`` and append proprio twice).
        """
        vb = self.video_backbone
        if vb is None:
            raise RuntimeError("video_backbone is None")
        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )
        for block_id in range(vb.num_layers):
            vstate = vb.run_block(block_id, vstate)
        return vb.finalize(vstate)

    @torch.no_grad()
    def generate(
        self,
        schedule,
        prompt: str,
        *,
        vace_video=None,
        first_frame_image=None,
        num_frames: int = 49,
        action_num_frames: Optional[int] = None,
        height: int = 384,
        width: int = 320,
        seed: int = 42,
        tiled: bool = True,
        input_video_latents: Optional[Tensor] = None,
        num_inference_steps: int = 50,
        shift: float = 5.0,
        tile_size: tuple = None,
        tile_stride: tuple = None,
        decode_video: bool = True,
        profile: bool = False,
        vace_cache: Optional[dict] = None,
        prompt_embed_cache: Optional[dict] = None,
        proprio: Optional[Tensor] = None,
    ) -> dict:
        """Two-stage IDM generation with the latent-action stream produced once.

        Stage 1: denoise video independently (video DiT only).
        Stage 2: freeze the denoised video; one clean pass produces ``la_pred``
                 and caches cond-video+la K/V; then the action chunk is denoised
                 against those frozen K/V. ``la`` costs zero extra steps.
        """
        from tqdm import tqdm

        self.eval()
        vb = self.video_backbone
        ab = self.action_backbone
        la = self.latent_action_expert
        device = self.device
        dtype = self.dtype

        action_num_frames = int(action_num_frames if action_num_frames is not None else num_frames)

        # Wan uses its own native tiling grid; tile_size/tile_stride are not
        # forwarded (as in OpenWAM's IDM). The backbone owns the inference
        # input construction — there is no ``InferenceInputs`` dataclass here.
        inputs_shared = vb.preprocess_input_for_inference(
            prompt=prompt,
            vace_video=vace_video,
            first_frame_image=first_frame_image,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            num_inference_steps=num_inference_steps,
            shift=shift,
            tiled=tiled,
            vace_cache=vace_cache,
            prompt_embed_cache=prompt_embed_cache,
        )
        if input_video_latents is not None:
            inputs_shared["latents"] = input_video_latents
        ref_latents = inputs_shared.get("first_frame_latents")
        if ref_latents is not None:
            latents = inputs_shared["latents"].clone()
            latents[:, :, : ref_latents.shape[2]] = ref_latents
            inputs_shared["latents"] = latents
        if self.uses_proprioception:
            if proprio is None:
                raise ValueError("use_proprioception=True requires `proprio` during generation.")
            inputs_shared["proprio"] = proprio.to(device=device, dtype=dtype)

        proprio_arg = inputs_shared.pop("proprio", None)
        inputs_shared_with_proprio = self._append_proprio_context_token(dict(inputs_shared), proprio_arg)

        action_latents = torch.randn(
            1,
            action_num_frames - 1,
            self.action_dim,
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        num_train_ts_v = float(self.video_scheduler.num_train_timesteps)
        num_train_ts_a = float(self.action_scheduler.num_train_timesteps)

        # ---- Stage 1: video ----
        did_video_step = False
        for i in tqdm(range(len(schedule) - 1), desc="la_tri_system IDM Stage 1: Video"):
            t_v, _ = schedule[i]
            t_v_next, _ = schedule[i + 1]
            sigma_v = t_v / num_train_ts_v
            sigma_v_next = t_v_next / num_train_ts_v
            if sigma_v == sigma_v_next:
                continue
            did_video_step = True
            v_timestep = torch.tensor([t_v], dtype=dtype, device=device)
            noise_pred = self._run_video_only_backbone(
                {**inputs_shared_with_proprio, "timestep": v_timestep}
            )
            new_latents = inputs_shared_with_proprio["latents"] + noise_pred * (sigma_v_next - sigma_v)
            ref_latents = inputs_shared_with_proprio.get("first_frame_latents")
            if ref_latents is not None:
                new_latents = new_latents.clone()
                new_latents[:, :, : ref_latents.shape[2]] = ref_latents
            inputs_shared_with_proprio["latents"] = new_latents

        if not did_video_step and input_video_latents is None:
            raise ValueError(
                "la_tri_system IDM generate(): Stage 1 produced no video denoising step and no "
                "input_video_latents were provided; Stage 2 would condition on random latents."
            )

        # ---- Stage 2: clean pass (video + la) then action ----
        cond_inputs = dict(inputs_shared_with_proprio)
        cond_timestep = torch.zeros(1, dtype=dtype, device=device)
        action_context = cond_inputs.get("context")
        action_context_mask = cond_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and cond_inputs.get("seq_lens") is not None:
            seq_lens = cond_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)

        cond_vstate = vb.prepare(
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
            timestep=cond_timestep,
            **cond_inputs,
        )
        driver = self._mot_driver or self.build_mot_driver()
        Fv = int(cond_vstate.grid_frames)
        B = int(cond_vstate.hidden_states.shape[0])
        s_cond_video = int(cond_vstate.hidden_states.shape[1])
        # Inference does NOT run the LAM (the expert predicts la, zero extra
        # steps). Emit the configured number of la tokens (online-LAM layout,
        # T_video-1) or the per-latent-frame Fv*k_la layout when unset.
        _la_dtype, _la_device = cond_vstate.hidden_states.dtype, cond_vstate.hidden_states.device
        if self._num_la_tokens is not None:
            lastate = la.prepare_state(B, num_la_tokens=self._num_la_tokens, dtype=_la_dtype, device=_la_device)
        else:
            lastate = la.prepare_state(B, Fv, dtype=_la_dtype, device=_la_device)
        kv_cache, lastate, _ = driver.prefill_video_and_la(cond_vstate, lastate)
        la_pred = la.extract_prediction(lastate)
        s_la = int(lastate.la_tokens.shape[1])

        for i in tqdm(range(len(schedule) - 1), desc="la_tri_system IDM Stage 2: Action"):
            _, t_a = schedule[i]
            _, t_a_next = schedule[i + 1]
            sigma_a = t_a / num_train_ts_a
            sigma_a_next = t_a_next / num_train_ts_a
            if sigma_a == sigma_a_next:
                continue
            a_timestep = torch.tensor([t_a], dtype=dtype, device=device)
            astate = ab.prepare_state(
                action_latents, a_timestep, context=action_context, context_mask=action_context_mask
            )
            astate = driver.run_action_with_caches(
                astate, kv_cache=kv_cache, s_cond_video=s_cond_video, s_la=s_la
            )
            action_noise_pred = ab.extract_prediction(astate)
            if action_noise_pred is not None:
                action_latents = self.action_scheduler.flow_step(
                    action_noise_pred, sigma_a, sigma_a_next, action_latents
                )

        if decode_video:
            video_frames = vb.decode_video(inputs_shared_with_proprio["latents"], tiled=tiled)
        else:
            video_frames = None

        actions_out = action_latents.squeeze(0).float().cpu().numpy()
        # Deploy attaches the action normalizer as ``self.normalizer`` (base
        # attach_normalizer); there is no ``action_normalizer`` attribute, so
        # reading that would silently skip unnormalization and return raw
        # model-space actions. Use ``self.normalizer``.
        normalizer = getattr(self, "normalizer", None)
        if normalizer is not None:
            actions_out = normalizer.unnormalize(actions_out)

        la_out = la_pred.squeeze(0).float().cpu().numpy()
        return {"video": video_frames, "actions": actions_out, "latent_action": la_out}

__all__ = ["LaTriSystemIDMArchitecture"]
