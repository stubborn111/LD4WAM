"""MoT driver for the ``la_tri_system`` IDM variant.

Standalone three-stream joint self-attention driver over
``[video, latent_action, action]`` with FastWAM-IDM two-stage semantics.
Adapted from OpenWAM's dual-system IDM driver (teacher-forcing mask +
merged noisy/cond video + frozen-video-KV inference) and its tri-system driver
(three-stream concat/split); the IDM logic is reimplemented here so the
``la_tri_system`` framework is self-contained.

Stream layout during IDM teacher-forcing training (one forward):

    [ noisy_video | cond_video | latent_action | action ]

The video region is the merged ``[noisy_video, cond_video]`` sequence (two
video timesteps concatenated along the sequence dim, mirroring
FastWAM-IDM's separated-timestep requirement). The latent-action stream is a
clean learnable-query stream (no noise, no timestep); the action stream is the
flow-matching denoising target.

Attention mask contract (rows = queries, cols = keys; True = attend):

                noisy_v  cond_v        la       action
    noisy_v     v2v      False         False    False
    cond_v      False    v2v           False    False
    la          False    la→cond(*)    True     False
    action      False    True          True     True

``(*)`` ``la→cond_v`` defaults to a full block: the latent action is extracted
from the *complete* clean/predicted video, which is the natural IDM setting
(the whole conditioning video is available to solve the action). It can be
switched to per-frame causal via ``la_video_attention_mode='per_frame_causal'``.

Load-bearing safety cells (must stay False): ``noisy_v→la`` and ``cond_v→la``
(the video denoising path must not depend on the latent-action stream, so the
video forward stays bit-identical to plain IDM), and ``la→noisy_v`` /
``la→action`` (la reads only clean video + itself). ``action→la`` is True — that
is the design intent: the action stream attends the latent-action summary at
every layer (layer-wise implicit coupling, no train/test gap).
"""

from __future__ import annotations

import copy
import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from openwam.model.architectures.utils.common import compute_video_tokens_per_frame

if TYPE_CHECKING:
    from openwam.model.action_backbone.base import ActionDiTBackbone
    from openwam.model.architectures.base import ActionState
    from openwam.model.latent_action_backbone import LatentActionExpert, LatentActionState
    from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone

logger = logging.getLogger(__name__)

_VALID_ATTENTION_MASK_MODES = ("bidirectional", "joint")
_VALID_LA_VIDEO_MODES = ("full", "per_frame_causal")


class LaTriSystemIDMMoTDriver:
    """Drive one MoT layer loop across video, latent-action, and action streams
    with IDM two-stage train/inference. Owns no parameters.
    """

    def __init__(
        self,
        vb: "VideoBackbone",
        ab: "ActionDiTBackbone",
        la: "LatentActionExpert",
        *,
        mot_checkpoint_mixed_attn: bool = True,
        attention_mask_mode: str = "joint",
        video_attention_mask_mode: Optional[str] = None,
        la_video_attention_mode: str = "full",
    ) -> None:
        for name, other in (("action", ab), ("latent_action", la)):
            if vb.num_layers != other.num_layers:
                raise ValueError(
                    f"LaTriSystemIDMMoTDriver: video num_layers ({vb.num_layers}) must equal "
                    f"{name} num_layers ({other.num_layers})."
                )
            if vb.num_heads != other.num_heads:
                raise ValueError(
                    f"LaTriSystemIDMMoTDriver: video num_heads ({vb.num_heads}) must equal "
                    f"{name} num_heads ({other.num_heads})."
                )
            if vb.head_dim != other.head_dim:
                raise ValueError(
                    f"LaTriSystemIDMMoTDriver: video head_dim ({vb.head_dim}) must equal "
                    f"{name} head_dim ({other.head_dim})."
                )
        if attention_mask_mode not in _VALID_ATTENTION_MASK_MODES:
            raise ValueError(
                f"LaTriSystemIDMMoTDriver: unknown attention_mask_mode '{attention_mask_mode}'. "
                f"Choose from: {_VALID_ATTENTION_MASK_MODES}."
            )
        if la_video_attention_mode not in _VALID_LA_VIDEO_MODES:
            raise ValueError(
                f"LaTriSystemIDMMoTDriver: unknown la_video_attention_mode '{la_video_attention_mode}'. "
                f"Choose from: {_VALID_LA_VIDEO_MODES}."
            )

        self.vb = vb
        self.ab = ab
        self.la = la
        self.num_layers = vb.num_layers
        self.num_heads = vb.num_heads
        self.head_dim = vb.head_dim
        self.mot_checkpoint_mixed_attn = bool(mot_checkpoint_mixed_attn)
        self.attention_mask_mode = attention_mask_mode
        self.la_video_attention_mode = la_video_attention_mode

        if video_attention_mask_mode is not None:
            try:
                vb.video_attention_mask_mode = video_attention_mask_mode  # type: ignore[misc]
            except AttributeError:
                logger.warning(
                    "video_attention_mask_mode='%s' supplied to LaTriSystemIDMMoTDriver but "
                    "%s does not expose a settable property; falling back to %s.",
                    video_attention_mask_mode,
                    type(vb).__name__,
                    vb.video_attention_mask_mode,
                )

    # ------------------------------------------------------------------
    # Mixed attention + tokens-per-frame
    # ------------------------------------------------------------------

    def _video_tokens_per_frame(self, vstate: "BlockLoopState") -> int:
        return compute_video_tokens_per_frame(vstate, "LaTriSystemIDMMoTDriver")

    def _mixed_attention(
        self, q_cat: Tensor, k_cat: Tensor, v_cat: Tensor, attn_mask: Optional[Tensor]
    ) -> Tensor:
        n = self.num_heads
        q = rearrange(q_cat, "b s (n d) -> b n s d", n=n)
        k = rearrange(k_cat, "b s (n d) -> b n s d", n=n)
        v = rearrange(v_cat, "b s (n d) -> b n s d", n=n)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return rearrange(out, "b n s d -> b s (n d)", n=n)

    def _la_to_video_block(
        self, s_la: int, s_cond_video: int, video_tokens_per_frame: int, device: torch.device
    ) -> Tensor:
        """Build the ``la → cond_video`` sub-block ``[s_la, s_cond_video]`` bool."""
        if self.la_video_attention_mode == "full":
            return torch.ones((s_la, s_cond_video), dtype=torch.bool, device=device)
        # per_frame_causal: la token of latent frame t (k_la tokens per frame)
        # may attend cond_video tokens of frames <= t.
        if s_cond_video % video_tokens_per_frame != 0:
            raise ValueError(
                f"cond_video seq_len ({s_cond_video}) must be divisible by "
                f"video_tokens_per_frame ({video_tokens_per_frame}) for per_frame_causal la mask."
            )
        num_frames = s_cond_video // video_tokens_per_frame
        k_la = self.la.k_la
        if s_la % k_la != 0:
            raise ValueError(f"la seq_len ({s_la}) must be divisible by k_la ({k_la}).")
        if s_la // k_la != num_frames:
            raise ValueError(
                f"la frame count ({s_la // k_la}) must equal cond_video frame count ({num_frames})."
            )
        la_frame = torch.arange(num_frames, device=device).repeat_interleave(k_la)  # (s_la,)
        vid_frame = torch.arange(num_frames, device=device).repeat_interleave(video_tokens_per_frame)
        return la_frame[:, None] >= vid_frame[None, :]

    def _build_teacher_forcing_mask(
        self,
        s_noisy_video: int,
        s_cond_video: int,
        s_la: int,
        s_action: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> Tensor:
        """Build the ``[T, T]`` bool mask over [noisy_v, cond_v, la, action]."""
        noisy_end = s_noisy_video
        cond_end = noisy_end + s_cond_video
        la_end = cond_end + s_la
        total = la_end + s_action
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        # video self-blocks (v2v); video never sees la or action.
        mask[:noisy_end, :noisy_end] = self.vb.build_video_to_video_mask(
            video_seq_len=s_noisy_video, video_tokens_per_frame=video_tokens_per_frame, device=device
        )
        mask[noisy_end:cond_end, noisy_end:cond_end] = self.vb.build_video_to_video_mask(
            video_seq_len=s_cond_video, video_tokens_per_frame=video_tokens_per_frame, device=device
        )

        # latent action: la→cond_video (full or per-frame causal), la→la = True.
        mask[cond_end:la_end, noisy_end:cond_end] = self._la_to_video_block(
            s_la, s_cond_video, video_tokens_per_frame, device
        )
        mask[cond_end:la_end, cond_end:la_end] = True

        # action: action→cond_video, action→la, action→action all True.
        mask[la_end:, noisy_end:cond_end] = True
        mask[la_end:, cond_end:la_end] = True
        mask[la_end:, la_end:] = True
        return mask

    def _build_action_stage_mask(
        self,
        s_cond_video: int,
        s_la: int,
        s_action: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        """Action-only stage-2 mask over keys ``[cond_video, la, action]``.

        Action attends everything available (cond_video, la, action). Returns
        ``None`` for ``bidirectional`` so SDPA picks the fused kernel.
        """
        if self.attention_mask_mode == "bidirectional":
            return None
        total_kv = s_cond_video + s_la + s_action
        return torch.ones((s_action, total_kv), dtype=torch.bool, device=device)

    def _build_prefill_mask(
        self,
        s_cond_video: int,
        s_la: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        """Clean-pass mask over [cond_video, la] (stage-2 prefill, no action).

                    cond_v        la
            cond_v  v2v           False
            la      la→cond(*)    True
        """
        if self.attention_mask_mode == "bidirectional":
            return None
        total = s_cond_video + s_la
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        mask[:s_cond_video, :s_cond_video] = self.vb.build_video_to_video_mask(
            video_seq_len=s_cond_video, video_tokens_per_frame=video_tokens_per_frame, device=device
        )
        mask[s_cond_video:, :s_cond_video] = self._la_to_video_block(
            s_la, s_cond_video, video_tokens_per_frame, device
        )
        mask[s_cond_video:, s_cond_video:] = True
        return mask

    @staticmethod
    def _get_action_tokens(astate: "ActionState") -> Tensor:
        payload = astate.payload
        if hasattr(payload, "x_action"):
            return payload.x_action
        raise RuntimeError("LaTriSystemIDMMoTDriver: action payload must expose `x_action`.")

    @staticmethod
    def _set_action_tokens(astate: "ActionState", value: Tensor) -> None:
        astate.payload.x_action = value

    def step(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        lastate: "LatentActionState",
        astate: "ActionState",
        attn_mask: Optional[Tensor],
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tuple["BlockLoopState", "LatentActionState", "ActionState"]:
        if use_gradient_checkpointing and self.ab.training:
            return self._step_checkpointed(
                layer_id, vstate, lastate, astate, attn_mask=attn_mask, offload=use_gradient_checkpointing_offload
            )
        return self._step_impl(layer_id, vstate, lastate, astate, attn_mask=attn_mask)

    def _step_impl(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        lastate: "LatentActionState",
        astate: "ActionState",
        attn_mask: Optional[Tensor],
        *,
        suppress_inner_attn_ckpt: bool = False,
    ) -> Tuple["BlockLoopState", "LatentActionState", "ActionState"]:
        """Per-layer body over [video, la, action].

        INVARIANT (consumed by ``_step_checkpointed``): only reassign the three
        layer-varying tensor fields ``vstate.hidden_states``, ``lastate.la_tokens``,
        ``astate.payload.x_action``. Do not mutate layer-invariant fields in
        place.
        """
        q_v, k_v, v_v, vpost = self.vb.pre_attn_at_layer(layer_id, vstate)
        q_l, k_l, v_l, lpost = self.la.pre_attn_at_layer(layer_id, lastate)
        q_a, k_a, v_a, apost = self.ab.pre_attn_at_layer(layer_id, astate)

        for name, t in (("la", q_l), ("action", q_a)):
            if t.dtype != q_v.dtype:
                raise RuntimeError(
                    f"LaTriSystemIDMMoTDriver: dtype mismatch at layer {layer_id} "
                    f"(video={q_v.dtype}, {name}={t.dtype})."
                )
            if t.device != q_v.device:
                raise RuntimeError(
                    f"LaTriSystemIDMMoTDriver: device mismatch at layer {layer_id} "
                    f"(video={q_v.device}, {name}={t.device})."
                )

        s_video, s_la, s_action = q_v.shape[1], q_l.shape[1], q_a.shape[1]
        q_cat = torch.cat([q_v, q_l, q_a], dim=1)
        k_cat = torch.cat([k_v, k_l, k_a], dim=1)
        v_cat = torch.cat([v_v, v_l, v_a], dim=1)

        if self.mot_checkpoint_mixed_attn and self.ab.training and not suppress_inner_attn_ckpt:
            mixed = torch.utils.checkpoint.checkpoint(
                self._mixed_attention, q_cat, k_cat, v_cat, attn_mask, use_reentrant=False
            )
        else:
            mixed = self._mixed_attention(q_cat, k_cat, v_cat, attn_mask)

        attn_v, attn_l, attn_a = mixed.split([s_video, s_la, s_action], dim=1)
        vstate = self.vb.post_attn_at_layer(layer_id, vstate, attn_v.contiguous(), vpost)
        lastate = self.la.post_attn_at_layer(layer_id, lastate, attn_l.contiguous(), lpost)
        astate = self.ab.post_attn_at_layer(layer_id, astate, attn_a.contiguous(), apost)
        return vstate, lastate, astate

    def _step_checkpointed(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        lastate: "LatentActionState",
        astate: "ActionState",
        *,
        attn_mask: Optional[Tensor],
        offload: bool,
    ) -> Tuple["BlockLoopState", "LatentActionState", "ActionState"]:
        outer_payload = astate.payload

        def _run(vx: Tensor, lx: Tensor, ax: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
            local_vstate = copy.copy(vstate)
            local_lastate = copy.copy(lastate)
            local_astate = copy.copy(astate)
            local_payload = copy.copy(outer_payload)
            local_astate.payload = local_payload
            local_vstate.hidden_states = vx
            local_lastate.la_tokens = lx
            local_payload.x_action = ax
            self._step_impl(
                layer_id, local_vstate, local_lastate, local_astate, attn_mask=attn_mask,
                suppress_inner_attn_ckpt=True,
            )
            return local_vstate.hidden_states, local_lastate.la_tokens, local_payload.x_action

        vx0, lx0, ax0 = vstate.hidden_states, lastate.la_tokens, outer_payload.x_action
        cm = torch.autograd.graph.save_on_cpu() if offload else nullcontext()
        with cm:
            new_vx, new_lx, new_ax = torch.utils.checkpoint.checkpoint(
                _run, vx0, lx0, ax0, use_reentrant=False
            )
        vstate.hidden_states = new_vx
        lastate.la_tokens = new_lx
        outer_payload.x_action = new_ax
        return vstate, lastate, astate

    # --- 2-stream (video + la, no action) per-layer step ---------------------
    # Mirror of step / _step_impl / _step_checkpointed for the pretrain
    # (lambda_action==0) path, so run_video_and_la_training_loop honors
    # use_gradient_checkpointing the SAME way the 3-stream idm loop does
    # (checkpoint the WHOLE layer, not just the inner mixed attention). Without
    # this the pretrain path retained every layer's block activations -> OOM.
    def _vl_step(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        lastate: "LatentActionState",
        attn_mask: Optional[Tensor],
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tuple["BlockLoopState", "LatentActionState"]:
        if use_gradient_checkpointing and self.la.training:
            return self._vl_step_checkpointed(
                layer_id, vstate, lastate, attn_mask=attn_mask, offload=use_gradient_checkpointing_offload
            )
        return self._vl_step_impl(layer_id, vstate, lastate, attn_mask=attn_mask)

    def _vl_step_impl(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        lastate: "LatentActionState",
        attn_mask: Optional[Tensor],
        *,
        suppress_inner_attn_ckpt: bool = False,
    ) -> Tuple["BlockLoopState", "LatentActionState"]:
        """Per-layer body over [video, la] (pretrain, no action stream).

        INVARIANT (consumed by ``_vl_step_checkpointed``): only reassign the
        layer-varying tensor fields ``vstate.hidden_states`` / ``lastate.la_tokens``.
        """
        q_v, k_v, v_v, vpost = self.vb.pre_attn_at_layer(layer_id, vstate)
        q_l, k_l, v_l, lpost = self.la.pre_attn_at_layer(layer_id, lastate)
        s_video, s_la = q_v.shape[1], q_l.shape[1]
        q_cat = torch.cat([q_v, q_l], dim=1)
        k_cat = torch.cat([k_v, k_l], dim=1)
        v_cat = torch.cat([v_v, v_l], dim=1)
        if self.mot_checkpoint_mixed_attn and self.la.training and not suppress_inner_attn_ckpt:
            mixed = torch.utils.checkpoint.checkpoint(
                self._mixed_attention, q_cat, k_cat, v_cat, attn_mask, use_reentrant=False
            )
        else:
            mixed = self._mixed_attention(q_cat, k_cat, v_cat, attn_mask)
        attn_v, attn_l = mixed.split([s_video, s_la], dim=1)
        vstate = self.vb.post_attn_at_layer(layer_id, vstate, attn_v.contiguous(), vpost)
        lastate = self.la.post_attn_at_layer(layer_id, lastate, attn_l.contiguous(), lpost)
        return vstate, lastate

    def _vl_step_checkpointed(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        lastate: "LatentActionState",
        *,
        attn_mask: Optional[Tensor],
        offload: bool,
    ) -> Tuple["BlockLoopState", "LatentActionState"]:
        def _run(vx: Tensor, lx: Tensor) -> Tuple[Tensor, Tensor]:
            local_vstate = copy.copy(vstate)
            local_lastate = copy.copy(lastate)
            local_vstate.hidden_states = vx
            local_lastate.la_tokens = lx
            self._vl_step_impl(
                layer_id, local_vstate, local_lastate, attn_mask=attn_mask,
                suppress_inner_attn_ckpt=True,
            )
            return local_vstate.hidden_states, local_lastate.la_tokens

        vx0, lx0 = vstate.hidden_states, lastate.la_tokens
        cm = torch.autograd.graph.save_on_cpu() if offload else nullcontext()
        with cm:
            new_vx, new_lx = torch.utils.checkpoint.checkpoint(_run, vx0, lx0, use_reentrant=False)
        vstate.hidden_states = new_vx
        lastate.la_tokens = new_lx
        return vstate, lastate

    def run_idm_training_loop(
        self,
        vstate_noisy: "BlockLoopState",
        vstate_cond: "BlockLoopState",
        lastate: "LatentActionState",
        astate: "ActionState",
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tuple["BlockLoopState", "BlockLoopState", "LatentActionState", "ActionState"]:
        """Merge [noisy_video | cond_video] and run the 3-stream joint loop.

        Returns ``(vstate_noisy, vstate_cond, lastate, astate)`` with the video
        halves split back out; ``vstate_noisy`` feeds the video loss,
        ``lastate``/``astate`` feed the la / action predictions.
        """
        if vstate_noisy.time_mod.ndim != 4 or vstate_cond.time_mod.ndim != 4:
            raise ValueError(
                "la_tri_system IDM requires token-wise (4D) video t_mod for the noisy and cond branches; "
                "ensure force_per_token_t_mod is set on vb.prepare()."
            )
        if (vstate_noisy.grid_height, vstate_noisy.grid_width) != (vstate_cond.grid_height, vstate_cond.grid_width):
            raise ValueError(
                "la_tri_system IDM requires noisy and cond video branches to share spatial token layout, "
                f"got noisy h/w={(vstate_noisy.grid_height, vstate_noisy.grid_width)} and cond h/w={(vstate_cond.grid_height, vstate_cond.grid_width)}."
            )
        s_noisy = vstate_noisy.hidden_states.shape[1]
        s_cond = vstate_cond.hidden_states.shape[1]
        s_la = lastate.la_tokens.shape[1]
        s_action = self._get_action_tokens(astate).shape[1]

        merged_vstate = copy.copy(vstate_noisy)
        merged_vstate.hidden_states = torch.cat([vstate_noisy.hidden_states, vstate_cond.hidden_states], dim=1)
        merged_vstate.rope_freqs = torch.cat([vstate_noisy.rope_freqs, vstate_cond.rope_freqs], dim=0)
        merged_vstate.time_mod = torch.cat([vstate_noisy.time_mod, vstate_cond.time_mod], dim=1)
        if vstate_noisy.vace_hints is not None or vstate_cond.vace_hints is not None:
            if vstate_noisy.vace_hints is None or vstate_cond.vace_hints is None:
                raise ValueError("la_tri_system IDM requires both video branches to have VACE hints or neither.")
            if len(vstate_noisy.vace_hints) != len(vstate_cond.vace_hints):
                raise ValueError("la_tri_system IDM VACE hint count mismatch between noisy and cond branches.")
            merged_vstate.vace_hints = [
                torch.cat([hn, hc], dim=1) for hn, hc in zip(vstate_noisy.vace_hints, vstate_cond.vace_hints)
            ]

        video_tokens_per_frame = self._video_tokens_per_frame(vstate_noisy)
        attn_mask = self._build_teacher_forcing_mask(
            s_noisy_video=s_noisy,
            s_cond_video=s_cond,
            s_la=s_la,
            s_action=s_action,
            video_tokens_per_frame=video_tokens_per_frame,
            device=merged_vstate.hidden_states.device,
        )

        for layer_id in range(self.num_layers):
            merged_vstate, lastate, astate = self.step(
                layer_id,
                merged_vstate,
                lastate,
                astate,
                attn_mask=attn_mask,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )

        vstate_noisy.hidden_states = merged_vstate.hidden_states[:, :s_noisy]
        vstate_cond.hidden_states = merged_vstate.hidden_states[:, s_noisy:]
        vstate_noisy.time_mod = merged_vstate.time_mod[:, :s_noisy]
        vstate_cond.time_mod = merged_vstate.time_mod[:, s_noisy:]
        return vstate_noisy, vstate_cond, lastate, astate

    def _build_video_la_training_mask(
        self,
        s_noisy_video: int,
        s_cond_video: int,
        s_la: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> Tensor:
        """Teacher-forcing mask over [noisy_v, cond_v, la] (no action stream).

        Identical cells to ``_build_teacher_forcing_mask`` with ``s_action=0``:
        video self-blocks (v2v), ``cond_v`` never sees la, ``la → cond_v``
        (full / per-frame-causal), ``la → la``. Used by the pretrain
        (lambda_action==0) path so the la stream still runs + receives gradient.
        """
        return self._build_teacher_forcing_mask(
            s_noisy_video=s_noisy_video,
            s_cond_video=s_cond_video,
            s_la=s_la,
            s_action=0,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

    def run_video_and_la_training_loop(
        self,
        vstate_noisy: "BlockLoopState",
        vstate_cond: "BlockLoopState",
        lastate: "LatentActionState",
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tuple["BlockLoopState", "BlockLoopState", "LatentActionState"]:
        """Pretrain (lambda_action==0) training loop: [noisy_video | cond_video | la],
        NO action stream. Gradient-enabled (unlike ``prefill_video_and_la``).

        Merges noisy+cond video exactly like ``run_idm_training_loop`` so the
        noisy branch still gets video-loss gradients, runs a 2-stream (video+la)
        mixed attention per layer, and returns ``(vstate_noisy, vstate_cond,
        lastate)`` with the video halves split back. ``lastate`` yields
        ``la_pred`` via ``la.extract_prediction`` — this is what makes pretrain
        actually supervise the latent-action stream.
        """
        if vstate_noisy.time_mod.ndim != 4 or vstate_cond.time_mod.ndim != 4:
            raise ValueError(
                "la_tri_system pretrain requires token-wise (4D) video t_mod for the noisy and cond "
                "branches; ensure force_per_token_t_mod is set on vb.prepare()."
            )
        if (vstate_noisy.grid_height, vstate_noisy.grid_width) != (vstate_cond.grid_height, vstate_cond.grid_width):
            raise ValueError(
                "la_tri_system pretrain requires noisy and cond video branches to share spatial token "
                f"layout, got noisy h/w={(vstate_noisy.grid_height, vstate_noisy.grid_width)} and cond "
                f"h/w={(vstate_cond.grid_height, vstate_cond.grid_width)}."
            )
        s_noisy = vstate_noisy.hidden_states.shape[1]
        s_cond = vstate_cond.hidden_states.shape[1]
        s_la = lastate.la_tokens.shape[1]

        merged_vstate = copy.copy(vstate_noisy)
        merged_vstate.hidden_states = torch.cat([vstate_noisy.hidden_states, vstate_cond.hidden_states], dim=1)
        merged_vstate.rope_freqs = torch.cat([vstate_noisy.rope_freqs, vstate_cond.rope_freqs], dim=0)
        merged_vstate.time_mod = torch.cat([vstate_noisy.time_mod, vstate_cond.time_mod], dim=1)
        if vstate_noisy.vace_hints is not None or vstate_cond.vace_hints is not None:
            if vstate_noisy.vace_hints is None or vstate_cond.vace_hints is None:
                raise ValueError("la_tri_system pretrain requires both video branches to have VACE hints or neither.")
            if len(vstate_noisy.vace_hints) != len(vstate_cond.vace_hints):
                raise ValueError("la_tri_system pretrain VACE hint count mismatch between noisy and cond branches.")
            merged_vstate.vace_hints = [
                torch.cat([hn, hc], dim=1) for hn, hc in zip(vstate_noisy.vace_hints, vstate_cond.vace_hints)
            ]

        video_tokens_per_frame = self._video_tokens_per_frame(vstate_noisy)
        attn_mask = self._build_video_la_training_mask(
            s_noisy_video=s_noisy,
            s_cond_video=s_cond,
            s_la=s_la,
            video_tokens_per_frame=video_tokens_per_frame,
            device=merged_vstate.hidden_states.device,
        )

        for layer_id in range(self.num_layers):
            merged_vstate, lastate = self._vl_step(
                layer_id,
                merged_vstate,
                lastate,
                attn_mask,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )

        vstate_noisy.hidden_states = merged_vstate.hidden_states[:, :s_noisy]
        vstate_cond.hidden_states = merged_vstate.hidden_states[:, s_noisy:]
        vstate_noisy.time_mod = merged_vstate.time_mod[:, :s_noisy]
        vstate_cond.time_mod = merged_vstate.time_mod[:, s_noisy:]
        return vstate_noisy, vstate_cond, lastate

    @torch.no_grad()
    def prefill_video_and_la(
        self,
        vstate: "BlockLoopState",
        lastate: "LatentActionState",
    ) -> Tuple[list, "LatentActionState", "BlockLoopState"]:
        """Stage-2 clean pass: run [cond_video, la] jointly once.

        Produces the per-layer cond-video + la K/V caches (so the action
        denoising loop can attend frozen video and la), advances ``lastate``
        through all layers, and returns ``(kv_cache, lastate, vstate)``. The
        refined ``lastate`` yields ``la_pred`` via ``la.extract_prediction``.
        """
        s_cond_video = int(vstate.hidden_states.shape[1])
        s_la = int(lastate.la_tokens.shape[1])
        video_tokens_per_frame = self._video_tokens_per_frame(vstate)
        attn_mask = self._build_prefill_mask(s_cond_video, s_la, video_tokens_per_frame, vstate.hidden_states.device)

        kv_cache: list[dict[str, Tensor]] = []
        for layer_id in range(self.num_layers):
            q_v, k_v, v_v, vpost = self.vb.pre_attn_at_layer(layer_id, vstate)
            q_l, k_l, v_l, lpost = self.la.pre_attn_at_layer(layer_id, lastate)
            q_cat = torch.cat([q_v, q_l], dim=1)
            k_cat = torch.cat([k_v, k_l], dim=1)
            v_cat = torch.cat([v_v, v_l], dim=1)
            mixed = self._mixed_attention(q_cat, k_cat, v_cat, attn_mask)
            attn_v, attn_l = mixed.split([s_cond_video, s_la], dim=1)
            vstate = self.vb.post_attn_at_layer(layer_id, vstate, attn_v.contiguous(), vpost)
            lastate = self.la.post_attn_at_layer(layer_id, lastate, attn_l.contiguous(), lpost)
            # Cache the cond-video AND la keys/values for the action stage.
            kv_cache.append({"k": torch.cat([k_v, k_l], dim=1), "v": torch.cat([v_v, v_l], dim=1)})
        return kv_cache, lastate, vstate

    def run_action_with_caches(
        self,
        astate: "ActionState",
        *,
        kv_cache: list,
        s_cond_video: int,
        s_la: int,
    ) -> "ActionState":
        """Stage-2 action denoising against frozen [cond_video, la] K/V."""
        if len(kv_cache) != self.num_layers:
            raise ValueError(f"kv_cache must contain {self.num_layers} layers, got {len(kv_cache)}.")
        s_action = int(self._get_action_tokens(astate).shape[1])
        attn_mask = self._build_action_stage_mask(s_cond_video, s_la, s_action, self._get_action_tokens(astate).device)
        expected_kv = s_cond_video + s_la
        for layer_id in range(self.num_layers):
            q_a, k_a, v_a, apost = self.ab.pre_attn_at_layer(layer_id, astate)
            cache = kv_cache[layer_id]
            if cache["k"].shape[1] != expected_kv:
                raise ValueError(
                    f"kv_cache[{layer_id}] seq length {cache['k'].shape[1]} != expected "
                    f"cond_video+la = {expected_kv}."
                )
            k_cat = torch.cat([cache["k"], k_a], dim=1)
            v_cat = torch.cat([cache["v"], v_a], dim=1)
            mixed = self._mixed_attention(q_a, k_cat, v_cat, attn_mask)
            astate = self.ab.post_attn_at_layer(layer_id, astate, mixed.contiguous(), apost)
        return astate


__all__ = ["LaTriSystemIDMMoTDriver"]
