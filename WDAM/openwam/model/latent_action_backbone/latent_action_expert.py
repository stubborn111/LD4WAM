"""Latent Action Expert for the ``la_tri_system`` framework.

A third MoT stream — alongside the Wan video DiT and the shared ActionDiT —
that predicts a compact *latent action* (frame-to-frame dynamics code) from
the clean/conditioning video. It is supervised by an offline-precomputed LAM
(latent action model) target with a plain masked MSE, NOT flow matching.

Design (lingbot-va scheme "S1", adapted to OpenWAM):

- **Learnable query, not denoising.** The expert owns a per-sub-position seed
  ``query_base`` of shape ``(k_la, la_hidden_dim)``; ``k_la`` latent-action
  sub-tokens are emitted per *video latent frame*. ``prepare_state`` broadcasts
  the seed to ``(B, Fv * k_la, la_hidden_dim)`` and the joint MoT attention
  refines it layer by layer. ``extract_prediction`` then maps the refined
  tokens to ``la_dim`` in one shot. Zero extra inference steps, no train/test
  gap (it behaves identically in training and the IDM clean pass).

- **No timestep / no AdaLN.** Like :class:`UnderstandingExpert`, the block is a
  plain norm → (shared-space) self-attention → norm → FFN residual stack. The
  latent-action stream is unconditional, so there is no diffusion-timestep
  modulation.

- **Heterogeneous width via the shared attention space.** Q/K/V project from
  ``la_hidden_dim`` into the video backbone's ``num_heads * head_dim`` space
  (reusing :class:`ActionSelfAttention`), so the latent-action stream can keep
  its own residual width yet still concatenate into the single joint attention.
  RoPE over the latent-action positions keeps it aligned with the per-frame
  causal mask the driver builds.

The MoT-driver split interface (``num_layers`` / ``num_heads`` / ``head_dim``
properties + ``prepare_state`` / ``pre_attn_at_layer`` /
``post_attn_at_layer`` + ``*_for_compile`` variants + ``extract_prediction``)
mirrors :class:`ActionDiT` exactly so the driver's per-layer Q/K/V
concatenation works without special-casing this stream.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from openwam.model.action_backbone.components import (
    precompute_freqs_cis_1d,
    rope_apply_1d,
)

# Latent-action sub-tokens per video latent frame. Fixed at 4 to match the
# lingbot-va LAM target layout (``la_gt`` is produced with k_la=4 sub-tokens
# per latent frame). The offline la_gt pipeline and this value must agree.
LA_K_PER_FRAME = 4


@dataclass
class LatentActionExpertConfig:
    """Domain hyperparameters for the latent action expert.

    ``num_heads`` / ``head_dim`` are NOT stored here — they are injected at
    construction from the video backbone geometry (mirrors
    :class:`UnderstandingExpertConfig`, which takes ``wan_dim`` / ``wan_num_heads``
    as constructor args rather than config fields).
    """

    la_dim: int = 512
    la_hidden_dim: int = 512
    la_ffn_dim: int = 2048
    num_layers: int = 30
    k_la: int = LA_K_PER_FRAME
    max_la_len: int = 1024
    eps: float = 1e-6


@dataclass
class LatentActionState:
    """Mutable container threaded through the joint self-attention loop.

    Populated by :meth:`LatentActionExpert.prepare_state` and consumed by
    :meth:`LatentActionExpert.pre_attn_at_layer` /
    :meth:`LatentActionExpert.post_attn_at_layer` (called per layer from the
    MoT driver) and finally by :meth:`LatentActionExpert.extract_prediction`.
    """

    la_tokens: torch.Tensor  # (B, Fv * k_la, la_hidden_dim)
    freqs: torch.Tensor  # 1D RoPE frequencies for the latent-action positions
    num_latent_frames: int = 0  # Fv (for shape assertions downstream)


class LatentActionExpertBlock(nn.Module):
    """One latent-action block driven by the MoT joint self-attention loop.

    Layout (no AdaLN, mirroring :class:`UnderstandingExpertBlock`):
    ``norm1 → self-attn (shared attention space, RoPE) → residual``,
    then ``norm2 → FFN → residual``. The block is split around self-attention
    so the driver can pull Q/K/V, run one mixed attention across all streams,
    and feed the latent-action slice back through the suffix.

    ``self_attn`` is the same :class:`ActionSelfAttention` used by the action
    expert: Q/K/V project ``la_hidden_dim → num_heads*head_dim`` and ``o``
    projects back, so a heterogeneous residual width still shares the joint
    attention space. ``o`` is left at standard linear init (NOT zero-init): the
    latent-action stream must stay gradient-connected through the joint
    attention from step 0. Early influence is kept small by the ``query_base``
    ~ randn/sqrt(dim) scaling, and since the video streams never attend la (see
    the driver mask) only the action stream sees it early. Stability of the la
    *prediction* is instead handled by the zero-init ``proj_out`` in
    :class:`LatentActionExpert`. See the NOTE in ``__init__`` below.
    """

    def __init__(
        self,
        la_hidden_dim: int,
        num_heads: int,
        head_dim: int,
        la_ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        # Local import avoids a circular import at module load (separate_action_dit
        # imports from this package's sibling components only, but keeping the
        # heavyweight ActionSelfAttention import lazy mirrors how ActionDiT
        # itself defers TimestepEmbedding).
        from openwam.model.action_backbone.separate_action_dit import ActionSelfAttention

        self.la_hidden_dim = la_hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.self_attn = ActionSelfAttention(la_hidden_dim, num_heads, head_dim, eps)
        self.self_attn_norm = nn.LayerNorm(la_hidden_dim, eps=eps, elementwise_affine=False)
        self.ffn_norm = nn.LayerNorm(la_hidden_dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(la_hidden_dim, la_ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(la_ffn_dim, la_hidden_dim),
        )
        # NOTE: ``self_attn.o`` is intentionally NOT zero-initialized. The
        # latent-action stream must stay gradient-connected through the joint
        # attention from step 0 (mirroring UnderstandingExpert, which relies on
        # small-variance Q/K/V init rather than a zero output gate). Early
        # influence on the video/action streams is kept small by the
        # ``query_base`` ~ randn/sqrt(dim) scaling and standard linear init;
        # stability of the la *prediction* is handled by the zero-init
        # ``proj_out`` in :class:`LatentActionExpert` instead.

    def forward(self, x: torch.Tensor, freqs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reference forward used in equivalence tests; not invoked by the MoT driver."""
        x = x + self.self_attn(self.self_attn_norm(x), freqs=freqs)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LatentActionExpert(nn.Module):
    """Latent action expert: a learnable-query third MoT stream.

    Constructed with the video backbone geometry injected (``wan_dim`` /
    ``wan_num_heads``) so its Q/K/V land in the shared attention space, exactly
    like :class:`UnderstandingExpert`.
    """

    def __init__(self, cfg: LatentActionExpertConfig, wan_dim: int, wan_num_heads: int):
        super().__init__()
        self.cfg = cfg
        self.la_dim = int(cfg.la_dim)
        self.la_hidden_dim = int(cfg.la_hidden_dim)
        self.k_la = int(cfg.k_la)
        self._num_heads = int(wan_num_heads)
        self._head_dim = int(wan_dim // wan_num_heads)
        if self._head_dim * wan_num_heads != wan_dim:
            raise ValueError(f"wan_dim ({wan_dim}) must be divisible by wan_num_heads ({wan_num_heads}).")
        self._num_layers = int(cfg.num_layers)
        self.max_la_len = int(cfg.max_la_len)

        # Per-sub-position learnable seeds: query_base[j] seeds the j-th
        # latent-action sub-token of every video latent frame. Broadcast to
        # (B, Fv, k_la, h) → (B, Fv*k_la, h) in ``get_query``.
        self.query_base = nn.Parameter(
            torch.randn(self.k_la, self.la_hidden_dim) / self.la_hidden_dim**0.5
        )

        self.blocks = nn.ModuleList(
            [
                LatentActionExpertBlock(
                    self.la_hidden_dim,
                    self._num_heads,
                    self._head_dim,
                    int(cfg.la_ffn_dim),
                    eps=float(cfg.eps),
                )
                for _ in range(self._num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(self.la_hidden_dim, eps=float(cfg.eps), elementwise_affine=False)
        self.proj_out = nn.Linear(self.la_hidden_dim, self.la_dim)
        # Zero-init the prediction head so early la_pred ≈ 0 (stable start; the
        # masked-MSE la loss simply pulls it toward the LAM target as it warms).
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        # RoPE freqs over latent-action positions, sized by the shared per-head
        # attention dim. Plain attribute (not a buffer) so model.to(bf16) does
        # not cast the complex tensor to real — mirrors ActionDiT.freqs.
        self.freqs = precompute_freqs_cis_1d(self._head_dim, self.max_la_len)

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    def get_query(self, B: int, total_la: int, *, device=None, dtype=None) -> torch.Tensor:
        """Broadcast per-sub-position seeds to ``(B, total_la, la_hidden_dim)``.

        The ``k_la`` learnable seeds are cycled to fill ``total_la`` query tokens
        (``seed[i % k_la]``), so ``total_la`` need NOT be a multiple of ``k_la``.
        When ``total_la == Fv*k_la`` (the per-latent-frame layout) this is exactly
        ``Fv`` repeats of the ``k_la`` seeds — identical to the original scheme.
        When the online-LAM target is one action per adjacent RGB-frame pair,
        ``total_la = T_video - 1`` and the seeds simply cycle; RoPE over the
        latent-action positions disambiguates the tokens.
        """
        q = self.query_base.to(device=device, dtype=dtype)  # (k_la, h)
        idx = torch.arange(total_la, device=q.device) % self.k_la
        seeds = q[idx]  # (total_la, h)
        return seeds[None, :, :].expand(B, total_la, self.la_hidden_dim).contiguous()

    def _get_rope_freqs(self, seq_len: int) -> torch.Tensor:
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Latent-action sequence length {seq_len} exceeds precomputed RoPE cache "
                f"length {self.freqs.shape[0]}; increase ``max_la_len``."
            )
        return self.freqs[:seq_len]

    def prepare_state(
        self,
        B: int,
        num_latent_frames: Optional[int] = None,
        *,
        num_la_tokens: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> LatentActionState:
        """Build the per-forward latent-action state.

        Provide exactly one of:
          - ``num_la_tokens``: emit this many latent-action tokens directly
            (online-LAM layout: one per adjacent RGB-frame pair, ``T_video-1``).
          - ``num_latent_frames`` (``Fv``): emit ``Fv * k_la`` tokens
            (the original per-latent-frame layout).
        """
        if num_la_tokens is not None:
            total_la = int(num_la_tokens)
        elif num_latent_frames is not None:
            total_la = int(num_latent_frames) * self.k_la
        else:
            raise ValueError("prepare_state requires either num_la_tokens or num_latent_frames.")
        x = self.get_query(B, total_la, device=device, dtype=dtype)
        freqs = self._get_rope_freqs(total_la).to(device=device)
        return LatentActionState(la_tokens=x, freqs=freqs, num_latent_frames=int(num_latent_frames or 0))

    def pre_attn_at_layer(
        self, layer_id: int, state: LatentActionState
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """First half of a block: norm + Q/K/V + RoPE.

        Returns Q/K/V shaped ``(B, Fv*k_la, num_heads*head_dim)`` — the same
        layout the video and action streams produce — ready to concatenate into
        the joint mixed attention. ``post_state`` carries the pre-attention
        residual for the suffix.
        """
        q, k, v, post_tuple = self.pre_attn_at_layer_for_compile(layer_id, state)
        (residual_x,) = post_tuple
        return q, k, v, {"residual_x": residual_x}

    def pre_attn_at_layer_for_compile(
        self, layer_id: int, state: LatentActionState
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """Compile-friendly pre-attention half using a tensor-tuple post-state."""
        block: LatentActionExpertBlock = self.blocks[layer_id]
        residual_x = state.la_tokens
        attn_input = block.self_attn_norm(residual_x)

        sa = block.self_attn
        q = sa.norm_q(sa.q(attn_input))
        k = sa.norm_k(sa.k(attn_input))
        v = sa.v(attn_input)

        q = rearrange(q, "b s (n d) -> b n s d", n=self._num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self._num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self._num_heads)
        q = rope_apply_1d(q, state.freqs)
        k = rope_apply_1d(k, state.freqs)
        q_out = rearrange(q, "b n s d -> b s (n d)", n=self._num_heads)
        k_out = rearrange(k, "b n s d -> b s (n d)", n=self._num_heads)
        v_out = rearrange(v, "b n s d -> b s (n d)", n=self._num_heads)
        return q_out, k_out, v_out, (residual_x,)

    def post_attn_at_layer(
        self,
        layer_id: int,
        state: LatentActionState,
        attn_out: torch.Tensor,
        post_state: dict,
    ) -> LatentActionState:
        """Second half: gate(residual, o(attn_out)) → FFN.

        ``attn_out`` is the unprojected latent-action slice of the joint mixed
        attention; ``self_attn.o`` (zero-init) is applied here.
        """
        if isinstance(post_state, dict):
            post_state = (post_state["residual_x"],)
        return self.post_attn_at_layer_for_compile(layer_id, state, attn_out, post_state)

    def post_attn_at_layer_for_compile(
        self,
        layer_id: int,
        state: LatentActionState,
        attn_out: torch.Tensor,
        post_state: tuple,
    ) -> LatentActionState:
        block: LatentActionExpertBlock = self.blocks[layer_id]
        (residual_x,) = post_state
        x = residual_x + block.self_attn.o(attn_out)
        x = x + block.ffn(block.ffn_norm(x))
        state.la_tokens = x
        return state

    def extract_prediction(self, state: LatentActionState) -> torch.Tensor:
        """Map refined latent-action tokens to ``(B, Fv*k_la, la_dim)``."""
        return self.proj_out(self.norm_out(state.la_tokens))


__all__ = [
    "LA_K_PER_FRAME",
    "LatentActionExpert",
    "LatentActionExpertBlock",
    "LatentActionExpertConfig",
    "LatentActionState",
]
