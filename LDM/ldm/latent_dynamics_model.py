import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lpips
import torch
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from ldm.attention import ContinuousPositionBias, FeedForward, STTransformer, Transformer
from ldm.softvq import SoftVQ
from ldm.utils import PatchEmbed, exists, get_vq_encoder, leaky_relu, pair
from ldm.dinov3 import DINOv2Encoder, DINOv3Encoder, VisionTransformerEncoder
from torch import nn
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

logger = logging.getLogger(__name__)


class ActionAlignHead(nn.Module):
    

    def __init__(
        self,
        quant_dim: int,
        action_dim: int,
        code_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        latent_dim = quant_dim * code_seq_len
        hidden_dim = max(256, latent_dim)
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        z = rearrange(z_q, "b t hq wq d -> b t (hq wq d)")
        return self.net(z)  

class LDM(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        quant_dim: int,
        codebook_size: int,
        image_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        enc_depth: int,
        dec_depth: int,
        dim_head: int = 64,
        heads: int = 8,
        channels: int = 3,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        code_seq_len: int = 1,
        encode_deltas: bool = True,
        discarding_threshold: float = 0.1,
        max_codebook_update_step: int = 10000,
        spatial_enc_type: str = "dino_base",  
        dinov3_model_path: str = "./pretrained/dinov3-vitl16-local",
        freeze_spatial_encoder: bool = True,
        semantic_loss_weight: float = 1.0,
        use_lpips_loss: bool = False,
        lpips_loss_weight: float = 0.0,
        use_flow_loss: bool = False,
        flow_loss_weight: float = 0.0,
        flow_loss_kickin_step: int = 0,
        flow_loss_warmup_steps: int = 10_000,
        feature_norm_loss_weight: float = 0.0,
        codebook_usage_loss_weight: float = 0.0,
        token_entropy_loss_weight: float = 0.0,
        action_dim: int = 0,           
        align_kickin_step: int = 20_000,
        align_loss_weight: float = 0.1,
        align_action_translation_scale: float = 0.02,
        align_action_rotation_scale: float = 0.20,
        align_action_gripper_scale: float = 0.02,
    ) -> None:
        super().__init__()

        self.code_seq_len = code_seq_len
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size
        self.patch_grid = (self.image_size[0] // patch_height, self.image_size[1] // patch_width)
        self.channels = channels

        action_h = int(math.sqrt(self.code_seq_len))
        action_w = self.code_seq_len // action_h
        self.action_size = (action_h, action_w)

        self.enc_spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads, num_dims=2)
        self.dec_spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads, num_dims=2)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        enc_st_transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal=False,
            peg=True,
            peg_causal=False,
        )

        dec_st_transformer_kwargs = dict(
            dim=dim,
            dim_cond=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal=True,
            peg=True,
            peg_causal=True,
            enable_conditioning=True,
        )

        spatial_enc_type_parts = spatial_enc_type.split("_")
        vit_size = "_".join(spatial_enc_type_parts[1:])
        self.spatial_enc_type = spatial_enc_type_parts[0]

        if self.spatial_enc_type == "dino":
            self.enc_spatial_transformer = DINOv2Encoder(
                image_size=image_size,
                patch_size=patch_size,
                vit_size=vit_size,
            )
        elif self.spatial_enc_type == "dinov3":
            self.enc_spatial_transformer = DINOv3Encoder(
                image_size=self.image_size[0],
                patch_size=self.patch_size[0],
                model_path=dinov3_model_path,
                freeze=freeze_spatial_encoder,
            )
        elif self.spatial_enc_type == "vit":
            self.enc_spatial_transformer = VisionTransformerEncoder(
                image_size=image_size,
                patch_size=patch_size,
                vit_size=vit_size,
            )
        else:
            raise ValueError("Invalid spatial_enc_type. Choose 'dino', 'dinov3', or 'vit'.")

        self.enc_st_transformer = STTransformer(depth=enc_depth, **enc_st_transformer_kwargs)

        self.encode_deltas = encode_deltas
        self.vq_project_in = nn.Linear(dim, quant_dim)
        self.vq_encoder = get_vq_encoder(code_seq_len, quant_dim)
        self.vq = SoftVQ(
            num_embeddings=codebook_size,
            embedding_dim=quant_dim,
            discarding_threshold=discarding_threshold,
        )
        self.max_codebook_update_step = max_codebook_update_step

        self.vq_to_cond = nn.Sequential(
            Rearrange("b t hq wq d -> b t (hq wq d)", hq=action_h, wq=action_w, d=quant_dim),
            nn.Linear(quant_dim * code_seq_len, dim),
            FeedForward(dim, mult=4.0, dropout=ff_dropout),
        )
        self.dec_transformer = STTransformer(depth=dec_depth, **dec_st_transformer_kwargs)
        self.to_dino_features = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

        self.semantic_loss_weight = semantic_loss_weight
        self.feature_norm_loss_weight = feature_norm_loss_weight
        self.codebook_usage_loss_weight = codebook_usage_loss_weight
        self.token_entropy_loss_weight = token_entropy_loss_weight
        self.use_lpips_loss = use_lpips_loss
        self.lpips_loss_weight = lpips_loss_weight
        if self.use_lpips_loss:
            self.lpips = lpips.LPIPS(net="vgg").eval().requires_grad_(False)

        self.use_flow_loss = use_flow_loss
        self.flow_loss_weight = flow_loss_weight
        self.flow_loss_kickin_step = flow_loss_kickin_step
        self.flow_loss_warmup_steps = flow_loss_warmup_steps
        if self.use_flow_loss:
            self.flow_model = (
                raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False).eval().requires_grad_(False)
            )

        self.action_dim = action_dim
        self.align_kickin_step = align_kickin_step
        self.align_loss_weight = align_loss_weight
        self.align_action_translation_scale = float(align_action_translation_scale)
        self.align_action_rotation_scale = float(align_action_rotation_scale)
        self.align_action_gripper_scale = float(align_action_gripper_scale)
        if action_dim > 0:
            self.action_align_head = ActionAlignHead(quant_dim, action_dim, code_seq_len=code_seq_len)

    def forward(
        self,
        videos: torch.Tensor,           
        mask: torch.BoolTensor = None,  
        step: int = 0,
        actions: Optional[torch.Tensor] = None,          
        action_mask: Optional[torch.BoolTensor] = None,  
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B, T, C, H, W = videos.shape
        device = videos.device
        mask = mask if mask is not None else torch.ones((B, T), device=device, dtype=torch.bool)
        mask_gt = mask[:, :-1]  
        mask_recon = mask[:, 1:]  
        mask_loss = torch.logical_and(mask_gt, mask_recon)  

        frame_tokens, tokens = self.encode(videos, mask=mask)  

        if self.encode_deltas:
            vq_inputs = tokens[:, 1:] - tokens[:, :-1]
        else:
            vq_inputs = tokens[:, 1:]

        quantized_actions, perplexity, num_unique_indices, vq_stats = self.quantize(vq_inputs)

        if (
            step > 0
            and step < self.max_codebook_update_step
            and (
                (100 <= step <= 1000 and step % 100 == 0)
                or (1000 < step <= 10_000 and step % 1000 == 0)
                or (10_000 < step <= 40_000 and step % 2000 == 0)
                or (40_000 < step <= 90_000 and step % 3000 == 0)
                or (step > 90_000 and step % 5_000 == 0)
            )
            and self.training
        ):
            self.vq.replace_unused_codebooks()

        pred_features = self.decode(frame_tokens[:, :-1], quantized_actions, mask_recon=mask_gt)
        target_features = frame_tokens[:, 1:]

        semantic_loss, semantic_cosine_loss, feature_norm_loss = self.compute_semantic_reconstruction_loss(
            pred_features, target_features, mask_loss
        )
        usage_kl_loss = vq_stats["usage_kl_loss"]
        token_entropy_loss = vq_stats["token_entropy_loss"]

        lpips_loss = 0.0
        flow_loss = 0.0
        flow_loss_weight = 0.0

        align_loss = 0.0
        n_align_samples = 0
        align_logs: Dict[str, float] = {}
        if (
            self.action_dim > 0
            and step >= self.align_kickin_step
            and actions is not None
            and action_mask is not None
            and action_mask.any()
        ):
            align_loss, n_align_samples, align_logs = self.compute_align_loss(
                quantized_actions, actions, action_mask, mask_loss
            )

        semantic_loss_contrib = self.semantic_loss_weight * semantic_loss
        align_loss_contrib = self.align_loss_weight * align_loss
        usage_kl_loss_contrib = self.codebook_usage_loss_weight * usage_kl_loss
        token_entropy_loss_contrib = self.token_entropy_loss_weight * token_entropy_loss
        feature_norm_loss_contrib = self.semantic_loss_weight * self.feature_norm_loss_weight * feature_norm_loss
        loss = semantic_loss_contrib + align_loss_contrib + usage_kl_loss_contrib + token_entropy_loss_contrib

        log_dict = {
            "loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
            "semantic_loss": semantic_loss.item() if isinstance(semantic_loss, torch.Tensor) else semantic_loss,
            "semantic_cosine_loss": semantic_cosine_loss.item() if isinstance(semantic_cosine_loss, torch.Tensor) else semantic_cosine_loss,
            "feature_norm_loss": feature_norm_loss.item() if isinstance(feature_norm_loss, torch.Tensor) else feature_norm_loss,
            "usage_kl_loss": usage_kl_loss.item() if isinstance(usage_kl_loss, torch.Tensor) else usage_kl_loss,
            "token_entropy_loss": token_entropy_loss.item() if isinstance(token_entropy_loss, torch.Tensor) else token_entropy_loss,
            "semantic_loss_contrib": semantic_loss_contrib.item() if isinstance(semantic_loss_contrib, torch.Tensor) else semantic_loss_contrib,
            "feature_norm_loss_contrib": feature_norm_loss_contrib.item() if isinstance(feature_norm_loss_contrib, torch.Tensor) else feature_norm_loss_contrib,
            "align_loss_contrib": align_loss_contrib.item() if isinstance(align_loss_contrib, torch.Tensor) else align_loss_contrib,
            "usage_kl_loss_contrib": usage_kl_loss_contrib.item() if isinstance(usage_kl_loss_contrib, torch.Tensor) else usage_kl_loss_contrib,
            "token_entropy_loss_contrib": token_entropy_loss_contrib.item() if isinstance(token_entropy_loss_contrib, torch.Tensor) else token_entropy_loss_contrib,
            "lpips_loss": lpips_loss.item() if isinstance(lpips_loss, torch.Tensor) else lpips_loss,
            "flow_loss": flow_loss.item() if isinstance(flow_loss, torch.Tensor) else flow_loss,
            "align_loss": align_loss.item() if isinstance(align_loss, torch.Tensor) else align_loss,
            "n_align_samples": n_align_samples,
            "perplexity": perplexity.item() if isinstance(perplexity, torch.Tensor) else perplexity,
            "num_unique_indices": (
                num_unique_indices.item() if isinstance(num_unique_indices, torch.Tensor) else num_unique_indices
            ),
            **align_logs,
        }
        avg_probs = vq_stats.get("avg_probs")
        if avg_probs is not None:
            for idx, prob in enumerate(avg_probs.detach().float().cpu().tolist()):
                log_dict[f"code_prob_{idx}"] = prob

        return loss, log_dict

    def encode(self, videos: torch.Tensor, mask: torch.BoolTensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = videos.device
        B = videos.shape[0]
        Hp, Wp = self.patch_grid

        videos_unrolled = rearrange(videos, "b t c h w -> (b t) c h w")
        frame_tokens, tokens = self.enc_spatial_transformer(videos_unrolled)

        frame_tokens = rearrange(frame_tokens, "(b t) (hp wp) d -> b t hp wp d", b=B, hp=Hp, wp=Wp)
        tokens = rearrange(tokens, "(b t) (hp wp) d -> b t (hp wp) d", b=B, hp=Hp, wp=Wp)

        tokens_shape = tuple(frame_tokens.shape[:-1])  
        attn_bias = self.enc_spatial_rel_pos_bias(Hp, Wp, device=device, dtype=tokens.dtype)
        tokens = self.enc_st_transformer(
            tokens, tokens_shape, spatial_attn_bias=attn_bias, attn_mask=mask
        )  
        tokens = rearrange(tokens, "b t (hp wp) d -> b t hp wp d", b=B, hp=Hp, wp=Wp)

        return frame_tokens, tokens

    def quantize(
        self, tokens: torch.Tensor, inference_mode: bool = False
    ) -> Union[Tuple[torch.Tensor, float, int], Tuple[torch.Tensor, torch.Tensor]]:
        B = tokens.shape[0]
        Hp, Wp = self.patch_grid
        Hq, Wq = self.action_size

        tokens = self.vq_project_in(tokens)  
        tokens = rearrange(tokens, "b t hp wp d -> b d t hp wp", b=B, hp=Hp, wp=Wp)
        tokens = self.vq_encoder(tokens)  

        tokens = rearrange(tokens, "b d t hq wq -> (b t hq wq) d", b=B, hq=Hq, wq=Wq)

        if inference_mode:
            quantized_actions, quantized_action_idxs = self.vq.inference(tokens)

            quantized_actions = rearrange(quantized_actions, "(b t hq wq) d -> b t hq wq d", b=B, hq=Hq, wq=Wq)
            quantized_action_idxs = rearrange(quantized_action_idxs, "(b t hq wq) -> b t hq wq", b=B, hq=Hq, wq=Wq)

            return quantized_actions, quantized_action_idxs

        quantized_actions, perplexity, num_unique_indices, vq_stats = self.vq(tokens)  

        quantized_actions = rearrange(quantized_actions, "(b t hq wq) d -> b t hq wq d", b=B, hq=Hq, wq=Wq)

        return quantized_actions, perplexity, num_unique_indices, vq_stats

    def decode(
        self, features: torch.Tensor, quantized_actions: torch.Tensor, mask_recon: torch.BoolTensor = None
    ) -> torch.Tensor:
        
        device = features.device
        B, Tm1 = features.shape[0], features.shape[1]
        Hp, Wp = self.patch_grid

        patch_tokens = rearrange(features, "b t hp wp d -> (b t) (hp wp) d", b=B, hp=Hp, wp=Wp)

        cond = self.vq_to_cond(quantized_actions)  

        tokens = torch.cat([patch_tokens, rearrange(cond, "b t d -> (b t) 1 d")], dim=1)  
        tokens = rearrange(tokens, "(b t) np d -> b t np d", b=B, t=Tm1, np=Hp * Wp + 1)  
        videos_shape = (B, Tm1, Hp, Wp)

        attn_bias = self.dec_spatial_rel_pos_bias(Hp, Wp, device=device, dtype=tokens.dtype)  
        attn_bias = F.pad(attn_bias, (0, 1, 0, 1), value=0.0)  
        tokens = self.dec_transformer(
            tokens,
            videos_shape,
            cond=cond,
            spatial_attn_bias=attn_bias,
            attn_mask=mask_recon,
        )  

        visual_tokens = tokens[:, :, :-1, :]
        pred_tokens = self.to_dino_features(visual_tokens)  
        pred_features = rearrange(pred_tokens, "b t (hp wp) d -> b t hp wp d", b=B, hp=Hp, wp=Wp)

        return pred_features

    def compute_semantic_reconstruction_loss(
        self, pred_features: torch.Tensor, target_features: torch.Tensor, mask_loss: torch.BoolTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        cosine_loss_map = 1.0 - F.cosine_similarity(pred_features, target_features.detach(), dim=-1)  
        norm_loss_map = torch.abs(pred_features.norm(dim=-1) - target_features.detach().norm(dim=-1))
        if exists(mask_loss):
            mult_mask = mask_loss[..., None, None].to(cosine_loss_map.dtype)
            denom = (mask_loss.sum() * pred_features.shape[2] * pred_features.shape[3]).clamp_min(1).to(cosine_loss_map.dtype)
            semantic_cosine_loss = (cosine_loss_map * mult_mask).sum() / denom
            feature_norm_loss = (norm_loss_map * mult_mask).sum() / denom
        else:
            semantic_cosine_loss = cosine_loss_map.mean()
            feature_norm_loss = norm_loss_map.mean()
        semantic_loss = semantic_cosine_loss + self.feature_norm_loss_weight * feature_norm_loss
        return semantic_loss, semantic_cosine_loss, feature_norm_loss

    def compute_lpips_loss(
        self, recon_videos: torch.Tensor, gt_videos: torch.Tensor, mask_loss: torch.BoolTensor
    ) -> torch.Tensor:
        
        B, T, C, H, W = recon_videos.shape

        flat_recon_videos = rearrange(recon_videos, "b t c h w -> (b t) c h w")
        flat_gt_videos = rearrange(gt_videos, "b t c h w -> (b t) c h w")

        flat_recon_videos = 2 * flat_recon_videos - 1
        flat_gt_videos = 2 * flat_gt_videos - 1

        lpips_loss = self.lpips(flat_recon_videos, flat_gt_videos).squeeze()  

        if exists(mask_loss):
            lpips_loss = lpips_loss[rearrange(mask_loss, "b t -> (b t)")]
            denom = mask_loss.sum().clamp_min(1).to(lpips_loss.dtype)
            return lpips_loss.sum() / denom
        else:
            return lpips_loss.mean()

    def compute_align_loss(
        self,
        quantized_actions: torch.Tensor,          
        actions: torch.Tensor,                    
        action_mask: torch.BoolTensor,            
        mask_loss: torch.BoolTensor,              
    ) -> Tuple[torch.Tensor, int, Dict[str, float]]:
        
        pred_actions = self.action_align_head(quantized_actions)
        gt_actions = actions[:, 1:]
        combined_mask = action_mask[:, None] & mask_loss

        n_align_samples = int(combined_mask.sum().item())
        if n_align_samples == 0:
            return torch.tensor(0.0, device=quantized_actions.device), 0, {}

        pred_flat = pred_actions[combined_mask]
        gt_flat = gt_actions[combined_mask]

        scale = torch.ones(pred_flat.shape[-1], device=pred_flat.device, dtype=pred_flat.dtype)
        if pred_flat.shape[-1] >= 14:
            trans_dims = [0, 1, 2, 7, 8, 9]
            rot_dims = [3, 4, 5, 10, 11, 12]
            grip_dims = [6, 13]
            scale[trans_dims] = self.align_action_translation_scale
            scale[rot_dims] = self.align_action_rotation_scale
            scale[grip_dims] = self.align_action_gripper_scale
        else:
            trans_dims = list(range(pred_flat.shape[-1]))
            rot_dims = []
            grip_dims = []

        scale = scale.clamp_min(1e-8)
        pred_norm = pred_flat / scale
        gt_norm = gt_flat / scale
        align_loss = F.mse_loss(pred_norm, gt_norm)

        with torch.no_grad():
            raw_sq = (pred_flat - gt_flat).pow(2)
            norm_sq = (pred_norm - gt_norm).pow(2)
            raw_abs = (pred_flat - gt_flat).abs()
            align_logs = {
                "align_raw_mse": raw_sq.mean().item(),
                "align_norm_mse": align_loss.detach().item(),
                "align_pred_abs_mean": pred_flat.abs().mean().item(),
                "align_gt_abs_mean": gt_flat.abs().mean().item(),
                "align_error_abs_mean": raw_abs.mean().item(),
            }
            if trans_dims:
                align_logs["align_trans_raw_mse"] = raw_sq[:, trans_dims].mean().item()
                align_logs["align_trans_norm_mse"] = norm_sq[:, trans_dims].mean().item()
            if rot_dims:
                align_logs["align_rot_raw_mse"] = raw_sq[:, rot_dims].mean().item()
                align_logs["align_rot_norm_mse"] = norm_sq[:, rot_dims].mean().item()
            if grip_dims:
                align_logs["align_grip_raw_mse"] = raw_sq[:, grip_dims].mean().item()
                align_logs["align_grip_norm_mse"] = norm_sq[:, grip_dims].mean().item()

        return align_loss, n_align_samples, align_logs

    def compute_flow_loss_weight(self, step: int) -> float:
        if self.flow_loss_warmup_steps > 0:
            warmup_progress = (step - self.flow_loss_kickin_step) / self.flow_loss_warmup_steps
            warmup_progress = min(warmup_progress, 1.0)
            current_flow_weight = warmup_progress * self.flow_loss_weight
        else:
            current_flow_weight = self.flow_loss_weight
        return current_flow_weight

    def get_flow(self, vid0, vid1):
        
        B, C, T, H, W = vid0.shape
        v0 = rearrange(vid0, "b c t h w -> (b t) c h w")
        v1 = rearrange(vid1, "b c t h w -> (b t) c h w")
        flow = self.flow_model(v0, v1)[-1]
        return rearrange(flow, "(b t) c h w -> b c t h w", b=B)

    def compute_flow_loss(self, recon_videos: torch.Tensor, gt_videos: torch.Tensor, mask_loss: torch.BoolTensor):
        
        B, T, C, H, W = recon_videos.shape
        recon_videos_flow = rearrange(recon_videos, "b t c h w -> b c t h w")
        gt_videos_flow = rearrange(gt_videos, "b t c h w -> b c t h w")

        recon_videos_flow = 2 * recon_videos_flow - 1
        gt_videos_flow = 2 * gt_videos_flow - 1

        with torch.no_grad():
            flow_gt_fwd = self.get_flow(gt_videos_flow[:, :, :-1], gt_videos_flow[:, :, 1:])
            flow_gt_bwd = self.get_flow(gt_videos_flow[:, :, 1:], gt_videos_flow[:, :, :-1])

        flow_recon_fwd = self.get_flow(recon_videos_flow[:, :, :-1], recon_videos_flow[:, :, 1:])
        flow_recon_bwd = self.get_flow(recon_videos_flow[:, :, 1:], recon_videos_flow[:, :, :-1])

        flow_loss_fwd = F.l1_loss(flow_recon_fwd, flow_gt_fwd, reduction="none")
        flow_loss_bwd = F.l1_loss(flow_recon_bwd, flow_gt_bwd, reduction="none")
        flow_loss = flow_loss_fwd + flow_loss_bwd  

        if exists(mask_loss):
            flow_mask = torch.logical_and(mask_loss[:, 1:], mask_loss[:, :-1])  
            mult_mask = flow_mask[:, None, :, None, None].to(flow_loss.dtype)  
            denom = (flow_mask.sum() * 2 * H * W).clamp_min(1).to(flow_loss.dtype)
            return 0.5 * (flow_loss * mult_mask).sum() / denom
        else:
            return 0.5 * flow_loss.mean()

    def state_dict(
        self, destination: Optional[Dict[str, Any]] = None, prefix: str = "", keep_vars: bool = False
    ) -> Dict[str, Any]:
        
        full_state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        filtered_state = {
            k: v for k, v in full_state.items() if not (k.startswith("lpips.") or k.startswith("flow_model."))
        }

        return filtered_state

    def get_trainable_parameters(
        self,
        lr: float,
        no_decay_keywords: List[str] = ["softvq.", "vq.", "codebooks"],
        filter_keywords: List[str] = [],
        pretrained_init_keywords: List[str] = [],
        pretrained_init_lr_mult_factor: float = 1.0,
    ) -> List[Dict[str, Any]]:
        
        pt_decay, pt_no_decay = [], []
        new_decay, new_no_decay = [], []

        pt_decay_names, pt_no_decay_names = [], []
        new_decay_names, new_no_decay_names = [], []

        excluded_param_names: set[str] = set()
        if hasattr(self, "use_lpips_loss") and self.use_lpips_loss and hasattr(self, "lpips"):
            for name, _ in self.lpips.named_parameters():
                excluded_param_names.add(f"lpips.{name}")
        if hasattr(self, "use_flow_loss") and self.use_flow_loss and hasattr(self, "flow_model"):
            for name, _ in self.flow_model.named_parameters():
                excluded_param_names.add(f"flow_model.{name}")

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if name in excluded_param_names:
                logger.info(f"Excluding {name} from optimizer (LPIPS/Flow).")
                continue

            if any(keyword in name for keyword in filter_keywords):
                logger.info(f"Filtering out {name} from optimizer due to filter_keywords.")
                continue

            is_pretrained = any(keyword in name for keyword in pretrained_init_keywords)

            is_no_decay = param.ndim == 1 or any(keyword in name for keyword in no_decay_keywords)

            if is_pretrained:
                if is_no_decay:
                    pt_no_decay.append(param)
                    pt_no_decay_names.append(name)
                else:
                    pt_decay.append(param)
                    pt_decay_names.append(name)
            else:
                if is_no_decay:
                    new_no_decay.append(param)
                    new_no_decay_names.append(name)
                else:
                    new_decay.append(param)
                    new_decay_names.append(name)

        logger.info(f"Pre-trained params with decay: {pt_decay_names}")
        logger.info(f"Pre-trained params without decay: {pt_no_decay_names}")
        logger.info(f"New params with decay: {new_decay_names}")
        logger.info(f"New params without decay: {new_no_decay_names}")

        param_groups = []

        pretrain_lr = lr * pretrained_init_lr_mult_factor
        if len(pt_decay) > 0:
            param_groups.append({"params": pt_decay, "lr": pretrain_lr})
        if len(pt_no_decay) > 0:
            param_groups.append({"params": pt_no_decay, "weight_decay": 0.0, "lr": pretrain_lr})

        if len(new_decay) > 0:
            param_groups.append({"params": new_decay})
        if len(new_no_decay) > 0:
            param_groups.append({"params": new_no_decay, "weight_decay": 0.0})

        return param_groups

    def load_weights(
        self,
        ckpt_path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = False,
        verbose: bool = True,
    ) -> Optional[nn.modules.module._IncompatibleKeys]:
        
        ckpt_path = Path(ckpt_path)

        if not ckpt_path.exists():
            if verbose:
                logger.error(f"Checkpoint not found: {ckpt_path}")
            return None

        try:
            if verbose:
                logger.info(f"Loading weights from {ckpt_path} …")

            payload = torch.load(ckpt_path, map_location=map_location)

            if isinstance(payload, dict):
                if "model" in payload:  
                    state_dict = payload["model"]
                elif "module" in payload:  
                    state_dict = payload["module"]
                elif "state_dict" in payload:  
                    state_dict = payload["state_dict"]
                else:  
                    state_dict = payload
            else:
                state_dict = payload  

            incompatible = self.load_state_dict(state_dict, strict=strict)

            if verbose:
                miss, unexp = incompatible.missing_keys, incompatible.unexpected_keys
                logger.info(f"Loaded with " f"{len(miss)} missing / {len(unexp)} unexpected keys.")

            return incompatible

        except Exception as exc:
            if verbose:
                logger.error(f"Failed to load checkpoint: {exc}")
            return None

    @torch.no_grad()
    def inference(
        self,
        videos: torch.Tensor,  
        mask: Optional[torch.BoolTensor] = None,  
        return_reconstructions: bool = True,
        return_quantized_actions: bool = False,
        return_quantized_actions_idxs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        
        B, T, C, H, W = videos.shape
        device = videos.device
        mask = mask if mask is not None else torch.ones((B, T), device=device, dtype=torch.bool)
        mask_gt = mask[:, :-1]  
        mask_recon = mask[:, 1:]  
        mask_loss = torch.logical_and(mask_gt, mask_recon)  

        frame_tokens, tokens = self.encode(videos, mask=mask)  

        if self.encode_deltas:
            vq_inputs = tokens[:, 1:] - tokens[:, :-1]
        else:
            vq_inputs = tokens[:, 1:]

        quantized_actions, quantized_actions_idxs = self.quantize(vq_inputs, inference_mode=True)

        pred_features = self.decode(frame_tokens[:, :-1], quantized_actions, mask_recon=mask_gt)

        return_dict = {}
        if return_reconstructions:
            return_dict["pred_features"] = pred_features
            return_dict["target_features"] = frame_tokens[:, 1:]
        if return_quantized_actions and exists(quantized_actions):
            return_dict["quantized_actions"] = quantized_actions  
        if return_quantized_actions_idxs:
            return_dict["quantized_actions_idxs"] = quantized_actions_idxs  

        return return_dict

    @torch.no_grad()
    def rollout(
        self,
        features: torch.Tensor,  
        quantized_actions: torch.Tensor,  
    ):
        pred_features = self.decode(features, quantized_actions)  
        return pred_features

    @torch.no_grad()
    def rollout_ar(
        self,
        videos: torch.Tensor,  
        quantized_actions: torch.Tensor,  
    ):
        T_init = videos.shape[1]
        T_total = quantized_actions.shape[1]
        T_gen = T_total - T_init + 1  

        recon_videos = []
        history_frames = [videos[:, i] for i in range(T_init)]  
        for t in range(T_gen):
            history_tensor = torch.stack(history_frames, dim=1)  

            action_t = quantized_actions[:, : (t + T_init)]  

            next_frame = self.decode(history_tensor, action_t)  
            next_frame = next_frame[:, -1:]  
            recon_videos.append(next_frame)
            history_frames.append(next_frame.squeeze(1))  

        return torch.cat(recon_videos, dim=1)  
