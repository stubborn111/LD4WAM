import logging

import torch
import torch.distributed as dist
import torch.distributions.uniform as uniform_dist
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SoftVQ(nn.Module):
    

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        temperature: float = 0.1,
        discarding_threshold: float = 0.01,
        initialization: str = "normal",
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.discarding_threshold = discarding_threshold
        self.eps = 1e-8

        if initialization == "normal":
            init_cb = torch.randn(num_embeddings, embedding_dim)
        elif initialization == "uniform":
            low, high = -1.0 / num_embeddings, 1.0 / num_embeddings
            init_cb = uniform_dist.Uniform(low, high).sample((num_embeddings, embedding_dim))
        else:
            raise ValueError("initialization must be 'normal' or 'uniform'")

        self.codebooks = nn.Parameter(init_cb)
        self.register_buffer("codebooks_used", torch.zeros(num_embeddings, dtype=torch.long))

    def _assignment_logits(self, input_data: torch.Tensor) -> torch.Tensor:
        x = F.normalize(input_data, dim=-1, eps=self.eps)
        cb = F.normalize(self.codebooks, dim=-1, eps=self.eps)
        return x @ cb.t() / max(self.temperature, self.eps)

    def forward(self, input_data: torch.Tensor):
        
        logits = self._assignment_logits(input_data)
        probs = torch.softmax(logits, dim=-1)
        quantized_input = probs @ self.codebooks
        max_idx = torch.argmax(probs, dim=-1)

        with torch.no_grad():
            local_counts = torch.bincount(max_idx, minlength=self.num_embeddings)
            self.codebooks_used += local_counts.to(self.codebooks_used.dtype)
            usage_counts = local_counts.to(torch.float32)
            if dist.is_initialized():
                dist.all_reduce(usage_counts, op=dist.ReduceOp.SUM)

            avg_probs = probs.detach().sum(dim=0)
            if dist.is_initialized():
                dist.all_reduce(avg_probs, op=dist.ReduceOp.SUM)
            avg_probs = avg_probs / avg_probs.sum().clamp_min(self.eps)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))
            num_unique_indices = int((usage_counts > 0).sum().item())

            logger.debug(f"SoftVQ perplexity: {perplexity.item()}")
            logger.debug(f"SoftVQ average usage: {avg_probs.cpu().tolist()}")

        avg_probs_for_loss = probs.mean(dim=0)
        uniform = torch.full_like(avg_probs_for_loss, 1.0 / self.num_embeddings)
        usage_kl_loss = torch.sum(avg_probs_for_loss * (torch.log(avg_probs_for_loss + self.eps) - torch.log(uniform + self.eps)))
        token_entropy_loss = -torch.sum(probs * torch.log(probs + self.eps), dim=-1).mean()
        vq_stats = {
            "usage_kl_loss": usage_kl_loss,
            "token_entropy_loss": token_entropy_loss,
            "avg_probs": avg_probs.detach(),
        }

        return quantized_input, perplexity, num_unique_indices, vq_stats

    @torch.no_grad()
    def replace_unused_codebooks(self):
        
        self.codebooks_used.zero_()

    @torch.no_grad()
    def inference(self, input_data: torch.Tensor):
        
        logits = self._assignment_logits(input_data)
        probs = torch.softmax(logits, dim=-1)
        quantized_input = probs @ self.codebooks
        max_idx = torch.argmax(probs, dim=-1)
        return quantized_input, max_idx

    @torch.no_grad()
    def convert_idx_to_embeddings(self, idxs: torch.Tensor):
        out_shape = idxs.shape + (self.embedding_dim,)
        idxs = idxs.reshape(-1)
        embeddings = self.codebooks[idxs]
        return embeddings.reshape(out_shape)
