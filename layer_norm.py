"""Layer normalization implementation from scratch.

Helps to stabilize training by reducing the variance of the activations (high variance leads to exploding or vanishing gradients) and by
providing a stable gradient flow. The alternative would be smaller learning rates to combat this, but this would slow down training.

We can't do batch norm for transformers because the sequence length is variable (and thus we have a variable number of padding tokens that
would throw off the statistics), so we need to normalize each token individually.

Layer normalization is done per-token, not per-batch, across the feature dimension. Advantages include:
- Batch independence (size of batch does not affect quality / values of norm output)
- Consistent behavior between training and inference (batch norm needs to compute running statistics during inference)
- More natural for sequential models
- Handles different length sequences gracefully
- No synchronization needed across GPUs

The formula for layer normalization is:

    LN(x) = [(x - mu) / sqrt(variance + epsilon)] * gamma + beta

where mu is the mean of the input x and sigma is the standard deviation of the input x.

Gamma and beta (learnable parameters) are used to scale and shift the normalized output, and operate across the batch dimension though.
"""

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """Layer normalization implementation from scratch."""

    def __init__(self, ndim: int, bias: bool, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))  # gamma, shape [D]
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        # Mean/variance over the feature dimension only; keepdim for broadcasting.
        # PyTorch uses population variance (unbiased=False), same as nn.LayerNorm.
        breakpoint()
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (input - mean) / torch.sqrt(var + self.eps)
        # Affine is elementwise (scale/shift per feature), not a matmul.
        out = normalized * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
