"""Minimal shape checks for LayerNorm — typical transformer tensor layout."""

import sys
import os

import torch

from layer_norm import LayerNorm


def test_transformer_shapes_b_t_d():
    # Typical pre-norm block: activations are (batch, seq, model_dim)
    B, T, D = 2, 128, 768
    x = torch.randn(B, T, D)

    ln = LayerNorm(ndim=D, bias=True)
    y = ln(x)

    # LN normalizes along the last dim only; leading dims unchanged
    assert x.shape == (B, T, D)
    assert y.shape == (B, T, D)
    assert ln.weight.shape == (D,)
    assert ln.bias is not None and ln.bias.shape == (D,)


def test_optional_print_dimensions(capsys):
    """Run with: pytest -s tests/test_layer_norm.py::test_optional_print_dimensions"""
    B, T, n_embd = 4, 16, 256
    x = torch.randn(B, T, n_embd)
    ln = LayerNorm(ndim=n_embd, bias=False)
    y = ln(x)
    print(f"x:  {tuple(x.shape)}  (B, T, D)")
    print(f"y:  {tuple(y.shape)}  (same as x)")
    print(f"γ:  {tuple(ln.weight.shape)}  (D,)")
    assert y.shape == x.shape


if __name__ == "__main__":
    # No pytest needed: python tests/test_layer_norm.py
    B, T, D = 2, 128, 768
    x = torch.randn(B, T, D)
    ln = LayerNorm(ndim=D, bias=True)
    y = ln(x)
    print("Transformer-style layout (batch, seq, hidden):")
    print(f"  x in:  {tuple(x.shape)}")
    print(f"  y out: {tuple(y.shape)}  (unchanged — norm is over last dim only)")
    print(f"  γ:     {tuple(ln.weight.shape)}")
    print(f"  β:     {tuple(ln.bias.shape)}")
    assert y.shape == x.shape
    print("ok")
