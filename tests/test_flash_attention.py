"""Output of flash attention should be the same as the regular attention"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from model import CausalSelfAttention, CausalFlashSelfAttention, GPTConfig


def small_config(**overrides):
    cfg = GPTConfig(
        block_size=32,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_models(config, seed=42):
    """Return (csa, cfsa) with shared c_attn weights."""
    torch.manual_seed(seed)
    csa = CausalSelfAttention(config).eval()
    cfsa = CausalFlashSelfAttention(config).eval()
    with torch.no_grad():
        cfsa.c_attn.weight.data.copy_(csa.c_attn.weight.data)
    return csa, cfsa


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_output_shapes():
    """O_final is (B, n_head, T, head_size) and L_final is (B, n_head, T)."""
    config = small_config()
    _, cfsa = make_models(config)
    B, T = 2, 8
    x = torch.randn(B, T, config.n_embd)

    O_final, L_final = cfsa(x)

    hs = config.n_embd // config.n_head
    assert O_final.shape == (B, config.n_head, T, hs), (
        f"Expected O_final shape {(B, config.n_head, T, hs)}, got {O_final.shape}"
    )
    assert L_final.shape == (B, config.n_head, T), (
        f"Expected L_final shape {(B, config.n_head, T)}, got {L_final.shape}"
    )


def test_csa_output_shapes():
    """CausalSelfAttention returns (B, T, C) output and a (k, v) cache tuple."""
    config = small_config()
    csa, _ = make_models(config)
    B, T = 2, 8
    x = torch.randn(B, T, config.n_embd)

    y, present_kv = csa(x)

    assert y.shape == (B, T, config.n_embd)
    assert len(present_kv) == 2  # (k, v)
    hs = config.n_embd // config.n_head
    assert present_kv[0].shape == (B, config.n_head, T, hs)  # k
    assert present_kv[1].shape == (B, config.n_head, T, hs)  # v


# ---------------------------------------------------------------------------
# Causal masking
# ---------------------------------------------------------------------------

def test_causal_masking():
    """Corrupting future tokens must not change outputs at earlier positions."""
    config = small_config()
    _, cfsa = make_models(config)
    B, T = 1, 8
    split = T // 2

    torch.manual_seed(0)
    x = torch.randn(B, T, config.n_embd)
    x_modified = x.clone()
    x_modified[:, split:, :] = torch.randn(B, T - split, config.n_embd)

    with torch.no_grad():
        O_orig, _ = cfsa(x)
        O_mod, _ = cfsa(x_modified)

    torch.testing.assert_close(
        O_orig[:, :, :split, :],
        O_mod[:, :, :split, :],
        msg="Future tokens should not influence earlier output positions",
    )


def test_csa_causal_masking():
    """Same causal-masking check for CausalSelfAttention."""
    config = small_config()
    csa, _ = make_models(config)
    B, T = 1, 8
    split = T // 2

    torch.manual_seed(0)
    x = torch.randn(B, T, config.n_embd)
    x_modified = x.clone()
    x_modified[:, split:, :] = torch.randn(B, T - split, config.n_embd)

    with torch.no_grad():
        y_orig, _ = csa(x)
        y_mod, _ = csa(x_modified)

    torch.testing.assert_close(
        y_orig[:, :split, :],
        y_mod[:, :split, :],
        msg="Future tokens should not influence earlier output positions",
    )


# ---------------------------------------------------------------------------
# Numerical agreement
# ---------------------------------------------------------------------------

def test_matches_standard_attention():
    """CausalFlashSelfAttention output should numerically match CausalSelfAttention.

    CausalSelfAttention returns the attention result after c_proj. Setting c_proj
    to identity exposes the raw pre-projection output so both implementations can
    be compared directly.  bfloat16 arithmetic inside the manual flash kernel
    introduces small errors, so we use a loose tolerance.
    """
    config = small_config()
    csa, cfsa = make_models(config)
    B, T = 2, 8

    torch.manual_seed(1)
    x = torch.randn(B, T, config.n_embd)

    # Make CSA's output projection an identity so it exposes the raw attn output.
    with torch.no_grad():
        csa.c_proj.weight.data.copy_(torch.eye(config.n_embd))

    with torch.no_grad():
        csa_out, _ = csa(x)   # (B, T, C) — identity proj means this equals raw attn
        O_final, _ = cfsa(x)  # (B, n_head, T, head_size)

    # Reshape CSA output to (B, n_head, T, head_size) for a direct comparison.
    nh = config.n_head
    hs = config.n_embd // nh
    csa_reshaped = csa_out.view(B, T, nh, hs).transpose(1, 2)

    # bfloat16 matmul in the flash kernel causes small numerical differences.
    torch.testing.assert_close(
        O_final.float(),
        csa_reshaped.float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_l_final_is_log_sum_exp():
    """L_final[b, h, q] must equal log(sum_k exp(S[b,h,q,k])) over causal keys.

    This is the only test that exercises L_final.  Any corruption of L_tile
    (e.g. the spurious '+ l_i' on the m_i + log(l_i) line) will be caught here.
    """
    config = small_config()
    _, cfsa = make_models(config)
    B, T = 2, 8

    torch.manual_seed(2)
    x = torch.randn(B, T, config.n_embd)

    with torch.no_grad():
        _, L_final = cfsa(x)

        # Recompute Q, K from the same projection weights so we can build
        # the expected log-sum-exp without running the tiled loop again.
        qkv = cfsa.c_attn(x)                                    # (B, T, 3*C)
        q, k, _ = qkv.split(config.n_embd, dim=2)
        nh, hs = config.n_head, config.n_embd // config.n_head
        q = q.view(B, T, nh, hs).transpose(1, 2).float()       # (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(1, 2).float()       # (B, nh, T, hs)

        scale = 1.0 / (hs ** 0.5)
        S = (q @ k.transpose(-2, -1)) * scale                   # (B, nh, T, T)

        # Apply causal mask: positions that a query cannot attend to get -inf.
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        S = S.masked_fill(~causal_mask, float('-inf'))

        # Expected: log-sum-exp over the key dimension.
        expected_L = torch.logsumexp(S, dim=-1)                 # (B, nh, T)

    torch.testing.assert_close(L_final.float(), expected_L, atol=1e-4, rtol=1e-4)
