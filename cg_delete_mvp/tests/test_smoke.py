#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Smoke tests for CG Delete MVP.

Basic sanity checks to ensure modules import correctly and core functionality works.
"""

import sys
import os
import pytest
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import utils
from src.attention_delete import AttentionEraser, HarmConfig, build_sot_exempt_mask


def test_imports():
    """Test that all modules import without errors."""
    from src import utils
    from src import attention_delete
    from src import cli

    assert utils is not None
    assert attention_delete is not None
    assert cli is not None


def test_harm_config():
    """Test HarmConfig creation and properties."""
    cfg = HarmConfig(enable=True, tau=0.15, gamma=1.5)

    assert cfg.enable == True
    assert cfg.tau == 0.15
    assert cfg.gamma == 1.5


def test_attention_eraser_init():
    """Test AttentionEraser initialization."""
    # Create dummy harm vector
    harm_vec = torch.randn(768)

    cfg = HarmConfig(enable=True, tau=0.1, gamma=1.0)
    eraser = AttentionEraser(harm_vec=harm_vec, harm_cfg=cfg)

    assert eraser.harm_vec is not None
    assert eraser.harm_vec.shape == (768,)
    assert eraser.harm_cfg.enable == True


def test_attention_eraser_no_harm():
    """Test AttentionEraser with no harm vector."""
    cfg = HarmConfig(enable=False, tau=0.1, gamma=1.0)
    eraser = AttentionEraser(harm_vec=None, harm_cfg=cfg)

    assert eraser.harm_vec is None
    assert eraser.harm_cfg.enable == False


def test_utils_set_seed():
    """Test seed setting."""
    utils.set_seed(42)

    # Check that random operations are deterministic
    a = torch.rand(10)
    utils.set_seed(42)
    b = torch.rand(10)

    assert torch.allclose(a, b)


def test_utils_schedulers():
    """Test scheduling functions."""
    # Linear scheduler
    sched_linear = utils.get_scheduler("linear")
    val = sched_linear(0, 10, 1.0, 0.0)
    assert abs(val - 1.0) < 1e-6

    val = sched_linear(10, 10, 1.0, 0.0)
    assert abs(val - 0.0) < 1e-6

    # Cosine scheduler
    sched_cosine = utils.get_scheduler("cosine")
    val = sched_cosine(0, 10, 1.0, 0.0)
    assert abs(val - 1.0) < 1e-6

    # Fixed scheduler
    sched_fixed = utils.get_scheduler("fixed")
    val = sched_fixed(5, 10, 1.0, 0.0)
    assert abs(val - 1.0) < 1e-6


def test_build_content_mask():
    """Test content mask building."""
    # Create mock tokenizer
    class MockTokenizer:
        bos_token_id = 0
        eos_token_id = 2
        pad_token_id = 1

    tokenizer = MockTokenizer()

    # Create dummy input
    input_ids = torch.tensor([[0, 5, 6, 7, 2, 1, 1]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])

    # Test with special tokens excluded
    mask = utils.build_content_mask(tokenizer, input_ids, attention_mask, include_special=False)
    expected = torch.tensor([[False, False, True, True, False, False, False]])

    assert torch.equal(mask, expected)

    # Test with special tokens included
    mask = utils.build_content_mask(tokenizer, input_ids, attention_mask, include_special=True)
    expected = torch.tensor([[True, False, True, True, True, False, False]])

    assert torch.equal(mask, expected)


def test_eraser_setters():
    """Test AttentionEraser setter methods."""
    cfg = HarmConfig(enable=True, tau=0.1, gamma=1.0)
    eraser = AttentionEraser(harm_vec=torch.randn(768), harm_cfg=cfg)

    # Test set_harm_gamma
    eraser.set_harm_gamma(2.5)
    assert eraser.harm_cfg.gamma == 2.5

    # Test set_harm_vec
    new_vec = torch.randn(768)
    eraser.set_harm_vec(new_vec)
    assert eraser.harm_vec is not None

    # Test set_soft_exempt_mask
    mask = torch.tensor([[True, False, False]])
    eraser.set_soft_exempt_mask(mask)
    assert eraser._soft_exempt_mask is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_attention_eraser_forward():
    """Test AttentionEraser forward pass (requires GPU).

    This test is skipped on CPU-only environments.
    """
    from diffusers.models.attention_processor import Attention

    # Create dummy attention module
    dim = 320
    cross_dim = 768
    num_heads = 8

    attn = Attention(
        query_dim=dim,
        cross_attention_dim=cross_dim,
        heads=num_heads,
        dim_head=dim // num_heads,
    ).cuda()

    # Create eraser
    harm_vec = torch.randn(cross_dim)
    cfg = HarmConfig(enable=True, tau=0.1, gamma=1.0)
    eraser = AttentionEraser(harm_vec=harm_vec, harm_cfg=cfg)

    # Dummy inputs
    batch_size = 2
    seq_len = 16
    text_len = 77

    hidden_states = torch.randn(batch_size, seq_len, dim).cuda()
    encoder_hidden_states = torch.randn(batch_size, text_len, cross_dim).cuda()

    # Forward pass
    output = eraser(attn, hidden_states, encoder_hidden_states)

    assert output.shape == hidden_states.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
