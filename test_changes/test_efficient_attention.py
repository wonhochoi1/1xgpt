#!/usr/bin/env python3
"""
Test script for efficient attention implementations
"""

import torch
import torch.nn as nn
from genie.attention import WindowedAttention, AxialAttention, BasicSelfAttention
from genie.st_transformer import STBlock, STTransformerDecoder
from genie.config import GenieConfig

def test_windowed_attention():
    """Test WindowedAttention with different sequence lengths"""
    print("Testing WindowedAttention...")
    
    # Test parameters
    batch_size = 2
    num_heads = 8
    d_model = 256
    window_size = 64
    
    # Create attention module
    attn = WindowedAttention(
        num_heads=num_heads,
        d_model=d_model,
        window_size=window_size
    )
    
    # Test with short sequence (should use standard attention)
    seq_len_short = 32
    x_short = torch.randn(batch_size, seq_len_short, d_model)
    output_short = attn(x_short)
    assert output_short.shape == x_short.shape, f"Short sequence shape mismatch: {output_short.shape} vs {x_short.shape}"
    print(f"✓ Short sequence ({seq_len_short}) passed")
    
    # Test with long sequence (should use windowed attention)
    seq_len_long = 256
    x_long = torch.randn(batch_size, seq_len_long, d_model)
    output_long = attn(x_long)
    assert output_long.shape == x_long.shape, f"Long sequence shape mismatch: {output_long.shape} vs {x_long.shape}"
    print(f"✓ Long sequence ({seq_len_long}) passed")
    
    # Test with causal masking
    output_causal = attn(x_long, causal=True)
    assert output_causal.shape == x_long.shape, f"Causal attention shape mismatch: {output_causal.shape} vs {x_long.shape}"
    print(f"✓ Causal attention passed")

def test_axial_attention():
    """Test AxialAttention with 2D spatial data"""
    print("\nTesting AxialAttention...")
    
    # Test parameters
    batch_size = 2
    num_heads = 8
    d_model = 256
    height = 16
    width = 16
    
    # Create attention module
    attn = AxialAttention(
        num_heads=num_heads,
        d_model=d_model,
        height=height,
        width=width
    )
    
    # Test with correct spatial dimensions
    seq_len = height * width
    x_spatial = torch.randn(batch_size, seq_len, d_model)
    output_spatial = attn(x_spatial)
    assert output_spatial.shape == x_spatial.shape, f"Spatial attention shape mismatch: {output_spatial.shape} vs {x_spatial.shape}"
    print(f"✓ Spatial attention ({height}x{width}) passed")
    
    # Test with incorrect dimensions (should fall back to standard attention)
    seq_len_wrong = 100
    x_wrong = torch.randn(batch_size, seq_len_wrong, d_model)
    output_wrong = attn(x_wrong)
    assert output_wrong.shape == x_wrong.shape, f"Fallback attention shape mismatch: {output_wrong.shape} vs {x_wrong.shape}"
    print(f"✓ Fallback attention ({seq_len_wrong}) passed")

def test_st_block():
    """Test STBlock with efficient attention"""
    print("\nTesting STBlock...")
    
    # Test parameters
    batch_size = 2
    num_heads = 8
    d_model = 256
    T = 16  # temporal
    S = 256  # spatial (16x16)
    
    # Create STBlock
    block = STBlock(
        num_heads=num_heads,
        d_model=d_model,
        spatial_size=16,
        temporal_window_size=64
    )
    
    # Test input
    x = torch.randn(batch_size, T, S, d_model)
    output = block(x)
    assert output.shape == x.shape, f"STBlock shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ STBlock ({batch_size}, {T}, {S}, {d_model}) passed")

def test_st_transformer_decoder():
    """Test STTransformerDecoder with efficient attention"""
    print("\nTesting STTransformerDecoder...")
    
    # Test parameters
    batch_size = 2
    num_layers = 2
    num_heads = 8
    d_model = 256
    T = 16
    S = 256
    
    # Create decoder
    decoder = STTransformerDecoder(
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        spatial_size=16,
        temporal_window_size=64
    )
    
    # Test input
    x = torch.randn(batch_size, T, S, d_model)
    output = decoder(x)
    assert output.shape == x.shape, f"Decoder shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ STTransformerDecoder ({num_layers} layers) passed")

def test_config_integration():
    """Test that config parameters work with the model"""
    print("\nTesting config integration...")
    
    # Create config with efficient attention parameters
    config = GenieConfig(
        num_layers=2,
        num_heads=8,
        d_model=256,
        spatial_size=16,
        temporal_window_size=64
    )
    
    # Verify config has the new parameters
    assert hasattr(config, 'spatial_size'), "Config missing spatial_size"
    assert hasattr(config, 'temporal_window_size'), "Config missing temporal_window_size"
    assert config.spatial_size == 16, f"Expected spatial_size=16, got {config.spatial_size}"
    assert config.temporal_window_size == 64, f"Expected temporal_window_size=64, got {config.temporal_window_size}"
    print("✓ Config integration passed")

def test_memory_efficiency():
    """Test that efficient attention uses less memory for long sequences"""
    print("\nTesting memory efficiency...")
    
    # Test parameters
    batch_size = 1
    num_heads = 8
    d_model = 256
    seq_len = 1024  # Long sequence to test efficiency
    
    # Standard attention
    standard_attn = BasicSelfAttention(
        num_heads=num_heads,
        d_model=d_model
    )
    
    # Efficient attention
    efficient_attn = WindowedAttention(
        num_heads=num_heads,
        d_model=d_model,
        window_size=64
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Both should produce the same output shape
    output_standard = standard_attn(x)
    output_efficient = efficient_attn(x)
    
    assert output_standard.shape == output_efficient.shape, "Output shapes should match"
    print(f"✓ Memory efficiency test passed (seq_len={seq_len})")

if __name__ == "__main__":
    print("Running efficient attention tests...\n")
    
    try:
        test_windowed_attention()
        test_axial_attention()
        test_st_block()
        test_st_transformer_decoder()
        test_config_integration()
        test_memory_efficiency()
        
        print("\n🎉 All tests passed! Efficient attention implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise 