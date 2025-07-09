#!/usr/bin/env python3
"""
Test script to verify action conditioning implementation with v2.0 data format.
"""

import torch
import numpy as np
import json
from pathlib import Path

# Add current directory to path
import sys
sys.path.append('..') 

from genie.config import GenieConfig
from genie.st_mask_git import STMaskGIT


def test_action_conditioning():
    """Test that action conditioning works correctly."""
    print("Testing action conditioning implementation...")
    
    # Create a minimal config for testing
    config = GenieConfig(
        num_layers=2,
        num_heads=4,
        d_model=128,
        T=4,  # Small temporal dimension for testing
        S=16,  # Small spatial dimension for testing
        image_vocab_size=1000,
        action_vocab_size=100,
        action_embed_dim=64,
        use_action_conditioning=True,
        use_v2_format=True,  # Use v2.0 format
        num_action_dims=25,  # 25 state dimensions
        factored_vocab_size=1000,
        num_factored_vocabs=1
    )
    
    # Create model
    model = STMaskGIT(config)
    model.init_weights()  # Initialize weights properly
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass with action tokens
    batch_size = 2
    T, H, W = config.T, int(np.sqrt(config.S)), int(np.sqrt(config.S))
    
    # Create dummy input data
    input_ids = torch.randint(0, config.image_vocab_size, (batch_size, T * H * W))
    labels = input_ids.clone()
    
    # Create dummy action tokens for v2.0 format
    # v2.0 has 25 state dimensions (0-24) as defined in the README
    action_tokens = {
        'robot_states': torch.randint(0, config.action_vocab_size, (batch_size, T - 1, 25)),  # 25 state dimensions
    }
    
    # Also test v1.1 format for backward compatibility
    config_v1 = GenieConfig(
        num_layers=2,
        num_heads=4,
        d_model=128,
        T=4,
        S=16,
        image_vocab_size=1000,
        action_vocab_size=100,
        action_embed_dim=64,
        use_action_conditioning=True,
        use_v2_format=False,  # Use v1.1 format
        factored_vocab_size=1000,
        num_factored_vocabs=1
    )
    
    model_v1 = STMaskGIT(config_v1)
    model_v1.init_weights()
    
    # v1.1 format action tokens
    action_tokens_v1 = {
        'neck_desired': torch.randint(0, config_v1.action_vocab_size, (batch_size, T - 1)),
        'driving_command': torch.randint(0, config_v1.action_vocab_size, (batch_size, T - 1)),
        'r_hand_closure': torch.randint(0, config_v1.action_vocab_size, (batch_size, T - 1)),
        'l_hand_closure': torch.randint(0, config_v1.action_vocab_size, (batch_size, T - 1)),
    }
    
    print("Testing v2.0 format forward pass with action conditioning...")
    outputs = model(input_ids=input_ids, labels=labels, action_tokens=action_tokens)
    print(f"v2.0 Forward pass successful! Loss: {outputs.loss.item():.4f}")
    
    # Test v1.1 format
    print("Testing v1.1 format forward pass with action conditioning...")
    outputs_v1 = model_v1(input_ids=input_ids, labels=labels, action_tokens=action_tokens_v1)
    print(f"v1.1 Forward pass successful! Loss: {outputs_v1.loss.item():.4f}")
    
    # Test forward pass without action tokens
    print("Testing forward pass without action conditioning...")
    outputs_no_action = model(input_ids=input_ids, labels=labels)
    print(f"Forward pass successful! Loss: {outputs_no_action.loss.item():.4f}")
    
    # Test compute_logits method for v2.0
    print("Testing v2.0 compute_logits method...")
    x_THW = input_ids.reshape(batch_size, T, H, W)
    logits_with_action = model.compute_logits(x_THW, action_tokens)
    logits_without_action = model.compute_logits(x_THW, None)
    
    print(f"v2.0 Logits shape with action: {logits_with_action.shape}")
    print(f"v2.0 Logits shape without action: {logits_without_action.shape}")
    
    # Verify that action conditioning changes the output
    logits_diff = torch.abs(logits_with_action - logits_without_action).mean()
    print(f"v2.0 Average difference in logits: {logits_diff.item():.6f}")
    
    if logits_diff > 1e-6:
        print("✅ v2.0 Action conditioning is working - logits are different!")
    else:
        print("❌ v2.0 Action conditioning may not be working - logits are identical")
    
    # Test compute_logits method for v1.1
    print("Testing v1.1 compute_logits method...")
    logits_with_action_v1 = model_v1.compute_logits(x_THW, action_tokens_v1)
    logits_without_action_v1 = model_v1.compute_logits(x_THW, None)
    
    print(f"v1.1 Logits shape with action: {logits_with_action_v1.shape}")
    print(f"v1.1 Logits shape without action: {logits_without_action_v1.shape}")
    
    # Verify that action conditioning changes the output
    logits_diff_v1 = torch.abs(logits_with_action_v1 - logits_without_action_v1).mean()
    print(f"v1.1 Average difference in logits: {logits_diff_v1.item():.6f}")
    
    if logits_diff_v1 > 1e-6:
        print("✅ v1.1 Action conditioning is working - logits are different!")
    else:
        print("❌ v1.1 Action conditioning may not be working - logits are identical")
    
    print("Action conditioning test completed!")


def test_v2_data_loading():
    """Test that v2.0 data loading works correctly."""
    print("\nTesting v2.0 data loading...")
    
    # Check if v2.0 data exists
    data_dir = Path("../data/train_v2.0")
    if not data_dir.exists():
        print("❌ v2.0 data directory not found, skipping data loading test")
        return
    
    try:
        # Read metadata
        metadata_path = data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✅ Metadata loaded: {metadata}")
        else:
            print("❌ Metadata file not found")
            return
        
        # Check for video and states files
        videos_dir = data_dir / "videos"
        states_dir = data_dir / "robot_states"
        
        if videos_dir.exists() and states_dir.exists():
            video_files = list(videos_dir.glob("video_*.bin"))
            states_files = list(states_dir.glob("states_*.bin"))
            
            print(f"✅ Found {len(video_files)} video shards")
            print(f"✅ Found {len(states_files)} states shards")
            
            if video_files and states_files:
                # Try to load a small sample from the first shard
                shard_id = 0
                video_path = videos_dir / f"video_{shard_id}.bin"
                states_path = states_dir / f"states_{shard_id}.bin"
                
                if video_path.exists() and states_path.exists():
                    print(f"✅ Testing data loading from shard {shard_id}")
                    
                    # Load a small sample of video data (assuming int32 tokens)
                    # v2.0 uses Cosmos Tokenizer with DV8×8×8 tokens
                    video_data = np.memmap(video_path, dtype=np.int32, mode="r")
                    print(f"  Video data shape: {video_data.shape}")
                    
                    # Load states data (float32)
                    states_data = np.memmap(states_path, dtype=np.float32, mode="r")
                    print(f"  States data shape: {states_data.shape}")
                    
                    # Reshape based on expected format
                    # Assuming video data is (num_frames, 3, 32, 32) for DV8×8×8
                    if len(video_data.shape) == 1:
                        # Reshape based on total size
                        total_elements = video_data.shape[0]
                        frames_per_batch = 17  # From cosmos_video_decoder.py
                        elements_per_frame = 3 * 32 * 32  # 3 channels, 32x32 spatial
                        num_frames = total_elements // elements_per_frame
                        
                        video_reshaped = video_data[:num_frames * elements_per_frame].reshape(num_frames, 3, 32, 32)
                        print(f"  Reshaped video data: {video_reshaped.shape}")
                    
                    # States should be (num_frames, 25) for the 25 state dimensions
                    if len(states_data.shape) == 1:
                        num_states = states_data.shape[0] // 25
                        states_reshaped = states_data[:num_states * 25].reshape(num_states, 25)
                        print(f"  Reshaped states data: {states_reshaped.shape}")
                    
                    print("✅ v2.0 data loading test successful!")
                else:
                    print(f"❌ Shard {shard_id} files not found")
            else:
                print("❌ No video or states files found")
        else:
            print("❌ Videos or robot_states directories not found")
            
    except Exception as e:
        print(f"❌ Error loading v2.0 data: {e}")


def test_v1_data_loading():
    """Test that v1.1 data loading still works (for backward compatibility)."""
    print("\nTesting v1.1 data loading (backward compatibility)...")
    
    # Check if v1.1 data exists
    data_dir = Path("../data/train_v1.1")
    if not data_dir.exists():
        print("❌ v1.1 data directory not found, skipping backward compatibility test")
        return
    
    # Check if video.bin exists (v1.1 has been replaced by v2.0)
    video_file = data_dir / "video.bin"
    if not video_file.exists():
        print("❌ v1.1 video.bin not found - v1.1 has been replaced by v2.0")
        print("   Use v2.0 data format instead")
        return
    
    try:
        # Try to load dataset with action conditioning
        from data import RawTokenDataset
        
        dataset = RawTokenDataset(
            data_dir, 
            window_size=4, 
            stride=1, 
            use_action_conditioning=True
        )
        
        # Get a sample
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        
        if "action_tokens" in sample:
            print("✅ v1.1 Action tokens loaded successfully!")
            for action_name, action_seq in sample["action_tokens"].items():
                print(f"  {action_name}: shape {action_seq.shape}")
        else:
            print("❌ v1.1 Action tokens not found in sample")
            
    except Exception as e:
        print(f"❌ Error loading v1.1 data: {e}")


if __name__ == "__main__":
    test_action_conditioning()
    test_v2_data_loading()
    test_v1_data_loading() 