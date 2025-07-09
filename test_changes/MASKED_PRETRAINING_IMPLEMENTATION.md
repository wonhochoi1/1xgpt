# Masked Pretraining Implementation

This document outlines the implementation of Masked Pretraining (MaskViT-style) for the 1XGPT video generation model.

## Overview

The masked pretraining feature bootstraps the transformer with bidirectional context before autoregressive fine-tuning, following the MaskViT approach. This involves:

1. **Higher initial mask ratios** (e.g., 80% → 10%) during pretraining
2. **Varied masking schedules** (cosine, linear, exponential)
3. **Two-stage training regimen**: masked pretraining → AR fine-tuning

## Files Changed/Created

### 1. New File: `masked_pretrain.py`

**Purpose**: Standalone script for masked pretraining that inherits from `train.py` but implements MaskViT-style training.

**Key Features**:
- Higher initial mask ratios (configurable via `--initial_mask_ratio`)
- Varied mask schedules (cosine, linear, exponential)
- Option to mask all frames or preserve first frame as context
- Specialized visualization for masked pretraining reconstructions
- Custom loss computation for masked token prediction

**Usage**:
```bash
python masked_pretrain.py \
    --genie_config configs/magvit_n32_h8_d256.json \
    --train_data_dir data/train_v1.1 \
    --val_data_dir data/val_v1.1 \
    --output_dir outputs/masked_pretrain \
    --initial_mask_ratio 0.8 \
    --final_mask_ratio 0.1 \
    --mask_schedule cosine \
    --mask_all_frames \
    --num_train_epochs 10
```

### 2. Modified: `data.py`

**Added**: `get_masked_pretrain_collator()` function

**Key Features**:
- Implements MaskViT-style masking across all frames
- Configurable mask ratios and schedules
- Option to mask all frames or preserve first frame as context
- Handles action conditioning during masked pretraining
- **Masking logic is now per-sample, per-frame:**
  - For each frame to be masked, a mask of shape `(batch, S)` is generated, where `S` is the number of spatial tokens.
  - For each sample in the batch, a random subset of tokens is selected to be masked.
  - The mask is applied to `x_THW[:, frame_idx]` for all samples, ensuring correct broadcasting and fixing previous dimension mismatch errors.

**Changes**:
```python
def get_masked_pretrain_collator(config: GenieConfig):
    ...
    for frame_idx in frames_to_mask:
        num_tokens_to_mask = int(S * current_mask_ratio)
        mask = torch.zeros((batch_size, S), dtype=torch.bool, device=device)
        for b in range(batch_size):
            mask_indices = torch.randperm(S, device=device)[:num_tokens_to_mask]
            mask[b, mask_indices] = True
        x_THW[:, frame_idx][mask] = mask_token_id
    ...
```

### 3. Modified: `genie/config.py`

**Added**: Masked pretraining configuration parameters

**New Parameters**:
- `initial_mask_ratio: float = 0.8` - Initial mask ratio for masked pretraining
- `final_mask_ratio: float = 0.1` - Final mask ratio for masked pretraining  
- `mask_schedule: str = "cosine"` - Schedule for mask ratio decay
- `mask_all_frames: bool = False` - Whether to mask all frames during pretraining
- `current_mask_ratio: float = None` - Current mask ratio (updated during training)

### 4. Modified: `train.py`

**Added**: Support for two-stage training regimen

**New Features**:
- `--masked_pretrain_checkpoint` argument to load masked pretraining checkpoints
- Automatic loading of masked pretraining weights before AR fine-tuning
- Seamless transition from masked pretraining to autoregressive training

**Usage**:
```bash
python train.py \
    --genie_config configs/magvit_n32_h8_d256.json \
    --masked_pretrain_checkpoint outputs/masked_pretrain/masked_pretrain_final \
    --train_data_dir data/train_v1.1 \
    --val_data_dir data/val_v1.1 \
    --output_dir outputs/ar_finetune
```

## Implementation Details

### Masking Strategy

The masked pretraining uses a progressive masking strategy:

1. **Initial Phase**: High mask ratio (80%) to force the model to learn strong representations
2. **Progressive Decay**: Mask ratio decreases over time using configurable schedules
3. **Final Phase**: Low mask ratio (10%) to prepare for autoregressive fine-tuning

### Mask Schedules

Three mask schedules are implemented:

1. **Cosine**: `mask_ratio = final_ratio + (initial_ratio - final_ratio) * cos(π * step / total_steps)`
2. **Linear**: `mask_ratio = initial_ratio - (initial_ratio - final_ratio) * step / total_steps`
3. **Exponential**: `mask_ratio = final_ratio + (initial_ratio - final_ratio) * exp(-5 * step / total_steps)`

### Training Regimen

The two-stage training approach:

1. **Stage 1 - Masked Pretraining**:
   - Use `masked_pretrain.py` with high initial mask ratios
   - Train model to reconstruct masked tokens across all frames
   - Focus on learning strong spatial-temporal representations

2. **Stage 2 - Autoregressive Fine-tuning**:
   - Use `train.py` with `--masked_pretrain_checkpoint`
   - Load pretrained weights from Stage 1
   - Fine-tune for autoregressive next-frame prediction

## Testing

### Unit Tests

- The test suite now passes all tests after fixing the masking logic to operate per-sample, per-frame.
- Previous dimension mismatch errors (e.g., `The size of tensor a (16) must match the size of tensor b (4) at non-singleton dimension 2`) are resolved.

### Integration Tests

1. **End-to-end masked pretraining**:
   ```bash
   python masked_pretrain.py --genie_config test_config.json --output_dir test_output
   ```

2. **Two-stage training**:
   ```bash
   # Stage 1
   python masked_pretrain.py --genie_config test_config.json --output_dir stage1
   
   # Stage 2  
   python train.py --genie_config test_config.json --masked_pretrain_checkpoint stage1 --output_dir stage2
   ```

## Benefits

1. **Better Initialization**: Masked pretraining provides better weight initialization for autoregressive training
2. **Bidirectional Context**: Model learns to use both past and future context during pretraining
3. **Robust Representations**: High mask ratios force the model to learn strong spatial-temporal representations
4. **Progressive Learning**: Mask schedule allows gradual transition to autoregressive objectives

## Configuration Examples

### Basic Masked Pretraining
```json
{
  "initial_mask_ratio": 0.8,
  "final_mask_ratio": 0.1,
  "mask_schedule": "cosine",
  "mask_all_frames": false
}
```

### Aggressive Masked Pretraining
```json
{
  "initial_mask_ratio": 0.9,
  "final_mask_ratio": 0.05,
  "mask_schedule": "exponential",
  "mask_all_frames": true
}
```

## Future Enhancements

1. **Adaptive Masking**: Implement adaptive masking based on token importance
2. **Curriculum Learning**: Progressive increase in sequence length during pretraining
3. **Multi-scale Masking**: Different mask ratios for different spatial scales
4. **Temporal Masking**: Specialized masking strategies for temporal dimensions

## Notes

- The masked pretraining uses the same model architecture as the autoregressive training
- Action conditioning is preserved during masked pretraining
- Visualization functions are adapted for masked pretraining reconstructions
- Checkpoint loading supports both standard and accelerator-saved checkpoints 