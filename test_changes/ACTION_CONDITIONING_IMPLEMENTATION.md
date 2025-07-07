# Action Conditioning Implementation for GENIE

This document describes the implementation of action conditioning for the GENIE model, which allows the model to use robot action tokens to disambiguate future frame predictions.

## Overview

Action conditioning enhances the GENIE model by incorporating robot action information (neck_desired, driving_command, r_hand_closure, l_hand_closure) into the video generation process. This helps the model better understand the relationship between actions and visual outcomes, leading to more accurate future frame predictions.

## Changes Made

### 1. Configuration (`genie/config.py`)

Added new configuration parameters:
- `action_vocab_size`: Vocabulary size for action tokens (default: 1000)
- `action_embed_dim`: Embedding dimension for actions (default: 256)
- `use_action_conditioning`: Flag to enable/disable action conditioning (default: true)

### 2. Data Loading (`data.py`)

Updated `RawTokenDataset` to:
- Load action data from `data/train_v1.1/actions/` directory
- Support four action types: neck_desired, driving_command, r_hand_closure, l_hand_closure
- Return action tokens alongside video tokens in dataset items
- Updated collator function to handle action tokens in batches

### 3. Model Architecture (`genie/st_mask_git.py`)

Enhanced `STMaskGIT` model with:
- Action embedding layers for each action type
- Action projection layer to map concatenated action embeddings to model dimension
- Modified `compute_logits()` method to incorporate action embeddings
- Updated `forward()`, `generate()`, and `maskgit_generate()` methods to accept action tokens

### 4. Training (`train.py`)

Updated training script to:
- Pass action conditioning flag to dataset initialization
- Automatically handle action tokens in the training loop

### 5. Generation (`genie/generate.py`)

Updated generation script to:
- Load action tokens when available
- Pass action tokens to the generation process

### 6. Evaluation (`genie/evaluate.py`)

Updated evaluation script to:
- Use appropriate collator based on action conditioning setting
- Pass action tokens to evaluation process

## Usage

### Training with Action Conditioning

```bash
# Train with action conditioning enabled
python train.py \
    --genie_config genie/configs/magvit_n32_h8_d256_with_actions.json \
    --output_dir data/genie_model_with_actions \
    --max_eval_steps 10
```

### Generation with Action Conditioning

```bash
# Generate frames using action conditioning
python genie/generate.py \
    --checkpoint_dir data/genie_model_with_actions/final_checkpt \
    --output_dir data/genie_generated_with_actions \
    --example_ind 0 \
    --maskgit_steps 2 \
    --temperature 0
```

### Evaluation with Action Conditioning

```bash
# Evaluate model with action conditioning
python genie/evaluate.py \
    --checkpoint_dir data/genie_model_with_actions/final_checkpt \
    --maskgit_steps 2
```

## Configuration Files

### New Configuration File
- `genie/configs/magvit_n32_h8_d256_with_actions.json`: Configuration with action conditioning enabled

### Key Parameters
```json
{
    "action_vocab_size": 1000,
    "action_embed_dim": 256,
    "use_action_conditioning": true
}
```

## Data Structure

### Action Data Format
Action data is stored in binary files in `data/train_v1.1/actions/`:
- `neck_desired.bin`: Neck movement commands
- `driving_command.bin`: Driving/direction commands  
- `r_hand_closure.bin`: Right hand closure commands
- `l_hand_closure.bin`: Left hand closure commands

Each action file contains uint16 tokens corresponding to each frame in the video sequence.

### Model Input Format
The model expects action tokens as a dictionary:
```python
action_tokens = {
    'neck_desired': torch.LongTensor,      # Shape: (B, T-1)
    'driving_command': torch.LongTensor,   # Shape: (B, T-1)
    'r_hand_closure': torch.LongTensor,    # Shape: (B, T-1)
    'l_hand_closure': torch.LongTensor,    # Shape: (B, T-1)
}
```

## Implementation Details

### Action Embedding Process
1. Each action type is embedded separately using its own embedding layer
2. All action embeddings are concatenated along the feature dimension
3. Concatenated embeddings are projected to the model dimension
4. Action embeddings are added to token embeddings before transformer processing
5. First frame has no action (zero embedding) since actions represent transitions

### Backward Compatibility
- Models without action conditioning continue to work unchanged
- Action conditioning can be disabled by setting `use_action_conditioning: false`
- Existing checkpoints can be loaded and used normally

## Testing

Run the test script to verify the implementation:
```bash
python test_action_conditioning.py
```

This will test:
- Model creation with action conditioning
- Forward pass with and without action tokens
- Data loading with action tokens
- Verification that action conditioning affects model outputs

## Expected Benefits

1. **Better Future Prediction**: Action information helps disambiguate multiple possible future states
2. **Improved Temporal Consistency**: Actions provide causal information about frame transitions
3. **Enhanced Robot Understanding**: Model learns relationships between actions and visual outcomes
4. **More Realistic Generation**: Generated videos should better reflect the robot's intended actions

## Notes

- Action conditioning is additive to the existing token embeddings
- The implementation maintains the same training and inference interfaces
- Action tokens are optional - the model works with or without them
- The first frame in each sequence has no action embedding (represents initial state) 