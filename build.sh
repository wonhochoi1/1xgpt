#!/usr/bin/bash
# This script sets up a conda environment for 1xgpt and installs all required dependencies

# Define environment name and Python version
ENV_NAME="1xgpt"
PY_VERSION="3.10"  # Use Python 3.10 for compatibility

# Initialize conda in the current shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME}\s"; then
    echo "Creating new conda environment: ${ENV_NAME} with Python ${PY_VERSION}"
    conda create -y -n "$ENV_NAME" python="$PY_VERSION"
fi

# Activate the conda environment
echo "Activating conda environment: ${ENV_NAME}"
conda activate "$ENV_NAME"

# Install dependencies in the correct order
echo "Installing dependencies..."

# Install basic dependencies first (ones that don't depend on PyTorch)
echo "Installing basic dependencies..."
pip install wheel packaging ninja einops tqdm matplotlib wandb

# Install PyTorch first (required for other packages)
echo "Installing PyTorch..."
pip install torch==2.3.0 torchvision==0.18.0

# Install transformers and other packages that might depend on PyTorch
echo "Installing transformers and other dependencies..."
pip install transformers==4.41.0 accelerate==0.30.1 lpips==0.1.4 lightning>2.3.1

# Install huggingface_hub to get the CLI tools
echo "Installing Hugging Face tools..."
pip install huggingface_hub

echo 'export XFORMERS_DISABLED=true' >> ~/.zshrc
source ~/.zshrc
# Install xformers after PyTorch is installed
echo "Installing xformers..."
pip install xformers==0.0.26.post1

# Handle the git+https installation manually since it requires special handling
echo "Installing mup package from GitHub..."
pip install --no-deps git+https://github.com/janEbert/mup.git@fsdp-fix

# Install flash-attention with special environment variable
echo "Installing flash-attention..."
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
pip install flash-attn==2.5.8 --no-build-isolation
unset FLASH_ATTENTION_SKIP_CUDA_BUILD

# Download datasets to data/train_v1.0, data/val_v1.0
echo "Downloading datasets..."
huggingface-cli download 1x-technologies/worldmodel --repo-type dataset --local-dir data

echo "Setup complete! Use 'conda activate ${ENV_NAME}' to activate the environment in new terminals."
