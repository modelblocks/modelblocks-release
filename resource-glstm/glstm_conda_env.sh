#!/bin/bash

# --- Configuration ---
ENV_NAME="glstm_env"
PYTHON_VER="3.5"
# Using CUDA 10.0. Change 'cu100' to 'cpu' if you don't have a GPU.
TORCH_WHL_URL="https://download.pytorch.org/whl/cpu/torch_stable.html"
#TORCH_WHL_URL="https://download.pytorch.org/whl/cu100/torch_stable.html"

# Stop script on first error
#set -e

echo "============================================"
echo "Starting automated setup for $ENV_NAME"
echo "============================================"

# 1. Initialize Conda for shell interaction
# This is necessary because 'conda activate' often doesn't work in shell scripts 
# without initializing the shell hook first.
#eval "$(conda shell.bash hook)"
source ~/miniconda3/etc/profile.d/conda.sh

# 2. Clean up existing environment if it exists
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Envinroment '$ENV_NAME' already exists. Exiting."
    exit 1
else
    echo "No existing environment found. Proceeding..."
fi

# 3. Create the environment
# Use --override-channels -c defaults to prevent Conda from searching 
# huge 3rd party repos
echo "--------------------------------------------"
echo "Creating Conda environment (Python $PYTHON_VER)..."
conda create -n $ENV_NAME python=$PYTHON_VER --override-channels -c defaults -y

# 4. Activate the environment
echo "--------------------------------------------"
echo "Activating $ENV_NAME..."
conda activate $ENV_NAME

# 5. Install PyTorch 1.0.0 via PIP
# This avoids the channel conflicts found in Conda
echo "--------------------------------------------"
echo "Installing PyTorch 1.0.0 and Torchvision 0.2.1..."
pip install torch==1.0.0 torchvision==0.2.1 -f $TORCH_WHL_URL

# 6. Install other dependencies
echo "--------------------------------------------"
echo "Installing Pandas and Numpy..."
pip install pandas numpy

# 8. Verification
echo "--------------------------------------------"
echo "Verifying installation..."
python -c "import torch; print('Success! PyTorch Version:', torch.__version__)"
python -c "import pandas; print('Success! Pandas Version:', pandas.__version__)"

echo "============================================"
echo "Setup Complete!"
echo "To use this environment, run: conda activate $ENV_NAME"
echo "============================================"

conda deactivate
