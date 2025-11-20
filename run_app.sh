#!/bin/bash
# Helper script to run Streamlit with proper conda environment and CUDA library paths

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate affective_subtitler

# Set environment to use PyTorch's bundled CUDA libraries
export LD_LIBRARY_PATH=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH

# Disable TF32 for better numerical accuracy (as suggested by pyannote warnings)
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

echo "Starting Streamlit app with PyTorch bundled CUDA libraries..."
echo "Environment: affective_subtitler"
streamlit run app.py
