#!/bin/bash
# Helper script to run Streamlit with proper CUDA library paths

# Set environment to use PyTorch's bundled CUDA libraries
export LD_LIBRARY_PATH=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH

# Disable TF32 for better numerical accuracy (as suggested by pyannote warnings)
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

echo "Starting Streamlit app with PyTorch bundled CUDA libraries..."
streamlit run app.py
