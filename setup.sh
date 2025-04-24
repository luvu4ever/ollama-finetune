#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python -m venv llama_env
source llama_env/bin/activate

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo "Installing the package in development mode..."
pip install -e .

# Create data directory
echo "Creating data directory..."
mkdir -p data

echo "Setup complete! Now place your data files in the 'data' directory."
echo "To start training, run: python run_training.py"
echo "For more options, run: python run_training.py --help"