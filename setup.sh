#!/bin/bash

# Update package repositories
echo "Updating package repositories..."
sudo apt-get update

# Install dependencies for building Python and other tools
echo "Installing system dependencies..."
sudo apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
  libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev \
  git curl

# Download and install Python 3.11
echo "Installing Python 3.11..."
if ! command -v python3.11 &> /dev/null; then
  wget https://www.python.org/ftp/python/3.11.12/Python-3.11.12.tgz
  tar -xf Python-3.11.12.tgz
  cd Python-3.11.12
  ./configure --enable-optimizations
  make -j $(nproc)
  sudo make altinstall
  cd ..
  rm -rf Python-3.11.12 Python-3.11.12.tgz
  echo "Python 3.11.12 installed successfully."
else
  echo "Python 3.11 is already installed."
fi

# Create virtual environment
echo "Creating virtual environment..."
python3.11 -m venv llama_env
source llama_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
# Intentionally not installing torchaudio as it's not needed

# Install core dependencies
echo "Installing core dependencies..."
pip install accelerate==1.5.2 \
    bitsandbytes==0.45.5 \
    datasets==3.5.0 \
    evaluate==0.4.3 \
    numpy==2.0.2 \
    packaging==24.2 \
    pandas==2.2.2 \
    peft==0.14.0 \
    psutil==5.9.5 \
    safetensors==0.5.3 \
    scikit-learn==1.6.1 \
    sentencepiece==0.2.0 \
    tensorboard==2.18.0 \
    tokenizers==0.21.1 \
    transformers==4.51.3 \
    trl==0.15.2 \
    unsloth==2025.3.19

# Install the package in development mode if setup.py exists
if [ -f setup.py ]; then
  echo "Installing the package in development mode..."
  pip install -e .
else
  echo "No setup.py found, skipping package installation."
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p data
mkdir -p outputs
mkdir -p models

# Create a simple test to verify everything is working
echo "Creating a simple test script..."
cat > test_environment.py << 'EOF'
import torch
import unsloth
import transformers
import datasets
import trl
import peft

print("==== Environment Test ====")
print(f"Python: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Unsloth: {unsloth.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Datasets: {datasets.__version__}")
print(f"TRL: {trl.__version__}")
print(f"PEFT: {peft.__version__}")
print("All libraries imported successfully!")
EOF

# Make scripts executable
echo "Making scripts executable..."
chmod +x monitor_and_resume.sh full_training_with_resumption.sh 2>/dev/null || true

echo "==== Setup complete! ===="
echo "To activate this environment in the future, run: source llama_env/bin/activate"
echo "To test your environment, run: python test_environment.py"
echo "To start training, run: python run_training_with_checkpoints.py"
echo "For automatic resumption on OOM errors, run: ./monitor_and_resume.sh"