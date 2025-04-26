FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system dependencies including C compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    wget \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install dependencies in batches to reduce memory usage
RUN pip install --no-cache-dir \
    accelerate==1.5.2 \
    bitsandbytes==0.45.5 \
    datasets==3.5.0 \
    evaluate==0.4.3 \
    numpy==2.0.2 \
    packaging==24.2 \
    pandas==2.2.2 \
    psutil==5.9.5

RUN pip install --no-cache-dir \
    peft==0.14.0 \
    safetensors==0.5.3 \
    scikit-learn==1.6.1 \
    sentencepiece==0.2.0 \
    tensorboard==2.18.0 \
    tokenizers==0.21.1

RUN pip install --no-cache-dir \
    transformers==4.51.3 \
    trl==0.15.2 \
    unsloth==2025.3.19
ENV TOKENIZERS_PARALLELISM=false
# Create necessary directories
RUN mkdir -p /app/data /app/outputs /app/models

# Copy your code into the container
COPY src/ /app/src/
COPY run_training_with_checkpoints.py /app/
COPY full_training_with_resumption.sh /app/
COPY monitor_and_resume.sh /app/

# Make scripts executable 
RUN chmod +x /app/full_training_with_resumption.sh /app/monitor_and_resume.sh

# Default command
CMD ["/app/monitor_and_resume.sh"]