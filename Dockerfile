# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3-opencv \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Add conda to path and initialize for bash
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda init bash && \
    echo "conda activate protovit" >> ~/.bashrc

# Clone the repository
RUN git clone https://github.com/Henrymachiyu/ProtoViT.git .

# Create necessary directories
RUN mkdir -p /app/datasets

# Create conda environment and install core dependencies
RUN conda create -n protovit python=3.8 -y && \
    conda install -n protovit -y -c pytorch -c nvidia \
    pytorch==2.0.0 \
    torchvision \
    torchaudio \
    pytorch-cuda=11.7 \
    numpy \
    opencv \
    matplotlib \
    scikit-learn \
    tqdm \
    && conda run -n protovit pip install \
    timm==0.4.12 \
    Augmentor \
    cleverhans \
    transformers \
    easydict \
    gpustat

# Copy datasets directory (you need to have these files in the build context)
COPY datasets/pins/ /app/datasets/pins

# Set up conda environment activation
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Command to run when container starts
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["bash", "-l"] 