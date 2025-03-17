FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Install basic utilities
RUN apt-get update && apt-get install -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Add conda to path
ENV PATH /opt/conda/bin:$PATH
 


# Initialize conda in bash
RUN conda init bash

# Create conda environment from environment.yml
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

# Activate the environment and install requirements
SHELL ["conda", "run", "-n", "dexgraspnet", "/bin/bash", "-c"]
RUN pip install mkl==2024.0
COPY thirdparty/pytorch_kinematics /tmp/thirdparty/pytorch_kinematics
RUN pip install -e /tmp/thirdparty/pytorch_kinematics

COPY thirdparty/TorchSDF /tmp/thirdparty/TorchSDF
WORKDIR /tmp/thirdparty/TorchSDF
RUN bash ./install.sh
RUN pip install six

# RUN pip install -r /tmp/requirements.txt
# RUN pip install -r /tmp/requirements@git.txt
 
 

