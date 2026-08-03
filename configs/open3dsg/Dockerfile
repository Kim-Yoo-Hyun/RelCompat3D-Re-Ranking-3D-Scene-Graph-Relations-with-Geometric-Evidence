FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/envs/open3dsg/bin:/opt/conda/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl git build-essential libgl1 libglib2.0-0 \
    libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && conda config --set channel_priority strict \
    && conda create -y -n open3dsg python=3.9 pip \
    && conda clean -afy

SHELL ["/bin/bash", "-lc"]

RUN conda run -n open3dsg python -m pip install --no-cache-dir \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN conda run -n open3dsg python -m pip install --no-cache-dir \
    torch-cluster==1.6.3 torch-scatter==2.1.2 torch-sparse==0.6.18 \
    torch-spline-conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

RUN conda run -n open3dsg python -m pip install --no-cache-dir \
    easydict ftfy mlflow numpy==1.23.3 open3d opencv-python \
    open_clip_torch graphviz plyfile pytorch-lightning==2.1.1 scipy \
    tensorflow==2.12.0 torch-geometric==2.5.0 tqdm \
    transformers==4.46.3 sentencepiece trimesh \
    git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup.git \
    git+https://github.com/openai/CLIP.git

WORKDIR /workspace

CMD ["python", "--version"]
