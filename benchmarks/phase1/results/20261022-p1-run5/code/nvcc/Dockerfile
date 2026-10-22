# Compile-only check of kernels/csrc/*.cu without a GPU: NVIDIA's nvcc for
# arm64 (sbsa) plus the CPU torch wheel's headers, the same packages the
# Colab cuda129 step installs (nvcc, cudart-dev, cccl) and the same torch.
#   docker build -t nano-nvcc benchmarks/phase1/nvcc
#   docker run --rm -v "$PWD":/repo:ro nano-nvcc
FROM ubuntu:24.04
ARG CUDA=12-8
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends ca-certificates wget python3 python3-pip python3-dev g++ \
 && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb && apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends cuda-nvcc-${CUDA} cuda-cudart-dev-${CUDA} cuda-cccl-${CUDA} \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --break-system-packages --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
CMD ["bash", "/repo/benchmarks/phase1/nvcc/compile_check.sh"]
