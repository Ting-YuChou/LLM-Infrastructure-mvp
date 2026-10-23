#!/usr/bin/env bash
# nvcc -c every kernels/csrc/*.cu for sm_80 with torch.utils.cpp_extension's flags.
set -uo pipefail
NVCC=$(ls -d /usr/local/cuda-*/bin/nvcc | head -1)
TORCH=$(python3 -c "import os, torch; print(os.path.dirname(torch.__file__))" 2>/dev/null)
PYINC=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])" 2>/dev/null)
# The CPU wheel ships c10/cuda's headers but not this generated one.
STUB=/tmp/stub
mkdir -p "${STUB}/c10/cuda/impl"
echo '#define C10_CUDA_BUILD_SHARED_LIBS' > "${STUB}/c10/cuda/impl/cuda_cmake_macros.h"
rc=0
for src in /repo/src/serving/nano_vllm/kernels/csrc/*.cu; do
  name=$(basename "${src}" .cu)
  if "${NVCC}" -c "${src}" -o "/tmp/${name}.o" -std=c++17 -O3 --use_fast_math -lineinfo \
      -gencode=arch=compute_80,code=sm_80 --compiler-options '-fPIC' \
      -DTORCH_EXTENSION_NAME="nano_vllm_${name}" -DTORCH_API_INCLUDE_EXTENSION_H \
      -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
      -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr \
      -isystem "${STUB}" -isystem "${TORCH}/include" -isystem "${TORCH}/include/torch/csrc/api/include" -isystem "${PYINC}" \
      -Xptxas -v 2> "/tmp/${name}.log"; then
    echo "OK   ${name}.cu"
    grep -E "registers|spill" "/tmp/${name}.log" | grep -E "spill stores [1-9]|bytes spill" | grep -v " 0 bytes spill stores, 0 bytes spill loads" | head -5
  else
    echo "FAIL ${name}.cu"; grep -E "error|warning" "/tmp/${name}.log" | head -30; rc=1
  fi
done
exit "${rc}"
