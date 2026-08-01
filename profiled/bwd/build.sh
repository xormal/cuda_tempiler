#!/bin/bash
# Сборка ОДНОЙ инстанциации бэкварда. Пишет только в ./profiled.
#   $1 = путь выхода без расширения
#   $2 = БОЕВОЕ|ДВОЙНИК  (BASE|TWIN) -- откуда берётся kernel_backward.h
#   $3 = "BI,BJ,MK,ALIGNED"
#   $4.. = доп. -D (маска, пломба, -lineinfo)
# Боевое дерево ТОЛЬКО в -I (чтение). Порядок -I решает: каталог двойника идёт ПЕРВЫМ.
set -e
CUDA_HOME=/opt/conda/miniconda3/envs/cuda128
TI=/opt/conda/miniconda3/envs/py311/lib/python3.11/site-packages/torch/include
REPO=../VLLM_fa2/solutions/fa2_sm70_cutlass_grade
TWININC=./profiled/bwd/inc
OUT=$1; MODE=$2; TILE=$3; shift 3
IFS=, read BI BJ MK AL <<< "$TILE"
mkdir -p "$(dirname "$OUT")"
case "$MODE" in
  TWIN) FIRST="-I $TWININC" ;;
  BASE) FIRST="" ;;
  *) echo "режим: BASE|TWIN"; exit 1 ;;
esac
cat > "$OUT.cu" <<CU
#define HAS_PYTORCH 1
#include <kernel_backward.h>
using PH_AK = AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t,
    /*kIsAligned*/ true, /*kApplyDropout*/ false, /*kPreload*/ false,
    /*kBlockSizeI*/ $BI, /*kBlockSizeJ*/ $BJ, /*kMaxK*/ $MK,
    /*kKeysQueriesAlignedToBlockSize*/ $AL, /*kEnableSplitKeys*/ true>;
template __global__ void attention_kernel_backward_batched_impl<PH_AK>(PH_AK::Params);
CU
$CUDA_HOME/bin/nvcc -arch=sm_70 -cubin -o "$OUT.cubin" "$OUT.cu" \
  $FIRST \
  -I $REPO/fa2_src/cutlass/include -I $REPO/fa2_src/cutlass/tools/util/include \
  -I $REPO/fa2_src/fmha_kernel \
  -isystem $TI -isystem $TI/torch/csrc/api/include -isystem $CUDA_HOME/include \
  -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
  --expt-relaxed-constexpr -O3 -std=c++17 --use_fast_math -DHAS_PYTORCH \
  -DFMHA_BWD_R7 -DFMHA_BWD_COAL_FO -Xptxas -v "$@" 2> "$OUT.log"
$CUDA_HOME/bin/cuobjdump -sass "$OUT.cubin" > "$OUT.sass"
grep -E "Used|stack frame|spill" "$OUT.log" | head -3
