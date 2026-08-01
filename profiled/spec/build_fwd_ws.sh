#!/bin/bash
# Сборка ОДНОГО варианта volta_fwd_ws в cubin+SASS. Только процессор: ядро не запускается.
#   $1 = каталог с volta_fwd_ws.cuh (двойник ИЛИ боевой csrc)
#   $2 = тег выходных файлов
#   $3..= доп. определения nvcc (например -DWS_D=64)
set -e
R=../VLLM_fa2/solutions/fa2_sm70_cutlass_grade
P=./profiled
NVCC=/opt/conda/miniconda3/envs/cuda128/bin/nvcc
CUOBJ=/opt/conda/miniconda3/envs/cuda128/bin/cuobjdump
INC=$1; TAG=$2; shift 2
mkdir -p $P/sass
$NVCC -arch=sm_70 -O3 -cubin -std=c++17 \
  -I"$INC" -I$R/fa2_sm70/csrc -I$R/fa2_src/fmha_kernel \
  -ccbin /usr/bin/g++ -Xptxas -v "$@" \
  -o $P/sass/$TAG.cubin $P/spec/drv_fwd_ws.cu 2> $P/sass/$TAG.ptxas
$CUOBJ -sass $P/sass/$TAG.cubin > $P/sass/$TAG.sass
