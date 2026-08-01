#!/bin/bash
# Сборка cubin ЦЕЛИ. Ядер НЕ ЗАПУСКАЕТ -- только nvcc/cuobjdump на процессоре.
#   build.sh <out.cubin> <mask> [prod|twin] [доп. флаги nvcc]
# «prod» собирает БОЕВОЙ заголовок как есть (двойник не подключается вовсе) -- это база сравнения.
# «twin» подставляет КАНОНИЧЕСКУЮ порождённую копию (tools/twin.py) через -I ПЕРЕД боевым
# каталогом; «twin2» -- копию второго, независимого генератора (fwdc/gen_twin.py) для сверки.
set -euo pipefail
OUT="$1"; MASK="$2"; WHICH="${3:-twin}"; shift 3 || true
HERE="$(cd "$(dirname "$0")" && pwd)"
PROD=../VLLM_fa2/solutions/fa2_sm70_cutlass_grade
NVCC=/opt/conda/miniconda3/envs/cuda128/bin/nvcc
INC=""
CANON=./profiled/fwd_cutlass/fa2_src/fmha_kernel   # порождает tools/twin.py
if [ "$WHICH" = "twin" ]; then INC="-I $CANON"; fi
if [ "$WHICH" = "twin2" ]; then INC="-I $HERE/twin/fmha_kernel"; fi   # независимый генератор
# ВОРОТА ДРЕЙФА ПЕРЕД КАЖДОЙ СБОРКОЙ: боевое дерево могло уехать ПОСЛЕ порождения двойника.
if [ "$WHICH" != "prod" ]; then
  python3 ./tools/twin.py check \
    --overlay ./profiled/overlays/fwd_cutlass.json >/dev/null || {
      echo "ВОРОТА ДРЕЙФА НЕ ПРОЙДЕНЫ -- сборка двойника отменена" >&2; exit 3; }
fi
exec "$NVCC" -arch=sm_70 -O3 -std=c++17 -cubin -lineinfo -Xptxas -v \
  -ccbin /usr/bin/g++ \
  $INC \
  -I "$PROD/fa2_src/cutlass/include" \
  -I "$PROD/fa2_src/fmha_kernel" \
  -DFMHA_STRIP_MASK="${MASK}u" \
  "$@" \
  -o "$OUT" "$HERE/inst_fwd.cu"
