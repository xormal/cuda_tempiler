#!/bin/bash
# Свип по маскам для ДВОЙНИКА. Для каждой маски печатает регистры/кадр/разлив -- на ядре,
# стоящем РОВНО на 255 регистрах, это условие работоспособности маски, а не формальность.
#   $1 = каталог выхода, $2 = "BI,BJ,MK,AL", $3.. = список масок (десятичных)
set -e
cd ./profiled/bwd
D=$1; TILE=$2; shift 2
mkdir -p "$D"
for M in "$@"; do
  ./build.sh "$D/m$M" TWIN "$TILE" -DFMHA_STRIP_MASK=$M -DFMHA_BWD_PHASE_SEAL=1 -lineinfo \
    > /dev/null 2>&1 || { echo "m$M: НЕ СОБРАЛСЯ"; continue; }
  printf "m%-6s " "$M"
  grep -E "Used [0-9]+ registers|stack frame" "$D/m$M.log" | tr '\n' ' '
  echo
done
