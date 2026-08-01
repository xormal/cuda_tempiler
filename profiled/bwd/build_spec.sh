#!/bin/bash
# Обёртка под phaseprof для БЭКВАРДА.  $1 = путь выхода .cubin, $2 = маска, $3.. = extra
# ВАЖНО: зовёт profiled/bwd/build.sh в режиме TWIN -- каталог двойника идёт ПЕРВЫМ в -I.
# Прежняя обёртка (./build/phase_bwd/build_spec.sh) звала сборку БЕЗ -I двойника,
# из-за чего <kernel_backward.h> резолвился в БОЕВОЙ файл, маска не действовала ни на что, а
# разложение выходило «все фазы бесплатны» -- ровно та мина, что описана в наряде.
set -e
OUT=${1%.cubin}; MASK=$2; shift 2
exec ./profiled/bwd/build.sh "$OUT" TWIN 128,64,128,true \
     -DFMHA_STRIP_MASK=$MASK -DFMHA_BWD_PHASE_SEAL=1 -lineinfo "$@"
