# -*- coding: utf-8 -*-
"""КОМУ ПРИНАДЛЕЖАТ STS/LDS РАЗДЕЛЯЕМОЙ: привязка команд к ФАЙЛУ-ИСТОЧНИКУ (nvdisasm -g).

Нужно затем, что «эпилог ходит через разделяемую» -- утверждение о ЧУЖОМ шаблонном коде cutlass,
и подтверждать его дельтой маски мало: дельта показывает, что команды ушли ВМЕСТЕ с фазой, но не
называет, ЧЕЙ это круг. Здесь команды раскладываются по заголовкам, из которых они встроены:
    tile_iterator_volta_tensor_op.h  -- ЗАПИСЬ накопителя в разделяемую площадку эпилога
    shared_load_iterator.h           -- ЧТЕНИЕ её обратно
Это ровно тот массив, на который пришлось 89.9% конфликтов банков форварда.
"""

import collections
import os
import re
import subprocess
import sys

NVDIS = "/opt/conda/miniconda3/envs/cuda128/bin/nvdisasm"
HERE = os.path.dirname(os.path.abspath(__file__))

WANT = (
    "tile_iterator_volta_tensor_op.h",
    "shared_load_iterator.h",
    "kernel_forward.h",
    "mma_from_smem.h",
    "mma_pipelined.h",
    "predicated_tile_iterator.h",
    "epilogue_base.h",
    "epilogue.h",
)


def main(cubin):
    txt = subprocess.run(
        [NVDIS, "-g", "-c", cubin], stdout=subprocess.PIPE, universal_newlines=True
    ).stdout
    cur_fn, cur_src = None, "?"
    per = collections.defaultdict(lambda: collections.Counter())
    for ln in txt.splitlines():
        m = re.match(r"\s*\.text\.(\S+):", ln)
        if m:
            cur_fn = m.group(1)
            continue
        m = re.search(r"//## File \"(.*?)\", line", ln)
        if m:
            cur_src = os.path.basename(m.group(1))
            continue
        if not cur_fn:
            continue
        for cls in ("STS", "LDS", "STG", "LDG", "HMMA"):
            if re.search(r"\b%s\b" % cls, ln):
                per[cur_fn][(cls, cur_src)] += 1
    for fn, c in per.items():
        # ВНИМАНИЕ: "Li128" встречается в ОБОИХ именах (kKeysPerBlock=128 у обоих). Различает
        # только последний целочисленный параметр перед Lb0 -- kMaxK. Ловил на этом уже.
        tag = (
            "d128 kMaxK=128 RF=TRUE "
            if "Li128ELb0" in fn
            else "d256 kMaxK=256 RF=FALSE"
            if "Li256ELb0" in fn
            else fn
        )
        print("=" * 96)
        print(tag)
        print("=" * 96)
        for cls in ("STS", "LDS", "STG"):
            rows = sorted(
                ((k[1], v) for k, v in c.items() if k[0] == cls), key=lambda x: -x[1]
            )
            tot = sum(v for _, v in rows)
            print("  %-4s всего %3d :" % (cls, tot))
            for src, v in rows:
                mark = "  <<< площадка ЭПИЛОГА" if src in WANT[:2] else ""
                print("        %-42s %4d%s" % (src, v, mark))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "m0.cubin"))
