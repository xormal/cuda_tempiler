# -*- coding: utf-8 -*-
"""ОТПЕЧАТОК МАСКИ В SASS: дельта счёта команд по классам для каждой маски. Ядер НЕ ЗАПУСКАЕТ.

Зачем это, если долю фазы даёт только ВРЕМЯ. Затем, что перед замером времени надо доказать, что
маска сняла ИМЕННО СВОЮ фазу и не унесла чужую. Доказательство здесь -- дельта счёта команд по
классам: у каждой фазы есть класс-подпись (gemm1/gemm2 -> HMMA, softmax -> MUFU.EX2 и FMNMX,
эпилог -> STG выхода, pstore -> STS+F2F). Если снятие фазы обнуляет ЧУЖУЮ подпись -- каскад, и
время мерить нельзя.

    python3 table.py            # собрать всё семейство и напечатать таблицу
    python3 table.py --only 0,2 # только перечисленные маски
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CUOBJ = "/opt/conda/miniconda3/envs/cuda128/bin/cuobjdump"

# инстанцирования из inst_fwd.cu; в имени ядра kMaxK кодируется как Li128/Li256
KERNELS = {
    "Li128ELb0": "d128  kMaxK=128<=kKeysPerBlock=128 -> kKeepOutputInRF=TRUE  (эпилог ОДИН на блок запросов)",
    "Li256ELb0": "d256  kMaxK=256> kKeysPerBlock=128 -> kKeepOutputInRF=FALSE (эпилог на КАЖДОМ блоке ключей)",
}

CLASSES = [
    ("HMMA", r"\bHMMA\b"),
    ("LDG", r"\bLDG\b"),
    ("STG_O", r"\bSTG"),
    ("ST.E", r"\bST\.E"),
    ("LDS", r"\bLDS\b"),
    ("STS", r"\bSTS\b"),
    ("LDL+STL", r"\bLDL\b|\bSTL\b"),
    ("EX2", r"MUFU\.EX2"),
    ("MUFU", r"\bMUFU\b"),
    ("RED/ATOM", r"\bRED\b|\bATOM"),
    ("FMNMX", r"\bFMNMX\b"),
    ("FFMA", r"\bFFMA\b"),
    ("FMUL", r"\bFMUL\b"),
    ("F2F", r"\bF2F\b"),
    ("PRMT", r"\bPRMT\b"),
    ("BAR", r"\bBAR\.SYNC\b"),
    ("ВСЕГО", r"."),
]

NAMES = {0: "prolog", 1: "gemm1", 2: "softmax", 3: "pstore", 4: "gemm2", 5: "epilog"}
FAMILY = (
    [(0, "база(0)")]
    + [(1 << b, "%s(%d)" % (NAMES[b], 1 << b)) for b in sorted(NAMES)]
    + [
        (2 | 16, "gemm1+gemm2(18)"),
        (4 | 32, "softmax+epilog(36)"),
        (63, "ВСЕ(63)"),
    ]
)


def build(mask, tag):
    cub = os.path.join(OUT, "m%d.cubin" % mask)
    r = subprocess.run(
        [os.path.join(HERE, "build.sh"), cub, str(mask), "twin"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    if r.returncode != 0:
        sys.stdout.write(r.stdout[-4000:])
        raise SystemExit("сборка провалилась, маска=%d" % mask)
    res = {}
    cur = None
    for ln in r.stdout.splitlines():
        m = re.search(r"entry function '(\S+)'", ln)
        if m:
            cur = next((k for k in KERNELS if k in m.group(1)), None)
        if cur:
            m = re.search(r"Used (\d+) registers", ln)
            if m:
                res.setdefault(cur, {})["рег"] = int(m.group(1))
            m = re.search(r"(\d+) bytes stack frame", ln)
            if m:
                res.setdefault(cur, {})["кадр"] = int(m.group(1))
    return cub, res


def sass_by_kernel(cubin):
    txt = subprocess.run(
        [CUOBJ, "-sass", cubin], stdout=subprocess.PIPE, universal_newlines=True
    ).stdout
    cur, res = None, collections.defaultdict(list)
    for ln in txt.splitlines():
        m = re.search(r"Function : (\S+)", ln)
        if m:
            cur = next((k for k in KERNELS if k in m.group(1)), None)
            continue
        if cur and re.match(r"\s*/\*[0-9a-f]{4,}\*/", ln):
            res[cur].append(ln)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    fam = FAMILY
    if a.only:
        keep = {int(x) for x in a.only.split(",")}
        fam = [f for f in FAMILY if f[0] in keep]

    data = {}
    for mask, label in fam:
        cub, ru = build(mask, label)
        sk = sass_by_kernel(cub)
        data[label] = {
            k: dict(
                {n: sum(1 for l in v if re.search(rx, l)) for n, rx in CLASSES},
                **ru.get(k, {}),
            )
            for k, v in sk.items()
        }
        print("собрано mask=%-3d %s" % (mask, label))
    json.dump(
        data, open(os.path.join(OUT, "counts.json"), "w"), ensure_ascii=False, indent=1
    )

    cols = [n for n, _ in CLASSES] + ["рег", "кадр"]
    for k, desc in KERNELS.items():
        print()
        print("=" * 150)
        print("ЯДРО  %s" % desc)
        print("=" * 150)
        hdr = "%-18s" % "снято" + "".join("%9s" % c for c in cols)
        print(hdr)
        print("-" * len(hdr))
        base = data["база(0)"][k]
        for mask, label in fam:
            d = data[label][k]
            row = "%-18s" % label
            for c in cols:
                if label == "база(0)":
                    row += "%9d" % d.get(c, -1)
                else:
                    dv = d.get(c, 0) - base.get(c, 0)
                    row += "%9s" % ("%+d" % dv if dv else "0")
            print(row)


if __name__ == "__main__":
    main()
