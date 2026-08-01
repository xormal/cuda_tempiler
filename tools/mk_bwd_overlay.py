# -*- coding: utf-8 -*-
"""ОДНОРАЗОВЫЙ извлекатель: разница (боевой файл -> размеченная копия) превращается в НАЛОЖЕНИЕ
из ЯКОРНЫХ правок (текст, а не номера строк). Каждый якорь проверяется на ЕДИНСТВЕННОСТЬ в боевом
файле; неоднозначный якорь расширяется соседним контекстом до единственности.

Запускать вручную при переразметке. Боевое дерево ТОЛЬКО ЧИТАЕТСЯ.
"""

import difflib
import hashlib
import json
import os
import sys

PROD = "../VLLM_fa2/solutions/fa2_sm70_cutlass_grade"
MARK = "./.cache"

PAIRS = [
    ("fa2_src/fmha_kernel/kernel_backward.h", "kernel_backward.h"),
    ("fa2_src/fmha_kernel/fused_qk_gradk.h", "fused_qk_gradk.h"),
]


def build(rel, markname):
    a = open(os.path.join(PROD, rel), encoding="utf-8").read().split("\n")
    b = open(os.path.join(MARK, markname), encoding="utf-8").read().split("\n")
    prod_text = "\n".join(a)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
    # СЛИЯНИЕ близких кусков: два якоря не должны ПЕРЕСЕКАТЬСЯ по тексту, иначе после применения
    # первого второй уже не найдётся (и ворота дрейфа упадут на ровном месте).
    GAP = 8
    merged = []
    for tag, i1, i2, j1, j2 in ops:
        if merged and i1 - merged[-1][2] <= GAP:
            p = merged[-1]
            merged[-1] = ("mix", p[1], i2, p[3], j2)
        else:
            merged.append((tag, i1, i2, j1, j2))
    edits = []
    for tag, i1, i2, j1, j2 in merged:
        ctx = 3
        while True:
            lo = max(0, i1 - ctx)
            hi = min(len(a), i2 + ctx)
            # контекст с той же стороны в b
            blo = j1 - (i1 - lo)
            bhi = j2 + (hi - i2)
            anchor = "\n".join(a[lo:hi])
            if prod_text.count(anchor) == 1:
                break
            ctx += 3
            if ctx > 200:
                raise SystemExit(
                    "НЕ УДАЛОСЬ сделать якорь единственным: %s строка %d" % (rel, i1)
                )
        repl = "\n".join(b[blo:bhi])
        edits.append({"anchor": anchor, "replace": repl})
    return {
        "file": rel,
        "md5": hashlib.md5(open(os.path.join(PROD, rel), "rb").read()).hexdigest(),
        "edits": edits,
    }


def main():
    out = [build(rel, mk) for rel, mk in PAIRS]
    json.dump(
        out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1
    )
    for f in out:
        print(
            "%-46s md5=%s  правок=%d  строк-якорей=%d"
            % (
                os.path.basename(f["file"]),
                f["md5"],
                len(f["edits"]),
                sum(e["anchor"].count("\n") + 1 for e in f["edits"]),
            )
        )


if __name__ == "__main__":
    main()
