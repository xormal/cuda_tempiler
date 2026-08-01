# -*- coding: utf-8 -*-
"""ОТПЕЧАТОК МАСКИ В SASS: доказать, что снятая фаза ДЕЙСТВИТЕЛЬНО снялась, и только она.

Замерять доли времени по маске имеет смысл, только если маска делает в коде то, что обещает.
Проверяется тремя числами:
  (1) дельта счёта команд по классам при одиночном снятии -- совпадает ли с тем, чем фаза ЯВЛЯЕТСЯ
      (у умножения обязан пропасть весь его HMMA, у подачи -- её LDG вместе с STS, и т.д.);
  (2) АДДИТИВНОСТЬ: сумма пяти одиночных дельт против дельты «все пять». Расхождение = каскад,
      то есть снятие одной фазы унесло работу другой -- и доли перепутаны;
  (3) поточечная невязка пар e_ij = c(i|j) - c(i) - c(j) + c(0). Ноль по КЛАССАМ РАБОТЫ
      (HMMA/LDS/STS/LDG/MUFU/SHFL/BAR) означает, что каскада нет НИ У ОДНОЙ пары, и перекрытие,
      посчитанное по времени, будет перекрытием ЖЕЛЕЗА, а не следом компилятора.

    python3 table_fwd_ws.py [префикс_файлов_SASS]      # по умолчанию n (n0..n31)
"""

import os
import re
import sys
import collections

SASS = "./profiled/sass"
INSTR = re.compile(r"^\s+/\*[0-9a-f]{4,}\*/\s+(.*?);", re.M)
NAMES = {0: "gemm1", 1: "gemm2", 2: "softmax", 3: "feed", 4: "rendez"}
# КЛАССЫ РАБОТЫ -- то, чем фаза является физически. Остальное (IADD3/IMAD/LOP3/MOV) -- счётная
# обвязка, её ptxas тасует между вариантами свободно, и требовать от неё нуля бессмысленно.
WORK = ["HMMA", "LDS", "STS", "LDG", "MUFU", "SHFL", "BAR"]
KEYS = (
    ["TOTAL"] + WORK + ["FADD", "FMUL", "FFMA", "IADD3", "IMAD", "PRMT", "LOP3", "STG"]
)


def counts(path):
    c = collections.Counter()
    for m in INSTR.finditer(open(path).read()):
        ins = re.sub(r"^@!?P\d+\s+", "", m.group(1).strip())
        c[ins.split()[0].split(".")[0]] += 1
    c["TOTAL"] = sum(v for k, v in c.items() if k != "TOTAL")
    return c


def main():
    pre = sys.argv[1] if len(sys.argv) > 1 else "n"
    C = {m: counts(os.path.join(SASS, "%s%d.sass" % (pre, m))) for m in range(32)}
    hdr = lambda t: "%-16s" % t + "".join("%7s" % k for k in KEYS)
    row = lambda t, f: "%-16s" % t + "".join("%+7d" % f(k) for k in KEYS)

    print("### БАЗА -- маска 0 (боевой путь через ветвь семейства)")
    print(hdr(""))
    print("%-16s" % "" + "".join("%7d" % C[0].get(k, 0) for k in KEYS))
    print("\n### (1) ОДИНОЧНЫЕ: дельта счёта команд к маске 0")
    print(hdr("фаза"))
    for b in range(5):
        print(row(NAMES[b], lambda k, b=b: C[1 << b].get(k, 0) - C[0].get(k, 0)))
    print(
        row(
            "сумма пяти",
            lambda k: sum(C[1 << b].get(k, 0) - C[0].get(k, 0) for b in range(5)),
        )
    )
    print(row("ВСЕ ПЯТЬ (31)", lambda k: C[31].get(k, 0) - C[0].get(k, 0)))

    print(
        "\n### (3) ПАРЫ: невязка e_ij = c(i|j) - c(i) - c(j) + c(0)   (ноль = каскада НЕТ)"
    )
    print(hdr("пара"))
    bad = []
    for i in range(5):
        for j in range(i + 1, 5):
            m = (1 << i) | (1 << j)
            f = lambda k, m=m, i=i, j=j: (
                C[m].get(k, 0)
                - C[1 << i].get(k, 0)
                - C[1 << j].get(k, 0)
                + C[0].get(k, 0)
            )
            print(row(NAMES[i] + "+" + NAMES[j], f))
            for k in WORK:
                if f(k):
                    bad.append((NAMES[i] + "+" + NAMES[j], k, f(k)))
    f5 = lambda k: (
        C[31].get(k, 0)
        - sum(C[1 << b].get(k, 0) - C[0].get(k, 0) for b in range(5))
        - C[0].get(k, 0)
    )
    print(row("ВСЕ ПЯТЬ", f5))
    for k in WORK:
        if f5(k):
            bad.append(("все пять", k, f5(k)))

    print("\n### ВЕРДИКТ")
    if bad:
        print("  КАСКАД ЕСТЬ -- доли перепутаны, нужен насильный потребитель:")
        for t, k, v in bad:
            print("    %-16s %s %+d" % (t, k, v))
    else:
        print(
            "  По КЛАССАМ РАБОТЫ (%s) невязка РОВНО НОЛЬ у всех десяти пар и у «всех пяти»."
            % "/".join(WORK)
        )
        print(
            "  Значит снятие фазы не уносит работу соседней, и перекрытие, которое даст замер"
        )
        print("  времени, будет перекрытием ЖЕЛЕЗА, а не следом компилятора.")
    resid = max(abs(f5(k)) for k in KEYS)
    print(
        "  Остаток на счётной обвязке (IADD3/IMAD/LOP3/...): не более %d команд из %d (%.2f %%)."
        % (resid, C[0]["TOTAL"], 100.0 * resid / C[0]["TOTAL"])
    )


if __name__ == "__main__":
    main()
