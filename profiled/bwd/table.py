# -*- coding: utf-8 -*-
"""ОТПЕЧАТОК МАСКИ БЕЗ ЗАПУСКА: дельта счёта команд по классам в SASS двойника.

Зачем именно так. Соответствие "фаза -> код" прочитано в исходнике, а не доказано; статический
разбор по -lineinfo для этого ядра ЗАНИЖАЕТ доли (почти вся работа фаз лежит во ВСТРОЕННЫХ телах
из mma_from_smem.h / mma_pipelined.h, и строки привязываются к ИХ файлам). Поэтому маска
доказывается ДЕЛЬТОЙ СЧЁТА КОМАНД: у каждого бита заранее объявлено, какие классы обязаны упасть
и какие обязаны НЕ упасть; расхождение = разметка не туда.

ОЖИДАНИЕ ОБЪЯВЛЕНО ДО ЗАМЕРА (см. EXPECT ниже) -- иначе это подгонка, а не проверка.
"""

import os
import re
import sys

sys.path.insert(0, "./build/phase_bwd")
from count import counts  # noqa: E402

NAMES = [
    "QK",
    "SOFTMAX",
    "GRADV",
    "DOIVJ",
    "DS",
    "DSSTORE",
    "GRADQ",
    "DQFOLD",
    "GRADK",
    "KVOUT",
    "DELTA",
]
COLS = [
    "HMMA",
    "LDS",
    "STS",
    "LDG",
    "STG",
    "RED",
    "MUFU",
    "FFMA",
    "FMUL",
    "FADD",
    "SHFL",
    "BAR",
    "LDL",
    "STL",
    "ВСЕГО",
]

# ОЖИДАНИЕ: "-" обязан упасть (дельта < 0), "0" обязан НЕ упасть (дельта >= 0).
# Классы, не названные здесь, не проверяются (шум от перепланирования).
EXPECT = {
    0: {"HMMA": "-", "LDS": "-", "LDG": "-", "STS": "-"},  # QK: умножение + подача K/Q
    1: {"HMMA": "0", "MUFU": "-", "STS": "-"},  # SOFTMAX: exp2 + укладка P^T
    2: {"HMMA": "-", "LDS": "-"},  # GRADV: умножение
    3: {"HMMA": "-", "LDS": "-"},  # DOIVJ: умножение
    4: {"HMMA": "0", "MUFU": "0"},  # DS: поэлементно, не тензорно
    5: {"HMMA": "0", "STS": "-"},  # DSSTORE: только укладка
    6: {"HMMA": "-", "LDS": "-"},  # GRADQ: умножение
    7: {"HMMA": "0", "RED": "-"},  # DQFOLD: глобальная свёртка
    8: {"HMMA": "-", "LDS": "-"},  # GRADK: умножение
    9: {"HMMA": "0", "STG": "-"},  # KVOUT: выгрузка dK/dV
    10: {"HMMA": "0", "LDG": "-"},  # DELTA: предпроход по dO/O
}


def regs(log):
    t = open(log, encoding="utf-8", errors="replace").read()
    r = re.search(r"Used (\d+) registers", t)
    s = re.search(r"(\d+) bytes stack frame", t)
    sp = re.search(r"(\d+) bytes spill stores", t)
    return (
        int(r.group(1)) if r else -1,
        int(s.group(1)) if s else -1,
        int(sp.group(1)) if sp else -1,
    )


def row(d, mask):
    f = os.path.join(d, "m%d.sass" % mask)
    if not os.path.exists(f):
        return None, None
    return counts(f), regs(os.path.join(d, "m%d.log" % mask))


def main(d):
    base, (br, bs, bsp) = row(d, 0)
    hdr = (
        "%-4s %-8s" % ("бит", "фаза")
        + "".join("%7s" % c for c in COLS)
        + "%6s%6s%6s  %s" % ("рег", "кадр", "разл", "ожидание")
    )
    print(hdr)
    print("-" * len(hdr))
    print(
        "%-4s %-8s" % ("--", "БАЗА")
        + "".join("%7d" % base.get(c, 0) for c in COLS)
        + "%6d%6d%6d" % (br, bs, bsp)
    )
    bad = 0
    for b, n in enumerate(NAMES):
        c, rr = row(d, 1 << b)
        if c is None:
            print("%-4d %-8s  НЕ СОБРАЛСЯ" % (b, n))
            bad += 1
            continue
        r, s, sp = rr
        verdict = []
        for k, want in EXPECT[b].items():
            dlt = c.get(k, 0) - base.get(k, 0)
            ok = (dlt < 0) if want == "-" else (dlt >= 0)
            if not ok:
                verdict.append("%s %+d НЕ %s" % (k, dlt, want))
        note = "ОК" if not verdict else "РАСХОЖДЕНИЕ: " + "; ".join(verdict)
        if verdict:
            bad += 1
        if sp != 0 or s != 0:
            note += " | РАЗЛИВ -- маска неизмерима"
            bad += 1
        print(
            "%-4d %-8s" % (b, n)
            + "".join("%7s" % ("%+d" % (c.get(k, 0) - base.get(k, 0))) for k in COLS)
            + "%6d%6d%6d  %s" % (r, s, sp, note)
        )
    for m, lbl in ((2047, "все"),):
        c, rr = row(d, m)
        if c is None:
            continue
        r, s, sp = rr
        print(
            "%-4s %-8s" % (lbl, "-")
            + "".join("%7s" % ("%+d" % (c.get(k, 0) - base.get(k, 0))) for k in COLS)
            + "%6d%6d%6d" % (r, s, sp)
        )
    print(
        "\nитог: %s"
        % (
            "ВСЕ БИТЫ ДАЛИ ОЖИДАННЫЙ ОТПЕЧАТОК"
            if bad == 0
            else "ПРОВЕРКА НЕ ПРОЙДЕНА в %d строках" % bad
        )
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "./profiled/bwd/out/sw128")
