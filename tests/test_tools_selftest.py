#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""САМОПРОВЕРКИ СУЩЕСТВУЮЩИХ ИНСТРУМЕНТОВ -- ФАЛЬСИФИКАТОР САМОЙ МИГРАЦИИ.

Правило, без которого перенос двадцати тысяч строк убивает проект:

    ШАГ, МЕНЯЮЩИЙ ОДНОВРЕМЕННО И МЕСТО ФАЙЛА, И ЕГО СОДЕРЖИМОЕ, ЗАПРЕЩЁН.
    Переносим файл целиком -> гоняем его самопроверку -> и только потом правим.

Инструменты этого дерева проверены ЯКОРЯМИ (сверка с ручным счётом, со счётчиком прибора, с
перебором).  Их нельзя переписывать: они и есть та часть продукта, которой можно верить.
Поэтому они встроены в конвейер ОБЁРТКОЙ, а этот файл каждый раз доказывает, что обёртка
ничего не сломала.

ЗАПУСК:  python3 tests/test_tools_selftest.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TOOLS = os.path.join(ROOT, "tools")


def _env_module():
    import importlib.util

    p = os.path.join(ROOT, "tempo", "cli", "env.py")
    spec = importlib.util.spec_from_file_location("tempo_env", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# (имя, аргументы, нужен ли интерпретатор >= 3.12, нужна ли карта, якорь в выводе)
CASES = [
    ("tempo/cli/env.py",         ["--selftest"], False, False, "ИТОГ: 8/8"),
    ("tools/timeit.py",          ["--selftest"], False, False, None),
    ("tools/calib.py",           ["selftest"],   False, False, None),
    ("tools/bankform.py",        ["selftest"],   False, False, None),
    ("tools/cc_ab.py",           ["selftest"],   False, False, None),
    ("tools/padsweep.py",        ["selftest"],   False, False, None),
    ("tools/bits_table.py",      ["selftest"],   False, False, None),
    ("tools/smem_lint.py",       ["--selftest"], False, False, "закон ОК"),
    ("tools/residency.py",       ["--selftest"], True,  False, None),
    ("tempo/core/graph/flows.py",      [],       False, False, None),
    ("tempo/core/graph/graphs.py",     [],       False, False, None),
    ("tempo/core/graph/partitions.py", [],       False, False, None),
]

# Требуют СВОБОДНОЙ КАРТЫ и (часть) прав на прибор. В обычном прогоне пропускаются с явной
# отметкой: молча пропущенная проверка -- это ложное «зелено».
CARD_CASES = [
    ("tools/ncu.py", ["--selftest"], "якорь прибора: собирает расширение боевого дерева и меряет"),
    ("tools/tempo.py", ["--selftest"], "643 точки стенда; требует собранного build/probe"),
]


def run(rel, args, py):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        return None, "файла нет"
    p = subprocess.run([py, path] + args, capture_output=True, text=True, cwd=ROOT, timeout=1800)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="без тяжёлых (calib, padsweep, cc_ab)")
    ap.add_argument("--with-card", action="store_true", help="гонять и то, что требует карты")
    a = ap.parse_args()

    env = _env_module()
    py = env.python_vllm() or sys.executable
    py312 = env.python_312() or py
    heavy = {"tools/calib.py", "tools/padsweep.py", "tools/cc_ab.py", "tools/bits_table.py"}

    print("САМОПРОВЕРКИ ИНСТРУМЕНТОВ (фальсификатор миграции)")
    print("  интерпретатор:      %s" % py)
    print("  интерпретатор 3.12: %s" % py312)
    print()
    ok = bad = skipped = 0
    for rel, args, need312, need_card, anchor in CASES:
        if a.quick and rel in heavy:
            print("  ПРОПУЩЕН %-34s (--quick)" % rel)
            skipped += 1
            continue
        rc, out = run(rel, args, py312 if need312 else py)
        if rc is None:
            print("  НЕТ ФАЙЛА %-32s" % rel)
            bad += 1
            continue
        good = rc == 0 and (anchor is None or anchor in out)
        print("  %-8s %-34s rc=%s%s" % ("ПРОЙДЕН" if good else "ПАДЁТ", rel, rc,
                                        "" if anchor is None else ("  якорь %r" % anchor)))
        if good:
            ok += 1
        else:
            bad += 1
            for ln in out.strip().split("\n")[-6:]:
                print("           | " + ln)

    for rel, args, why in CARD_CASES:
        if not a.with_card:
            print("  ПРОПУЩЕН %-34s ТРЕБУЕТ СВОБОДНОЙ КАРТЫ: %s" % (rel, why))
            skipped += 1
            continue
        rc, out = run(rel, args, py)
        good = rc == 0
        print("  %-8s %-34s rc=%s" % ("ПРОЙДЕН" if good else "ПАДЁТ", rel, rc))
        if good:
            ok += 1
        else:
            bad += 1
            for ln in (out or "").strip().split("\n")[-8:]:
                print("           | " + ln)

    print()
    print("ИТОГ: пройдено %d, упало %d, пропущено %d" % (ok, bad, skipped))
    if skipped:
        print("  ПРОПУЩЕННОЕ НЕ СЧИТАЕТСЯ ПРОЙДЕННЫМ. Пропуск напечатан с причиной намеренно.")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
