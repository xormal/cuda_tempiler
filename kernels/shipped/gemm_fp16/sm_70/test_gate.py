#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ГЕЙТ КОРРЕКТНОСТИ, ВОСПРОИЗВОДИМЫЙ В ХОСТ-ДЕРЕВЕ.

«Прошло у нас» не равно «пройдёт у них»: другой компилятор, другая версия фреймворка, другой
каталог сборки.  Поэтому гейт едет ВМЕСТЕ С ЯДРОМ.

ЧТО ЭТОТ ФАЙЛ ДЕЛАЕТ БЕЗ КАРТЫ (и делает всегда):
  * проверяет ПРЕДИКАТ поставки на боевых формах -- какие покрыты, какие уходят в откат;
  * проверяет, что таблица инстанциаций (`dispatch.inc`) НЕ РАЗЪЕХАЛАСЬ с `select.py`;
  * печатает процедуру заполнения `oracle.json` (сам файл намеренно не выдуман).

ЧТО ТРЕБУЕТ КАРТЫ (и честно отказывается без неё):
  * сверку значений с эталоном ПОВЫШЕННОЙ точности и покрытие атомарным штампом.

ЛОВУШКА, КОТОРУЮ ЗДЕСЬ НЕЛЬЗЯ ПОВТОРИТЬ: сверять с умножением в ПОЛОВИННОЙ точности нельзя --
оно возвращает половинную и переполняет ВЫХОД при исправном одинарном накопителе, давая
ЛОЖНЫЙ ОТРИЦАТЕЛЬНЫЙ.  Эталон -- одинарная точность и выше.

ЗАПУСК:
    python3 test_gate.py                       # проверки без карты
    python3 test_gate.py --host-tree /путь     # то же + сверка сборки в хост-дереве
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Боевые формы линейной части (M задаётся отдельно): имя, N, K
SHAPES = [
    ("q_proj", 4096, 3840),
    ("k,v_proj", 2048, 3840),
    ("o_proj", 3840, 4096),
    ("gate,up", 15360, 3840),
    ("down_proj", 3840, 15360),
]
MS = [1, 8, 32, 128, 512, 2048, 4096]


def _select():
    spec = importlib.util.spec_from_file_location("tempo_select", os.path.join(HERE, "select.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check_predicate():
    sel = _select()
    covered = uncovered = 0
    rows = []
    for name, N, K in SHAPES:
        for M in MS:
            r = sel.select("gemm", M, N, K)
            if r:
                covered += 1
            else:
                uncovered += 1
                rows.append("  ОТКАТ  %-10s M=%-5d N=%-6d K=%-6d" % (name, M, N, K))
    print("ПРЕДИКАТ ПОСТАВКИ: покрыто %d форм, уходит в откат %d" % (covered, uncovered))
    for r in rows:
        print(r)
    if uncovered == 0:
        print("  ВНИМАНИЕ: покрыто ВСЁ -- проверьте предикат, полное покрытие подозрительно")
    return True


def check_dispatch_matches_select():
    """Таблица инстанциаций ПОРОЖДАЕТСЯ из select.py.  Здесь -- что она не отстала."""
    inc = os.path.join(HERE, "dispatch.inc")
    if not os.path.exists(inc):
        print("dispatch.inc отсутствует -- нечего сверять")
        return False
    text = open(inc, encoding="utf-8").read()
    sel = _select()
    missing = []
    for m_max, tag, BM, BN, BK, WM, WN, MINB in sel.LADDER:
        if ("<%d,%d,%d,%d,%d," % (BM, BN, BK, WM, WN)) not in text:
            missing.append(tag)
    for tag, BM, BN, BK, WM, WN, MINB in (sel.WIDE, sel.NARROW):
        if ("<%d,%d,%d,%d,%d," % (BM, BN, BK, WM, WN)) not in text:
            missing.append(tag)
    if missing:
        print("РАСХОЖДЕНИЕ select.py <-> dispatch.inc: нет инстанциаций для " + ", ".join(missing))
        print("  лечение: python3 gen_dispatch.py > dispatch.inc")
        return False
    if "TEMPO_FALLBACK" not in text:
        print("В dispatch.inc НЕТ отката -- поставка сломает сервер на первой нестандартной форме")
        return False
    print("dispatch.inc согласован с select.py, откат на месте")
    return True


def check_manifest():
    import json

    p = os.path.join(HERE, "manifest.json")
    d = json.load(open(p, encoding="utf-8"))
    need = ("op", "arch", "predicate", "card", "gate", "source_md5", "license")
    miss = [k for k in need if k not in d]
    if miss:
        print("МАНИФЕСТ НЕПОЛОН: нет " + ", ".join(miss))
        return False
    card = d["card"]
    ok = True
    if isinstance(card, dict) and card.get("foreign_procs", 0) not in (0, "0"):
        print("МАНИФЕСТ: замер снят при чужих процессах на карте -- числа недействительны")
        ok = False
    print("манифест: полон; карта %s" % (card if isinstance(card, str) else card.get("index")))
    return ok


def oracle_procedure():
    print()
    print("ORACLE.JSON НЕ ЗАПОЛНЕН НАМЕРЕННО. Файл-эталон, которому нельзя верить, опаснее")
    print("отсутствующего. Процедура заполнения (нужна свободная карта):")
    print("  1. закрепить частоты, убедиться, что чужих процессов 0 (иначе замер недействителен);")
    print("  2. на каждой форме приёмки посчитать выход НАШИМ ядром;")
    print("  3. сверить с эталоном ОДИНАРНОЙ точности и выше (НЕ с половинной -- ложный отрицательный);")
    print("  4. отдельно проверить ПОКРЫТИЕ атомарным штампом: непокрытых ячеек 0, повторных 0;")
    print("  5. записать хэш выхода и допуск relL2 по каждой форме;")
    print("  6. снять частоты (-rgc) и записать число чужих процессов ПОСЛЕ.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-tree", help="корень хост-дерева, где ядро собирается")
    a = ap.parse_args()
    ok = True
    ok &= check_predicate()
    ok &= check_dispatch_matches_select()
    ok &= check_manifest()
    if a.host_tree:
        if not os.path.isdir(a.host_tree):
            print("хост-дерево %r не найдено" % a.host_tree)
            ok = False
        else:
            print("хост-дерево: %s" % a.host_tree)
            print("  сборка и сверка значений требуют СВОБОДНОЙ карты; замер с соседом недействителен")
    oracle_procedure()
    print()
    print("ИТОГ ГЕЙТА БЕЗ КАРТЫ: " + ("ПРОЙДЕН" if ok else "НЕ ПРОЙДЕН"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
