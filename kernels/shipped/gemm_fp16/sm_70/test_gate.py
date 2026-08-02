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
  * `--with-card`: СОБРАТЬ И ЗАПУСТИТЬ отгружаемый путь (`ship_probe.cu` + `launch.cu` +
    `dispatch.inc` + `kernel.cuh`, НИ ОДНОЙ строки испытательной обвязки) и сверить значения;
  * сверку значений с эталоном ПОВЫШЕННОЙ точности и покрытие атомарным штампом.

ПОЧЕМУ ЗАПУСК ПОСТАВКИ ПРИШЛОСЬ ДОБАВИТЬ ОТДЕЛЬНО. Гейт без карты сверял `dispatch.inc` с
`select.py` и полноту паспорта -- и ЗЕЛЕНЕЛ при поставке, которая не исполнялась вовсе:
ядро объявляет `extern __shared__`, а `launch.cuh` пускал его с НУЛЁМ динамической
разделяемой (`an illegal memory access` на ПЕРВОЙ боевой форме, контекст мёртв), плюс
`tempo::gen::launch` был объявлен и нигде не определён.  Числа отчёта при этом
существовали -- их снимала испытательная обвязка, которая инстанцирует ядра сама.
ПРАВИЛО: гейт, который не ЗАПУСКАЕТ отгружаемое, проверяет не поставку.

Свободная карта для `--with-card` НЕ нужна: здесь сверяются ЗНАЧЕНИЯ, а не время, и сосед
на карте значения не искажает.  Секундомера в этом гейте нет намеренно.

ЛОВУШКА, КОТОРУЮ ЗДЕСЬ НЕЛЬЗЯ ПОВТОРИТЬ: сверять с умножением в ПОЛОВИННОЙ точности нельзя --
оно возвращает половинную и переполняет ВЫХОД при исправном одинарном накопителе, давая
ЛОЖНЫЙ ОТРИЦАТЕЛЬНЫЙ.  Эталон -- одинарная точность и выше.

ЗАПУСК:
    python3 test_gate.py                       # проверки без карты
    python3 test_gate.py --host-tree /путь     # то же + сверка сборки в хост-дереве
    python3 test_gate.py --with-card           # то же + СБОРКА И ЗАПУСК поставляемого пути
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# ФАЙЛЫ, ИЗ КОТОРЫХ СОБИРАЕТСЯ ПОСТАВЛЯЕМЫЙ ПУТЬ. Список закрыт намеренно: он и есть
# определение «поставки». Появится здесь файл обвязки -- гейт перестанет проверять поставку.
SHIP_SOURCES = ("kernel.cuh", "launch.cuh", "launch.cu", "dispatch.inc")

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
    spec = importlib.util.spec_from_file_location(
        "tempo_select", os.path.join(HERE, "select.py")
    )
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
    print(
        "ПРЕДИКАТ ПОСТАВКИ: покрыто %d форм, уходит в откат %d" % (covered, uncovered)
    )
    for r in rows:
        print(r)
    if uncovered == 0:
        print(
            "  ВНИМАНИЕ: покрыто ВСЁ -- проверьте предикат, полное покрытие подозрительно"
        )
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
    rows = [tuple(r[1:]) for r in sel.LADDER] + [sel.WIDE, sel.NARROW]
    for tag, BM, BN, BK, WM, WN, MINB, FPREF in rows:
        # СВЕРЯЕТСЯ ВСЯ ИНСТАНЦИАЦИЯ, а не только геометрия.  Прежняя редакция смотрела
        # только на <BM,BN,BK,WM,WN,> -- и потому НЕ ЗАМЕТИЛА, что отгружался FPREF=2 там,
        # где мерили FPREF=1.  Гейт, сверяющий часть подписи, пропускает подмену остатка.
        want = "<%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d>" % (
            BM,
            BN,
            BK,
            WM,
            WN,
            sel.COMMON["STAGES"],
            sel.COMMON["GSTAGE"],
            FPREF,
            sel.COMMON["GROUP"],
            sel.COMMON["EPI"],
            sel.COMMON["SWZ"],
            "true" if sel.COMMON["PRED"] else "false",
            MINB,
        )
        if want not in text:
            missing.append("%s (ожидалось %s)" % (tag, want))
    if missing:
        print(
            "РАСХОЖДЕНИЕ select.py <-> dispatch.inc: нет инстанциаций для "
            + ", ".join(missing)
        )
        print("  лечение: python3 gen_dispatch.py > dispatch.inc")
        return False
    if "TEMPO_FALLBACK" not in text:
        print(
            "В dispatch.inc НЕТ отката -- поставка сломает сервер на первой нестандартной форме"
        )
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
        print(
            "МАНИФЕСТ: замер снят при чужих процессах на карте -- числа недействительны"
        )
        ok = False
    print(
        "манифест: полон; карта %s"
        % (card if isinstance(card, str) else card.get("index"))
    )
    return ok


def check_ship_sources():
    """Полнота поставки БЕЗ КАРТЫ: объявленный диспетчер обязан быть ОПРЕДЕЛЁН."""
    miss = [f for f in SHIP_SOURCES if not os.path.exists(os.path.join(HERE, f))]
    if miss:
        print("ПОСТАВКА НЕПОЛНА: нет " + ", ".join(miss))
        return False
    cu = open(os.path.join(HERE, "launch.cu"), encoding="utf-8").read()
    if "dispatch.inc" not in cu:
        print("launch.cu не включает dispatch.inc -- диспетчер объявлен и НЕ ОПРЕДЕЛЁН")
        return False
    cuh = open(os.path.join(HERE, "launch.cuh"), encoding="utf-8").read()
    if "SmemBytes" not in cuh or "cudaFuncSetAttribute" not in cuh:
        print(
            "launch.cuh пускает ядро без РАЗМЕРА динамической разделяемой либо без согласия "
            "на >48 КБ -- поставка не исполнится"
        )
        return False
    print(
        "поставка полна: %s; размер разделяемой и согласие на >48 КБ на месте"
        % ", ".join(SHIP_SOURCES)
    )
    return True


def run_with_card(nvcc=None, device=None):
    """СОБРАТЬ И ЗАПУСТИТЬ поставляемый путь.  Сборка -- ТОЛЬКО из файлов поставки.

    Свободная карта не нужна: сверяются ЗНАЧЕНИЯ, а не время.  Секундомера здесь нет.
    """
    probe = os.path.join(HERE, "ship_probe.cu")
    if not os.path.exists(probe):
        print("ЗАПУСК ПОСТАВКИ: нет ship_probe.cu -- проверять нечем")
        return False
    nvcc = nvcc or os.path.join(
        os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc"
    )
    if not os.path.exists(nvcc):
        nvcc = shutil.which("nvcc") or nvcc
    if not os.path.exists(nvcc):
        print(
            "ЗАПУСК ПОСТАВКИ: nvcc не найден (CUDA_HOME=%r) -- ПРОПУЩЕНО, а не пройдено"
            % os.environ.get("CUDA_HOME")
        )
        return False
    work = tempfile.mkdtemp(prefix="tempo_ship_")
    out = os.path.join(work, "ship_probe")
    log = os.path.join(work, "log.txt")
    cmd = [
        nvcc,
        "-O3",
        "-std=c++17",
        "-arch=sm_70",
        "-I",
        HERE,
        probe,
        os.path.join(HERE, "launch.cu"),
        "-o",
        out,
    ]
    cc = os.environ.get("CXX")
    if cc:
        cmd[1:1] = ["-ccbin", cc]

    def _run(argv, env=None):
        # Вывод -- В ФАЙЛ, а не в канал: сборка длинная, и канал под нагрузкой рвётся
        # (наблюдалось: selectors отдаёт None вместо тройки, и падает не сборка, а гейт).
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.call(argv, stdout=f, stderr=subprocess.STDOUT, env=env)
        return rc, open(log, encoding="utf-8", errors="replace").read()

    rc, text = _run(cmd)
    if rc != 0:
        print("ЗАПУСК ПОСТАВКИ: СБОРКА НЕ ПРОШЛА\n" + text[-2000:])
        return False
    env = dict(os.environ)
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    rc, text = _run([out], env=env)
    sys.stdout.write(text)
    return rc == 0


def oracle_procedure():
    print()
    print(
        "ORACLE.JSON НЕ ЗАПОЛНЕН НАМЕРЕННО. Файл-эталон, которому нельзя верить, опаснее"
    )
    print("отсутствующего. Процедура заполнения (нужна свободная карта):")
    print(
        "  1. закрепить частоты, убедиться, что чужих процессов 0 (иначе замер недействителен);"
    )
    print("  2. на каждой форме приёмки посчитать выход НАШИМ ядром;")
    print(
        "  3. сверить с эталоном ОДИНАРНОЙ точности и выше (НЕ с половинной -- ложный отрицательный);"
    )
    print(
        "  4. отдельно проверить ПОКРЫТИЕ атомарным штампом: непокрытых ячеек 0, повторных 0;"
    )
    print("  5. записать хэш выхода и допуск relL2 по каждой форме;")
    print("  6. снять частоты (-rgc) и записать число чужих процессов ПОСЛЕ.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-tree", help="корень хост-дерева, где ядро собирается")
    ap.add_argument(
        "--with-card",
        action="store_true",
        help="СОБРАТЬ И ЗАПУСТИТЬ поставляемый путь (нужна карта; свободная НЕ нужна)",
    )
    ap.add_argument("--device", help="номер карты для --with-card")
    a = ap.parse_args()
    ok = True
    ok &= check_predicate()
    ok &= check_dispatch_matches_select()
    ok &= check_ship_sources()
    ok &= check_manifest()
    if a.with_card:
        print()
        ok &= run_with_card(device=a.device)
    if a.host_tree:
        if not os.path.isdir(a.host_tree):
            print("хост-дерево %r не найдено" % a.host_tree)
            ok = False
        else:
            print("хост-дерево: %s" % a.host_tree)
            print(
                "  сборка и сверка значений требуют СВОБОДНОЙ карты; замер с соседом недействителен"
            )
    oracle_procedure()
    print()
    print(
        "ИТОГ ГЕЙТА%s: %s"
        % (
            " (С ЗАПУСКОМ ПОСТАВКИ)" if a.with_card else " БЕЗ КАРТЫ",
            "ПРОЙДЕН" if ok else "НЕ ПРОЙДЕН",
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
