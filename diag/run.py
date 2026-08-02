#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ЕДИНАЯ ТОЧКА ЗАПУСКА ЯКОРНЫХ ИНСТРУМЕНТОВ.

Логики здесь нет и быть не должно: сам инструмент лежит в `tools/`, проверен якорем и
переписыванию не подлежит.  Этот файл делает ровно две вещи, обе -- лечение замеренных
граблей:

  1. ВЫБИРАЕТ ИНТЕРПРЕТАТОР и объясняет выбор.  Замерено: один из инструментов не
     разбирается на 3.11 (вложенные форматные строки), а базовый интерпретатор дистрибутива
     тащит слишком новый хост-компилятор, который компилятор устройства отвергает.
  2. ВЫЧИЩАЕТ КАТАЛОГ ИНСТРУМЕНТОВ из путей поиска перед запуском.  Замерено: файл
     инструмента с именем стандартного модуля перехватывает импорт у сторонней библиотеки,
     тело падает ДО первой полезной строки, а прибор отдаёт пустую таблицу, которая
     читается как «нарушений нет».

    python3 diag/run.py                      -- карта инструментов
    python3 diag/run.py timeit --selftest    -- прокинуть аргументы как есть
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TOOLS = os.path.join(ROOT, "tools")

# инструмент -> (нужен ли интерпретатор >= 3.12, нужна ли свободная карта, что отвечает)
CATALOG = {
    "timeit": (False, True, "во сколько раз A быстрее B (парные отношения, 6 гейтов)"),
    "phaseprof": (
        False,
        True,
        "доля ФАЗЫ во времени (снять фазу, читать только время)",
    ),
    "ncu": (False, True, "вайвфронты и доля конфликтов, два независимых маршрута"),
    "smem_lint": (False, False, "конфликтность из ИСХОДНИКА, без карты"),
    "bankform": (False, False, "замкнутая форма конфликтности"),
    "cc_ab": (False, False, "регистры / разлив / КАДР СТЕКА / занятость"),
    "knee": (False, False, "точка регистрового излома"),
    "tempo": (False, False, "граница периода и связывающий канал"),
    "schedule": (False, False, "циклическое расписание, MaxLive, вердикт"),
    "relayout": (
        False,
        False,
        "ЧТО СВЯЗЫВАЕТ и ЧТО ПЕРЕЛОЖИТЬ: вердикт + меню переукладок",
    ),
    "twin": (False, False, "ворота дрейфа профилируемого двойника"),
    "calib": (False, False, "ставки из замера + ОТЧЁТ О НЕВЯЗКЕ"),
    "padsweep": (False, True, "кривая дополнения (замеренных кривых 0 -- ДОЛГ)"),
    "residency": (True, False, "влезает ли рабочее множество в уровни памяти"),
    "predict_fmt": (False, False, "предсказать ПОРЯДОК вариантов"),
    "predict_kernels": (False, False, "доля пика, которую разрешает отпечаток"),
    "bankaudit": (False, True, "аудит конфликтности по отгруженным телам"),
    "sasscount": (False, False, "счётчик опкодов по дампу"),
    "sweep": (False, True, "дисциплина раунда: чередование, медиана, состояние карты"),
    "bits_table": (False, True, "разрядность и её цена по точности"),
    "precheck": (False, False, "состояние окружения ДО тяжёлого замера"),
    "tempo_cli": (False, False, "диспетчер прежнего стека (6 подкоманд)"),
}


def _env():
    p = os.path.join(ROOT, "tempo", "cli", "env.py")
    spec = importlib.util.spec_from_file_location("tempo_env", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def catalog():
    print("ЯКОРНЫЕ ИНСТРУМЕНТЫ (физически -- tools/, карта -- diag/README.md)")
    print()
    for name, (py312, card, what) in sorted(CATALOG.items()):
        marks = []
        if py312:
            marks.append("нужен 3.12+")
        if card:
            marks.append("НУЖНА СВОБОДНАЯ КАРТА")
        print("  %-18s %-58s %s" % (name, what, "; ".join(marks)))
    print()
    print("Запуск:  python3 diag/run.py <имя> [аргументы инструмента как есть]")
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        return catalog()
    name = argv[0]
    if name not in CATALOG:
        print("нет инструмента %r; см. python3 diag/run.py" % name)
        return 2
    path = os.path.join(TOOLS, name + ".py")
    if not os.path.exists(path):
        print("файл %s отсутствует" % path)
        return 2
    e = _env()
    py312, card, _ = CATALOG[name]
    py = (e.python_312() if py312 else e.python_vllm()) or sys.executable
    print("# интерпретатор: %s" % py)
    if card:
        print(
            "# ВНИМАНИЕ: инструменту нужна СВОБОДНАЯ карта. Замер с соседом НЕДЕЙСТВИТЕЛЕН."
        )
    env = dict(os.environ)
    # вычищаем каталог инструментов из путей поиска: см. шапку
    pp = [
        q
        for q in env.get("PYTHONPATH", "").split(os.pathsep)
        if q and os.path.abspath(q) != TOOLS
    ]
    env["PYTHONPATH"] = os.pathsep.join(pp)
    env.setdefault("LC_ALL", "C")
    return subprocess.call([py, path] + list(argv[1:]), cwd=ROOT, env=env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
