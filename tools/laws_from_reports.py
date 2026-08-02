#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ПОПОЛНЕНИЕ РЕЕСТРА ИЗ ОТЧЁТОВ: регистрация висит на артефакте, который и так пишется.

ЗАЧЕМ ИМЕННО ТАК.  «Заводить реестр» как отдельную дисциплину бесполезно: её соблюдают неделю.
Механизм обязан быть таким, чтобы ВНЕСТИ было ДЕШЕВЛЕ, чем НЕ внести.  Поэтому он висит на
`reports/EV_*.md` -- на единственном артефакте, который производится каждый раз, когда что-то
открыто.  Написание отчёта СТАНОВИТСЯ регистрацией.

ЧТО ДЕЛАЕТ
  anchors  сверяет ЧИСЛА якорей реестра с ТЕЛОМ названного отчёта.  Это ловит ровно ту ошибку,
           ради которой реестр и заводился: закон со ссылкой на отчёт, в котором его числа НЕТ
           (в описи дня такой случай был -- «M 16->32 стоит -2.7 %» со ссылкой на таблицу,
           которая начинается с M=32).
  extract  вынимает блоки  <!--LAW {...} -->  из отчётов и печатает их как записи реестра;
           отказывает, если блок без фальсификатора, без области или с чужими процессами на карте.
  cover    какие отчёты НЕ дали ни одной записи (кандидаты на «открытие потеряно»).

ЧЕГО ОН НЕ ДЕЛАЕТ.  Он НЕ ПИШЕТ в реестр сам.  Перенос записи в `data/laws/` -- отдельный шаг
с прогоном гейтов между: место и содержимое одним шагом здесь не меняют.

ЗАПУСК
    python3 tools/laws_from_reports.py anchors  [--reports DIR]
    python3 tools/laws_from_reports.py extract  [--reports DIR]
    python3 tools/laws_from_reports.py cover    [--reports DIR]
    python3 tools/laws_from_reports.py --selftest      # карта не нужна
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tempo.core import laws as L  # noqa: E402

# Каталог отчётов лежит ВНЕ дерева продукта намеренно: отчёт -- запись о замере, а не исходник.
# Путь -- умолчание, а не вшитая зависимость: без каталога инструмент ПРОПУСКАЕТ с причиной.
DEFAULT_REPORTS = os.environ.get("TEMPO_REPORTS", "/mnt/d1/alex/reports")

BLOCK_OPEN = "<!--LAW"
BLOCK_CLOSE = "-->"


# ------------------------------------------------------------------------------------------------
# 1. СВЕРКА ЧИСЕЛ ЯКОРЯ С ТЕЛОМ ОТЧЁТА
# ------------------------------------------------------------------------------------------------
def check_anchors(root: str, reports: str, out=print) -> int:
    """Каждое число якоря обязано ВСТРЕЧАТЬСЯ в названном отчёте.

    Сверка нормализует пробелы и регистр: число, перенесённое по строке, -- то же число, а
    проверка, которую обходит перенос строки, не проверяет ничего.
    """
    if not os.path.isdir(reports):
        out(
            "  ПРОПУЩЕНО: каталога отчётов %s нет (запись о замере вне дерева продукта)"
            % reports
        )
        return 0
    bad = 0
    for law in L.load(root):
        path = os.path.join(reports, law.anchor.report)
        if not os.path.exists(path):
            out("  ПАДЁТ  %-28s отчёта %s НЕТ" % (law.id, law.anchor.report))
            bad += 1
            continue
        with open(path, encoding="utf-8", errors="ignore") as fh:
            body = L.normalize(fh.read())
        miss = [n for n in law.anchor.numbers if L.normalize(n) not in body]
        if miss:
            out(
                "  ПАДЁТ  %-28s в %s НЕТ чисел: %s"
                % (law.id, law.anchor.report, "; ".join(miss))
            )
            bad += 1
    out("  сверено записей: %d, расхождений: %d" % (len(L.load(root)), bad))
    return bad


# ------------------------------------------------------------------------------------------------
# 2. ИЗВЛЕЧЕНИЕ БЛОКОВ ИЗ ОТЧЁТОВ
# ------------------------------------------------------------------------------------------------
def blocks_in(text: str) -> list:
    """Все блоки <!--LAW {...} --> в порядке следования.  Разбор без регулярных выражений:
    внутри JSON бывают и скобки, и стрелки, и на них regex спотыкается молча."""
    out = []
    i = 0
    while True:
        i = text.find(BLOCK_OPEN, i)
        if i < 0:
            return out
        j = text.find(BLOCK_CLOSE, i)
        if j < 0:
            out.append((i, None, "блок открыт и НЕ ЗАКРЫТ"))
            return out
        raw = text[i + len(BLOCK_OPEN) : j].strip()
        try:
            out.append((i, json.loads(raw), ""))
        except ValueError as e:
            out.append((i, None, "блок не разбирается как запись: %s" % e))
        i = j + len(BLOCK_CLOSE)


def strip_blocks(text: str) -> str:
    """Тело отчёта БЕЗ блоков регистрации.

    Сверять число якоря с текстом, включающим сам блок, бессмысленно: число найдётся в нём же.
    Первая редакция самопроверки на этом и попалась -- проверка была зелёной и не проверяла
    ничего.  Якорь обязан указывать на ПРОЗУ отчёта, то есть на то, что померено.
    """
    out = []
    i = 0
    while True:
        j = text.find(BLOCK_OPEN, i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        k = text.find(BLOCK_CLOSE, j)
        if k < 0:
            return "".join(out)
        i = k + len(BLOCK_CLOSE)


def extract(reports: str, out=print) -> int:
    if not os.path.isdir(reports):
        out("  ПРОПУЩЕНО: каталога отчётов %s нет" % reports)
        return 0
    bad = 0
    found = 0
    for fn in sorted(os.listdir(reports)):
        if not (fn.startswith("EV_") and fn.endswith(".md")):
            continue
        path = os.path.join(reports, fn)
        with open(path, encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        body = L.normalize(strip_blocks(text))
        for _pos, d, err in blocks_in(text):
            if d is None:
                out("  ПАДЁТ  %s: %s" % (fn, err))
                bad += 1
                continue
            found += 1
            try:
                law = L._law_from(dict(d, schema=d.get("schema", L.SCHEMA)), "", fn)
            except L.LawError as e:
                out("  ПАДЁТ  %s: %s" % (fn, e))
                bad += 1
                continue
            problems = L.check_record(law)
            # Число, которого нет в теле СВОЕГО ЖЕ отчёта, -- главная ловушка регистрации.
            problems += [
                "%s: числа %r нет в теле отчёта %s" % (law.id, n, fn)
                for n in law.anchor.numbers
                if L.normalize(n) not in body
            ]
            for p in problems:
                out("  ПАДЁТ  " + p)
            bad += len(problems)
            if not problems:
                out("  ПРИНЯТ %-28s %s" % (law.id, fn))
    out("  блоков найдено: %d, отказов: %d" % (found, bad))
    return bad


def cover(root: str, reports: str, out=print) -> int:
    """Отчёты, не давшие ни одной записи.  Это НЕ отказ: отчёт бывает и отрицательным
    результатом без закона.  Но список полезен -- открытие теряется молча."""
    if not os.path.isdir(reports):
        out("  ПРОПУЩЕНО: каталога отчётов %s нет" % reports)
        return 0
    used = {l.anchor.report for l in L.load(root)}
    names = [
        f
        for f in sorted(os.listdir(reports))
        if f.startswith("EV_") and f.endswith(".md")
    ]
    silent = [f for f in names if f not in used]
    out(
        "  отчётов: %d, из них дали запись реестра: %d"
        % (len(names), len(names) - len(silent))
    )
    for f in silent:
        with open(os.path.join(reports, f), encoding="utf-8", errors="ignore") as fh:
            has_block = BLOCK_OPEN in fh.read()
        out(
            "    БЕЗ ЗАПИСИ  %-28s %s"
            % (f, "(блок есть, перенос не сделан)" if has_block else "")
        )
    return 0


# ------------------------------------------------------------------------------------------------
# 3. САМОПРОВЕРКА
# ------------------------------------------------------------------------------------------------
FIXTURE = """
# Заведомо неверный отчёт-образец

Число, которое в теле есть: 12.34 такта.

## ЗАКОНЫ ЭТОГО ОТЧЁТА
<!--LAW {"id": "L-FIXTURE-OK", "kind": "ДОЛГ",
 "statement": "образец правильной записи",
 "anchor": {"report": "EV_fixture.md", "numbers": ["12.34"], "status": "SASS"},
 "falsifier": {"what": "что угодно", "how": "как угодно"},
 "scope": {"applies": "образец"}} -->
<!--LAW {"id": "L-FIXTURE-NO-NUMBER", "kind": "ДОЛГ",
 "statement": "число якоря в теле отчёта ОТСУТСТВУЕТ",
 "anchor": {"report": "EV_fixture.md", "numbers": ["99.99"], "status": "SASS"},
 "falsifier": {"what": "что угодно", "how": "как угодно"},
 "scope": {"applies": "образец"}} -->
<!--LAW {"id": "L-FIXTURE-NO-FALSIFIER", "kind": "ДОЛГ",
 "statement": "нет фальсификатора и нет области",
 "anchor": {"report": "EV_fixture.md", "numbers": ["12.34"], "status": "SASS"}} -->
<!--LAW {"id": "L-FIXTURE-FOREIGN", "kind": "ДОЛГ",
 "statement": "замер снят при чужом процессе на карте",
 "anchor": {"report": "EV_fixture.md", "numbers": ["12.34"], "status": "MEASURED",
            "card": {"index": 0, "clock_mhz": 1530, "foreign_procs": 3, "date": "2026-08-02"}},
 "falsifier": {"what": "что угодно", "how": "как угодно"},
 "scope": {"applies": "образец"}} -->
"""


def selftest(out=print) -> int:
    ok = bad = 0

    def chk(label, cond, detail=""):
        nonlocal ok, bad
        out(
            "  %-9s %s%s"
            % ("ПРОЙДЕН" if cond else "ПАДЁТ", label, ("  " + detail) if detail else "")
        )
        if cond:
            ok += 1
        else:
            bad += 1

    out("САМОПРОВЕРКА ПОПОЛНЕНИЯ РЕЕСТРА (карта не нужна)")

    # 1. Разбор блоков: четыре записи, ни одна не потеряна.
    blks = blocks_in(FIXTURE)
    chk(
        "1. блоки отчёта разбираются",
        len(blks) == 4 and all(d is not None for _, d, _ in blks),
        "найдено %d" % len(blks),
    )

    # 2. ГЛАВНАЯ ЛОВУШКА: число якоря, которого нет в теле СВОЕГО ЖЕ отчёта.
    body = L.normalize(strip_blocks(FIXTURE))
    miss = []
    for _p, d, _e in blks:
        if d is None:
            continue
        for n in d["anchor"]["numbers"]:
            if L.normalize(n) not in body:
                miss.append((d["id"], n))
    chk(
        "2. ловится число, которого нет в теле отчёта",
        miss == [("L-FIXTURE-NO-NUMBER", "99.99")],
        str(miss),
    )

    # 3. Запись без фальсификатора и без области ОТВЕРГАЕТСЯ на разборе.
    try:
        L._law_from(dict(blks[2][1], schema=L.SCHEMA), "", "EV_fixture.md")
        rejected = False
    except L.LawError:
        rejected = True
    chk("3. запись без фальсификатора отвергается ОТКАЗОМ", rejected)

    # 4. MEASURED при чужих процессах на карте -- замер недействителен.
    law = L._law_from(dict(blks[3][1], schema=L.SCHEMA), "", "EV_fixture.md")
    probs = L.check_record(law)
    chk(
        "4. MEASURED при чужом процессе на карте отвергается",
        any("чужих процессах" in p for p in probs),
        "; ".join(probs) or "претензий нет",
    )

    # 5. Правильная запись проходит ВСЕ проверки записи (иначе проверка ловит всё подряд).
    good = L._law_from(dict(blks[0][1], schema=L.SCHEMA), "", "EV_fixture.md")
    chk(
        "5. правильная запись проходит",
        not L.check_record(good),
        "; ".join(L.check_record(good)),
    )

    # 6. ЧИСЛА ЯКОРЕЙ БОЕВОГО РЕЕСТРА против тел отчётов -- если отчёты доступны.
    n_bad = check_anchors(_ROOT, DEFAULT_REPORTS, out=lambda *_: None)
    if os.path.isdir(DEFAULT_REPORTS):
        chk(
            "6. числа якорей реестра ЕСТЬ в названных отчётах",
            n_bad == 0,
            "расхождений %d" % n_bad,
        )
    else:
        out("  ПРОПУЩЕН  6. сверка с отчётами: каталога %s нет" % DEFAULT_REPORTS)

    out("ИТОГ САМОПРОВЕРКИ: пройдено %d, упало %d" % (ok, bad))
    return 0 if bad == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="laws_from_reports",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("cmd", nargs="?", choices=("anchors", "extract", "cover"))
    ap.add_argument("--reports", default=DEFAULT_REPORTS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest or not a.cmd:
        return selftest()
    if a.cmd == "anchors":
        print("СВЕРКА ЧИСЕЛ ЯКОРЕЙ С ТЕЛАМИ ОТЧЁТОВ")
        return 1 if check_anchors(_ROOT, a.reports) else 0
    if a.cmd == "extract":
        print("БЛОКИ ЗАКОНОВ В ОТЧЁТАХ")
        return 1 if extract(a.reports) else 0
    print("ПОКРЫТИЕ ОТЧЁТОВ РЕЕСТРОМ")
    return cover(_ROOT, a.reports)


if __name__ == "__main__":
    sys.exit(main())
