#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ФОРМА закона конфликтности.  Числа -- у плагина, здесь только алгебра.

    конфликтность = |R| / |{ r*S mod B : r in R }|

где B -- число банков (параметр!), S -- шаг строки В СЛОВАХ, R -- множество строк, несущих
полосы одной команды.  Вывод формы: внутри команды все индексы, кроме строки, одинаковы,
поэтому банк = (const + r*S) mod B.

ЧТО ФОРМА НЕ ПОКРЫВАЕТ (объявлять вместе с ней, иначе она врёт молча):
  * нелинейную адресацию (перестановки/XOR) -- отображение не аффинно;
  * взаимодействие соседних массивов;
  * неоднородные карты полос (тогда R берётся замером, а не из объявления массива).

И ГЛАВНАЯ ОГОВОРКА О ЕДИНИЦЕ: форма предсказывает ТРАФИК, а не ВРЕМЯ.  Курс не единичный и
замерен: снятие 18.4 % трафика в ЭПИЛОГЕ дало 0.2 % времени, а 14.5 % в ТЕЛЕ ЦИКЛА -- 7.5 %.
Курс = доля конфликтов x сколько раз отрабатывает место.
"""

from __future__ import annotations


def degree(rows, stride_words: int, banks: int) -> float:
    """|R| / |{ r*S mod B }|.  banks -- ПАРАМЕТР, а не константа."""
    rows = list(rows)
    if not rows or banks <= 0:
        return 1.0
    return len(rows) / float(len({(r * stride_words) % banks for r in rows}))


def degree_from_lane_map(lane_words, banks: int) -> float:
    """Конфликтность прямо из карты полос: максимум по банкам числа РАЗЛИЧНЫХ слов."""
    per = {}
    for w in lane_words:
        if w is None:
            continue
        per.setdefault(int(w) % banks, set()).add(int(w))
    return float(max((len(s) for s in per.values()), default=1))


def period(stride_words: int, banks: int) -> int:
    """Период кривой дополнения = шаг столбца итератора.

    Знание периода превращает перебор из поиска в ФАЛЬСИФИКАТОР: хватает одного периода,
    а не всего диапазона.
    """
    g = _gcd(abs(stride_words), banks) or 1
    return max(1, banks // g)


def pad_candidates(stride_words: int, banks: int):
    """Кандидаты в дополнение -- ровно один период.

    ОДНОСТОРОННЕЕ «дополнить» НЕВЕРНО: замерено, что pad=4 и pad=12 дают ХУЖЕ базы.
    И минимум != выбор: pad=9 даёт 15.57 ценой +3456 Б, pad=1 даёт 15.58 ценой +384 Б.
    """
    return range(0, period(stride_words, banks) + 1)


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a
