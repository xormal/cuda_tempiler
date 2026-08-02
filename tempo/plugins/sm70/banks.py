#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""sm_70: ЧИСЛА банковой арифметики.  ФОРМА закона -- в core/model/banks.py.

Разделение намеренное и проверяемое гейтом G1: `|R| / |{r*S mod B}|` не содержит ничего
волтовского, кроме B.  Здесь B = 32 банка по 4 байта.

ЕДИНСТВЕННАЯ РЕАЛИЗАЦИЯ ЗАКОНА живёт в `tools/bankform.py` (самопроверка пройдена, 13 точек
перебора, уровни 1/2/4 разделены с разбросом 0.46/0.02/0.00 п.п.).  Здесь -- мост к ней.
"""

from __future__ import annotations

import importlib.util
import os

BANKS = 32
BANK_BYTES = 4

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def _bankform():
    p = os.path.join(_ROOT, "tools", "bankform.py")
    spec = importlib.util.spec_from_file_location("tempo_tool_bankform", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def degree(rows, stride_words: int) -> float:
    """Замкнутая форма: |R| / |{ r*S mod 32 }|.  R -- строки из КАРТЫ ПОЛОС, не из объявления."""
    rows = list(rows)
    if not rows:
        return 1.0
    banks = {(r * stride_words) % BANKS for r in rows}
    return len(rows) / float(len(banks))


def phase_rows(width_bytes: int = 16):
    """Строки ОДНОЙ ФАЗЫ доступа.

    ЭТО ТА САМАЯ ЕДИНИЦА, НА КОТОРОЙ ОТГРУЖЕННЫЙ СВИЗЛ ОКАЗАЛСЯ НЕВЕРЕН (2026-08-02):
    LDS.128 обслуживается ФАЗАМИ ПО ВОСЕМЬ ПОЛОС, и внутри фазы карта b_col даёт строки
    {0,1,2,3,8,9,10,11} -- они различаются РАЗРЯДОМ 3, которого нет ни в шаге строки, ни в
    XOR-свизле `c ^ (row & (kChunks-1))`.  Отсюда 2.038 конфликта на LDS.128 при обещанной
    бесконфликтности.  Лечение -- свизл по паре разрядов {1,3}: перебор всех 30 пар даёт
    0.008 ровно на ней и ни на какой другой.
    """
    lanes_per_phase = max(1, 32 * 4 // max(4, width_bytes))
    quad = lanes_per_phase // 4
    return [(l & 3) | ((l >> 2) << 3) for l in range(lanes_per_phase)][: 4 * max(1, quad)]


def swizzle_phase_injective(row: int, chunks: int) -> int:
    """x = ((row>>1)&1) | ((row>>2)&2) -- фазово-инъективный свизл при BK=32.

    Найден ПЕРЕБОРОМ всех 30 пар разрядов, а не выведен: замкнутой формы для свизлов у нас
    нет (форма конфликтности НЕ покрывает XOR-адресацию -- отображение нелинейно).
    """
    return (((row >> 1) & 1) | ((row >> 2) & 2)) & max(0, chunks - 1)
