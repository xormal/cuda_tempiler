#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""sm_70: ПАМЯТЬ.  Замкнутая форма конфликтности и цена в ВАЙВФРОНТАХ.

ФОРМА ЗАКОНА живёт в core/model/banks.py (она не архитектурная: 'банков B, шаг S').
ЗДЕСЬ -- только числа этой машины (B = 32 банка по 4 Б) и семейства дополнений/свизлов.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from ..base import Layout, MemLevel, Rate, WavefrontCost
from .machine import Sm70Machine, data

BANKS = 32
BANK_BYTES = 4

_M = Sm70Machine()


class Sm70Memory:
    def levels(self):
        d = data("memory")
        out = []
        for lv in d["levels"]:
            out.append(
                MemLevel(
                    name=lv["name"],
                    bytes_=Rate(
                        symbol="MEM.%s.BYTES" % lv["name"],
                        value=float(lv["bytes"]),
                        units="байт",
                        status=lv["bytes_status"],
                        note=lv.get("note", ""),
                    ),
                    bandwidth=Rate(
                        symbol="MEM.%s.BW" % lv["name"],
                        value=float(lv["bw"]),
                        units="ГБ/с",
                        status=lv["bw_status"],
                        note=lv.get("note", ""),
                    ),
                    latency=Rate(
                        symbol="MEM.%s.LAT" % lv["name"],
                        value=float(lv["lat"]),
                        units="такт",
                        status=lv["lat_status"],
                        note=lv.get("note", ""),
                    ),
                )
            )
        return tuple(out)

    # -- ЗАКОН ЦЕНЫ --------------------------------------------------------------------------
    def wavefronts(self, lane_words: Sequence, width_bytes: int) -> WavefrontCost:
        """вайвфронтов = max( конфликтность , ширина_на_полосу/8 Б , 1 ).

        lane_words -- адрес каждой полосы В СЛОВАХ (None = полоса не участвует).
        Конфликтность = максимум по банкам числа РАЗЛИЧНЫХ слов, запрошенных в этом банке.
        Рассылка одного адреса БЕСПЛАТНА ПОЛНОСТЬЮ (замерено: ровно 32/DUP, насыщения нет).
        """
        per_bank = {}
        for w in lane_words:
            if w is None:
                continue
            b = int(w) % BANKS
            per_bank.setdefault(b, set()).add(int(w))
        degree = float(max((len(s) for s in per_bank.values()), default=1))
        floor = max(width_bytes / 8.0, 1.0)
        return WavefrontCost(degree=degree, floor=floor, wavefronts=max(degree, floor))

    def alignment_rule(self, width_bytes: int) -> int:
        """Шаг строки обязан быть кратен ШИРИНЕ ДОСТУПА В БАЙТАХ, а не в элементах."""
        return int(width_bytes)

    # -- СЕМЕЙСТВА ЛЕЧЕНИЯ -------------------------------------------------------------------
    def pad_family(self, layout: Layout) -> Iterable[int]:
        """Дополнения-кандидаты В СЛОВАХ.

        ПЕРЕБОР, а НЕ вывод: замерено, что аргминимум рассуждением не предсказывается (для
        шага 68 слов аргминимум по семейству карт доступа = {0,1,2}, единого нет).
        Период кривой = шаг столбца итератора, поэтому хватает одного периода.
        """
        period = max(1, BANKS // max(1, _gcd(layout.row_words, BANKS)))
        return range(0, max(period, 16))

    def swizzle_family(self, layout: Layout) -> Iterable[str]:
        """Свизлы-кандидаты.

        ЗАМЕРЕНО (2026-08-02): 'xor_chunk' в отгруженном volta_hmma.h НЕ бесконфликтен --
        он берёт разряды строки {0..kChunks-1}, а фаза из восьми полос различается
        РАЗРЯДОМ 3.  Отсюда 'xor_phase' (пара разрядов {1,3}) -- он и даёт 0.008.
        """
        n = max(1, layout.row_words * layout.elem_bytes // 16)
        out = ["none", "xor_chunk", "xor_phase"]
        if n >= 4:
            out.append("xor_phase_wide")
        return out

    def residency_policy(self) -> str:
        return data("memory")["residency_policy"]["value"]


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a or 1
