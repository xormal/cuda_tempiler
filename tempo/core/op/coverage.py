#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ПОКРЫТИЕ АТОМАРНЫМ ШТАМПОМ -- гейт корректности, который ловит то, чего не ловит сверка.

ЗАЧЕМ ЭТО ВООБЩЕ ЕСТЬ (случай оплачен днём работы).  Первая версия плотного мейнлупа считала
ЧЕТВЕРТЬ плитки и ПРОХОДИЛА сверку значений с эталоном -- потому что сверка сравнивала ровно
те ячейки, которые вычислялись.  Ошибку показал ЗАМЕР СКОРОСТИ («слишком быстро»), а не
проверка значений.

    КОРРЕКТНОСТЬ ОПЕРАТОРА ПРОВЕРЯТЬ ПОКРЫТИЕМ, А НЕ ЗНАЧЕНИЯМИ.

Механика: ядро собирается со штампом -- каждая записанная ячейка выхода атомарно увеличивает
свой счётчик.  После прогона обязано быть РОВНО ПО ОДНОМУ на ячейку.
    0 -- ячейка не посчитана (та самая четверть плитки);
    >1 -- посчитана дважды (наложение блоков; при накоплении даёт удвоение молча).
"""

from __future__ import annotations

from dataclasses import dataclass

STAMP_MACRO = "TEMPO_COVERAGE_STAMP"


@dataclass
class Coverage:
    total: int
    zero: int
    over: int
    max_hits: int

    @property
    def ok(self) -> bool:
        return self.zero == 0 and self.over == 0

    def render(self) -> str:
        if self.ok:
            return "покрытие ПОЛНОЕ: %d ячеек, каждая ровно один раз" % self.total
        return (
            "покрытие НАРУШЕНО: непокрытых %d, посчитанных больше одного раза %d "
            "(максимум %d на ячейку) из %d"
            % (self.zero, self.over, self.max_hits, self.total)
        )


def check(stamps) -> Coverage:
    """stamps -- одномерная последовательность счётчиков по ячейкам выхода."""
    total = zero = over = 0
    mx = 0
    for v in stamps:
        total += 1
        v = int(v)
        if v == 0:
            zero += 1
        elif v > 1:
            over += 1
        if v > mx:
            mx = v
    return Coverage(total=total, zero=zero, over=over, max_hits=mx)


def instrument_hint() -> str:
    return (
        "Собрать ядро с -D%s: каждая запись в выход делает atomicAdd(&stamp[idx], 1). "
        "Штамп ОТКЛЮЧАЕТСЯ в поставке; его цена измеряется отдельно и в замер не входит."
        % STAMP_MACRO
    )
