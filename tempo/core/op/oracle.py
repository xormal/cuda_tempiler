#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ВХОД КАК ЭТАЛОН И КАК ЗНАМЕНАТЕЛЬ.

Наивный исходник играет ТРИ роли, и их нельзя путать:
    1. источник спецификации (блок TEMPO-OP);
    2. ЭТАЛОН ЗНАЧЕНИЙ -- относительная L2 считается к нему и к независимому счёту;
    3. ЗНАМЕНАТЕЛЬ метрики «во сколько раз выход обгоняет вход».

ЛОВУШКА ЭТАЛОНА, ЗАМЕРЕННАЯ И ЗАПИСАННАЯ ПРАВИЛОМ.  Сверять с половинной точностью нельзя:
штатное матричное умножение в половинной точности ВОЗВРАЩАЕТ ПОЛОВИННУЮ, и суммы порядка
4e6 переполняют ВЫХОД при исправном одинарном накопителе.  Проверка «штатным» путём даёт
ЛОЖНЫЙ ОТРИЦАТЕЛЬНЫЙ.  Эталон обязан быть в одинарной точности или выше.

И ВТОРАЯ, СИММЕТРИЧНАЯ: две сверки, читающие ОДИН И ТОТ ЖЕ промежуточный буфер, соглашаются,
будучи ОБЕ неверны.  Симметрия ломается подменой на ЧУЖОЙ путь по одному узлу (например,
независимый счёт в двойной точности на разрежённой выборке ячеек).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OracleResult:
    rel_l2: float
    tol: float
    coverage: object = None
    independent_rel_l2: float = float("nan")
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        if self.rel_l2 != self.rel_l2:  # NaN
            return False
        if self.rel_l2 > self.tol:
            return False
        if self.coverage is not None and not self.coverage.ok:
            return False
        return True

    def render(self) -> str:
        head = "ГЕЙТ КОРРЕКТНОСТИ: relL2 = %.3e при допуске %.3e -- %s" % (
            self.rel_l2, self.tol, "ПРОЙДЕН" if self.ok else "НЕ ПРОЙДЕН"
        )
        if self.coverage is not None:
            head += "\n  " + self.coverage.render()
        if self.independent_rel_l2 == self.independent_rel_l2:
            head += "\n  независимая сверка (ЧУЖОЙ путь, ломка симметрии): %.3e" % (
                self.independent_rel_l2
            )
        for n in self.notes:
            head += "\n  " + n
        return head


def rel_l2(got, ref) -> float:
    """Относительная L2.  Обе последовательности обязаны быть в одинарной точности и выше."""
    num = 0.0
    den = 0.0
    for a, b in zip(got, ref):
        d = float(a) - float(b)
        num += d * d
        den += float(b) * float(b)
    if den <= 0.0:
        return float("nan")
    return (num / den) ** 0.5


def gate(got, ref, tol: float, stamps=None, independent=None) -> OracleResult:
    """ГЕЙТ КОРРЕКТНОСТИ РАНЬШЕ СЕКУНДОМЕРА.  Порядок не обсуждается: не оптимизируют неверное."""
    from .coverage import check

    cov = check(stamps) if stamps is not None else None
    ind = rel_l2(got, independent) if independent is not None else float("nan")
    res = OracleResult(rel_l2=rel_l2(got, ref), tol=tol, coverage=cov, independent_rel_l2=ind)
    if cov is not None and cov.zero:
        res.notes.append(
            "НЕПОКРЫТЫЕ ЯЧЕЙКИ при пройденной сверке значений -- это ровно тот случай, когда "
            "сверка смотрит только на посчитанное. Верить покрытию."
        )
    return res


def speedup_vs_input(t_input_s: float, t_output_s: float) -> float:
    """Собственная метрика компилятора.  ЧЕСТНА, НО ТРИВИАЛЬНО ВЕЛИКА.

    Печатать её РАЗРЕШЕНО ТОЛЬКО В ПАРЕ со второй планкой (сравнение с сильным соперником):
    обогнать плохой вход легко, и число без пары -- самообман.
    """
    if t_output_s <= 0:
        return float("nan")
    return t_input_s / t_output_s
