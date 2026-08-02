#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ПЛЕЧО-СОПЕРНИК КАК ОБЪЕКТ.  Число без названной базы сравнения -- не число.

ТРИ РАЗНЫЕ ПЛАНКИ, и смешивать их запрещено:

  INPUT    -- наивный вход, собранный как есть.  Собственная метрика компилятора
              («во сколько раз выход обгоняет вход»).  ЧЕСТНА, НО ТРИВИАЛЬНО ВЕЛИКА, и
              печатается ТОЛЬКО В ПАРЕ со второй планкой.
  LIBRARY  -- вылизанная чужая библиотека той же машины.  АБСОЛЮТНАЯ планка.
  SHIPPED  -- наше собственное отгруженное ядро.  Планка для тех операций, где у нас уже
              есть сильное тело; сравнение с соломенным чучелом обесценивает отчёт.
  FORMAT   -- наш же путь в другом формате данных (планка для форматных ходов, где у чужой
              библиотеки пути просто НЕТ и сравнивать не с кем).

ПОЧЕМУ ЭТО ОТДЕЛЬНЫЙ ТИП.  Наряд этого проекта однажды опёрся на отношение 1.85, взятое из
колонки таблицы, у которой база сравнения была ДРУГАЯ (не то, что подразумевал читатель).
Ошибка прошла через памятку, сводку, постановку и отчёт, ни разу не встретив замера, потому
что все цитировали ОДИН источник.  Тип `Baseline` делает базу обязательным полем.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

KINDS = ("INPUT", "LIBRARY", "SHIPPED", "FORMAT")


@dataclass(frozen=True)
class Baseline:
    kind: str  # один из KINDS
    name: str  # человекочитаемое имя плеча
    what: str  # ЧТО ИМЕННО меряется этим плечом
    entry_proof: Optional[str] = None  # чем доказано, что зашли именно туда

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError("база %r вне %r" % (self.kind, KINDS))

    def label(self) -> str:
        return "%s (%s)" % (self.name, self.kind)


@dataclass(frozen=True)
class Comparison:
    baseline: Baseline
    ratio: float  # >1 = мы быстрее
    ci: tuple = ()  # доверительный интервал медианы отношений
    rounds: int = 0
    note: str = ""

    def render(self) -> str:
        ci = ""
        if self.ci and len(self.ci) == 2:
            ci = " [%.3f..%.3f]" % (self.ci[0], self.ci[1])
        return "против %-28s x%.3f%s  раундов %d%s" % (
            self.baseline.label(), self.ratio, ci, self.rounds,
            ("  -- " + self.note) if self.note else "",
        )


def two_bars_required(comps) -> None:
    """ДВЕ ПЛАНКИ ОБЯЗАТЕЛЬНЫ.  Первая без второй -- самообман; вторая без первой не отвечает
    на вопрос, продукт ли это сделал."""
    kinds = {c.baseline.kind for c in comps}
    if "INPUT" not in kinds:
        raise ValueError(
            "нет планки INPUT: без неё неизвестно, компилятор ли сделал результат"
        )
    if not kinds & {"LIBRARY", "SHIPPED", "FORMAT"}:
        raise ValueError(
            "нет ВТОРОЙ планки: «во сколько раз обогнали вход» тривиально велико "
            "(наивный исходник даёт единицы против десятков) и в одиночку является самообманом"
        )


def wall_share_required(share: Optional[float]) -> str:
    """ТРЕТЬЕ ЧИСЛО: доля работы этого ядра в стене.

    Урок оплачен днём: ядро дало x1.36 ИЗОЛИРОВАННО и 0.6 % СКВОЗНЯКОМ, потому что его работа
    -- 16.3 % стены.  Счётчик стоил одного прогона, а встраивание -- дня.
    """
    if share is None:
        return (
            "ДОЛЯ В СТЕНЕ НЕ ИЗМЕРЕНА -- ускорение ядра НЕЛЬЗЯ переводить в ускорение задачи"
        )
    return "доля работы этого ядра в стене: %.1f %% (потолок сквозного эффекта)" % (100.0 * share)
