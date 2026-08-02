#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""АРИФМЕТИКА ВОЛНЫ И ХВОСТА.  Число процессоров даёт ПЛАГИН, здесь только счёт.

Урок, стоивший проекту победы над соперником: полезные размеры сетки кратны ЧИСЛУ
ПРОЦЕССОРОВ, а не круглым числам.  Перебор по {1,2,4,6,8,12} систематически промахивал 80 и
промахнёт 108; попадание в квант дало +2.4..7.7 %.
"""

from __future__ import annotations


def waves(ctas: int, quantum: int) -> float:
    if quantum <= 0:
        return float("nan")
    return ctas / float(quantum)


def efficiency(ctas: int, quantum: int) -> float:
    """Полезная доля машины: сколько из запущенных волн заполнено.

    ctas / (квант * ceil(ctas/квант)).  При ctas=64 и кванте 80 это 0.8 -- то есть 16
    процессоров простаивают ЦЕЛИКОМ всю единственную волну.
    """
    if quantum <= 0 or ctas <= 0:
        return 0.0
    full = -(-ctas // quantum)
    return ctas / float(quantum * full)


def tail_share(ctas: int, quantum: int) -> float:
    """Доля времени, приходящаяся на неполную последнюю волну."""
    return 1.0 - efficiency(ctas, quantum)


def nearest_quantised(ctas: int, quantum: int) -> int:
    """Ближайшее сверху число блоков, дающее полные волны."""
    if quantum <= 0:
        return ctas
    return quantum * -(-ctas // quantum)
