#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""СЧЁТЧИКИ: стадия конвейера.  ИМЕНА СЧЁТЧИКОВ ЖИВУТ У ПЛАГИНА, здесь их нет ни одного.

ПРАВИЛО ПЕРЕВОДА, без которого счётчик врёт (оплачено сутками):

    прибор меряет РЕСУРС.
    СВЯЗЫВАНИЕ доказывает только СДВИГ ВРЕМЕНИ от снятия ресурса.
    КУРС между ними = сколько раз отрабатывает место.

Замерено: снятие 18.4 % трафика разделяемой в ЭПИЛОГЕ дало 0.2 % времени, а 14.5 % в ТЕЛЕ
ЦИКЛА -- 7.5 % (6/6 форм).  Один и тот же процент ресурса стоил в 37 раз разного времени.
Поэтому вывод «связывает X» НИКОГДА не делается по одному показанию прибора.

И ловушка локали: под русской локалью разбор «5 238» падает и МОЛЧА теряет все значения
>= 1000.  Инструмент разбирает с LC_ALL=C -- это его забота, но знать об этом обязан каждый.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResourceReading:
    kind: str
    value: float
    units: str
    plugin_id: str

    def render(self) -> str:
        return "%s = %.4g %s (%s)" % (self.kind, self.value, self.units, self.plugin_id)


def read(plugin, binary, kernel, kind):
    """Спросить у плагина имена, снять прибором, вернуть показание с указанием плагина."""
    names = plugin.meters.counters(kind)
    raw = plugin.meters.profile(binary, kernel, names)
    total = sum(float(v) for v in raw.values())
    return ResourceReading(kind=kind, value=total, units="ед.", plugin_id=plugin.id)


def binding_requires_time_shift(reading, t_base: float, t_without: float) -> str:
    """Единственный законный способ назвать связывающий ресурс."""
    if t_base <= 0:
        return "нет базы времени -- утверждать о связывании НЕЛЬЗЯ"
    share = 1.0 - t_without / t_base
    return (
        "ресурс %s: показание %.4g; снятие даёт %.1f %% ВРЕМЕНИ. "
        "Связывающим он называется по ВТОРОМУ числу, не по первому."
        % (reading.kind, reading.value, 100.0 * share)
    )
