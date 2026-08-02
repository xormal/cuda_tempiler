#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""АТОМ -- единственное представление, которым оперирует конвейер в версии 1.

Атом не «инструкция».  Единица анализа -- ФАЗА: множество операций, занимающее канал на
промежутке.  Счёт команд четырежды дал правку, уронившую счёт и НЕ уронившую время (одна --
на 4.8 % хуже), и противоположный порядок форматов.  Поэтому атом несёт ОТПЕЧАТОК по
каналам, а не имя команды: имя (`op_id`) для конвейера непрозрачно.

Чего здесь НЕТ намеренно: подъёма представления из машинного кода, сериализуемых регионов и
раскладок.  Версия 1 ПОРОЖДАЕТ ядро по спецификации, а не поднимает его; полное
представление нужно стадии, которой в версии 1 нет.
"""

from __future__ import annotations

from tempo.plugins.base import Atom, AtomKind, Dep  # noqa: F401  (единственное место определения)

__all__ = ["Atom", "AtomKind", "Dep", "channel_load", "region_of", "summary"]


def channel_load(atoms) -> dict:
    """Суммарная занятость каждого канала одной итерацией."""
    out = {}
    for a in atoms:
        for ch, cyc in a.footprint.items():
            out[ch] = out.get(ch, 0.0) + float(cyc)
    return out


def region_of(atoms, region: str):
    return [a for a in atoms if a.region == region]


def summary(atoms) -> str:
    load = channel_load(atoms)
    top = sorted(load.items(), key=lambda kv: -kv[1])
    return "атомов %d; каналы: %s" % (
        len(atoms),
        ", ".join("%s=%.1f" % (k, v) for k, v in top),
    )
