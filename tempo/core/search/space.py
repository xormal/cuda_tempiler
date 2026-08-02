#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ПРОСТРАНСТВО ПОИСКА = variants() плагина x объявленные им ОСИ + фильтры конвейера.

Конвейер НЕ ЗНАЕТ, что значит каждая ось.  Он знает только: у оси есть имя и множество
ресурсов, которые она двигает (`Axis.affects`).  Этого достаточно, чтобы (а) перечислить,
(б) объяснить в отчёте, почему вариант оставлен, (в) на новой архитектуре принять НОВУЮ ось
без единой правки здесь.
"""

from __future__ import annotations


def enumerate_variants(plugin, op, limit=None):
    out = []
    for i, h in enumerate(plugin.skeletons.variants(op)):
        out.append(h)
        if limit is not None and len(out) >= limit:
            break
    return out


def axes_table(plugin) -> str:
    rows = ["ОСИ ПОИСКА, ОБЪЯВЛЕННЫЕ ПЛАГИНОМ %s" % plugin.id]
    for a in plugin.skeletons.axes():
        rows.append("  %-14s двигает: %s" % (a.name, ", ".join(sorted(a.affects)) or "-"))
    return "\n".join(rows)


def touching(plugin, resource: str):
    """Оси, двигающие названный ресурс.  Так отчёт объясняет выбор, не зная архитектуры."""
    return tuple(a.name for a in plugin.skeletons.axes() if resource in a.affects)
