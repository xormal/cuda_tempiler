#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ЗАГРУЗКА КАЛИБРОВКИ: ставки ЗАМЕРА подменяют ставки МОДЕЛИ.

Подмена разрешена только В ЭТУ СТОРОНУ и только при совпадении `plugin`. Обратное (модель
поверх замера) запрещено: это стирает единственное, ради чего замер делался.

Реализация ворот -- `tools/calib.py`; здесь стадия и правило подмены.
"""

from __future__ import annotations

import importlib.util
import os

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
TOOL = os.path.join(ROOT, "tools", "calib.py")


def tool():
    spec = importlib.util.spec_from_file_location("tempo_calib", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def apply(symbols, records):
    """Вернуть новую таблицу ставок, где MEASURED-записи подменили MODEL/NOT_MEASURED."""
    out = dict(symbols)
    for rec in records:
        for b in rec.get("binds", ()):
            sym = b["symbol"]
            if sym not in out:
                raise KeyError(
                    "калибровка привязывается к символу %r, которого НЕТ в закрытой таблице; "
                    "привязка к неизвестному символу -- отказ, а не расширение" % sym
                )
            old = out[sym]
            if old.status == "MEASURED" and rec.get("kind") != "rate":
                continue
            out[sym] = type(old)(
                symbol=old.symbol,
                value=float(b["value"]),
                units=old.units,
                status="MEASURED",
                prov=old.prov,
                note=b.get("note", old.note),
            )
    return out
