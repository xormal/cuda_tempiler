#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""СХЕМА КАЛИБРОВКИ `tempo/calib/2` = прежняя `/1` + ОБЯЗАТЕЛЬНОЕ поле `plugin`.

Ровно одно изменение относительно версии 1, и оно вынужденное: запись замера обязана знать,
к КАКОЙ МАШИНЕ она относится, иначе ставка, снятая на одной архитектуре, подменит ставку на
другой -- молча и с правдоподобным числом.

Проверка записей -- в `tools/calib.py` (самопроверка 18/18, все ворота проверены
СРАБАТЫВАНИЕМ). Второй реализации ворот здесь нет.
"""

from __future__ import annotations

SCHEMA = "tempo/calib/2"
REQUIRED = ("id", "kind", "quantity", "units", "taken_with", "shape", "card",
            "observability", "provenance", "plugin")
KINDS = ("padcurve", "phases", "law_points", "wf_per_inst", "conflict_share", "rate")
OBSERVABILITY = ("clean", "code_edit", "derived")


def validate(rec: dict) -> None:
    for f in REQUIRED:
        if f not in rec:
            raise ValueError("запись калибровки без поля %r (схема %s)" % (f, SCHEMA))
    if rec["kind"] not in KINDS:
        raise ValueError("вид записи %r вне %r" % (rec["kind"], KINDS))
    if rec["observability"] not in OBSERVABILITY:
        raise ValueError("observability %r вне %r" % (rec["observability"], OBSERVABILITY))
    card = rec.get("card") or {}
    if card.get("foreign_procs", 1) != 0:
        raise ValueError("замер снят при чужих процессах на карте -- недействителен")
    if "license" not in rec:
        rec["license"] = "LicenseRef-TRL-1.0"   # умолчание; для присланных -- явное иное
