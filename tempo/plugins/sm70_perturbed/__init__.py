#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ВОЗМУЩЁННЫЙ sm_70 -- фальсификатор гейта G6.

Отличается от боевого плагина РОВНО ОДНОЙ СТАВКОЙ: ёмкость канала разделяемой памяти
удвоена (128 -> 256 байт за такт на процессор).  Всё остальное побитово то же.

ЧТО ГЕЙТ ТРЕБУЕТ.  Не «вывод изменился» -- этого мало, изменение могло быть случайным.
Требуется, чтобы вывод сдвинулся В ПРЕДСКАЗАННУЮ СТОРОНУ И НИ В КАКУЮ ДРУГУЮ:

    удвоение ёмкости канала => нагрузка на него ВДВОЕ МЕНЬШЕ => граница по этому каналу
    падает ровно вдвое, а прочие каналы НЕ ДВИГАЮТСЯ.

Если граница не сдвинулась -- ставка не участвует в выводе (она украшение).
Если сдвинулись и другие каналы -- отсекатель считает что-то своё поверх плагина.
"""

from __future__ import annotations

from ..base import CONTRACT, Channel, Rate, Report
from ..sm70 import Sm70Plugin

PERTURBED_SYMBOL = "CAP.MIO"
FACTOR = 2.0


class PerturbedMachine:
    """Прокси: всё как у боевой машины, кроме одной ставки."""

    def __init__(self, inner):
        self._inner = inner
        base = dict(inner.symbols())
        old = base[PERTURBED_SYMBOL]
        base[PERTURBED_SYMBOL] = Rate(
            symbol=old.symbol,
            value=old.value * FACTOR,
            units=old.units,
            status=old.status,
            prov=old.prov,
            note="ВОЗМУЩЕНО x%.0f ради гейта G6. Это НЕ замер этой машины." % FACTOR,
        )
        self._syms = base

    def symbols(self):
        return self._syms

    def rate(self, s):
        from ..base import closed_table_get

        return closed_table_get(self._syms, s)

    def channels(self):
        out = {}
        for name, ch in self._inner.channels().items():
            out[name] = Channel(
                name=ch.name, scope=ch.scope, capacity=self.rate("CAP." + name)
            )
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


class Sm70PerturbedPlugin(Sm70Plugin):
    id = "sm_70_perturbed"
    contract = CONTRACT
    description = (
        "sm_70 с ОДНОЙ удвоенной ставкой (%s x%.0f). Фальсификатор гейта G6."
        % (PERTURBED_SYMBOL, FACTOR)
    )

    def __init__(self):
        super().__init__()
        self.machine = PerturbedMachine(self.machine)

    def declared_stubs(self):
        return (
            "возмущённая копия sm_70: пригодна ТОЛЬКО для гейта, не для замеров",
        ) + tuple(super().declared_stubs())

    def selftest(self) -> Report:
        r = Report(self.id)
        base = Sm70Plugin()
        b = base.machine.symbols()
        p = self.machine.symbols()
        diff = [
            k
            for k in b
            if abs(b[k].value - p[k].value) > 1e-12
            or (b[k].value != b[k].value) != (p[k].value != p[k].value)
        ]
        r.check(
            "возмущена РОВНО ОДНА ставка", diff == [PERTURBED_SYMBOL], ", ".join(diff)
        )
        r.check(
            "возмущение ровно x%.0f" % FACTOR,
            abs(p[PERTURBED_SYMBOL].value - b[PERTURBED_SYMBOL].value * FACTOR) < 1e-9,
        )
        r.check(
            "возмущённая ставка ПОМЕЧЕНА как не относящаяся к машине",
            "ВОЗМУЩЕНО" in p[PERTURBED_SYMBOL].note,
        )
        return r


_P = None


def load():
    global _P
    if _P is None:
        _P = Sm70PerturbedPlugin()
    return _P
