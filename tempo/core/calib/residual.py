#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ОТЧЁТ О НЕВЯЗКЕ -- ГЛАВНЫЙ ВЫХОД КАЛИБРОВКИ, а не побочный.

Калибровка ценна не тем, что подставила число, а тем, что показала, ГДЕ модель расходится с
замером и НА СКОЛЬКО. Ставка, подставленная без отчёта о невязке, прячет ошибку модели внутрь
самой модели.

Правило чтения: невязка, которую удалось обнулить ПОДБОРОМ ставки, -- не подтверждение
модели. Модель подтверждается ПРЕДСКАЗАНИЕМ на теле, которого она не видела.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Residual:
    rows: list = field(default_factory=list)  # (что, замер, модель)

    def add(self, what, measured, model):
        self.rows.append((what, float(measured), float(model)))

    def render(self) -> str:
        if not self.rows:
            return (
                "ОТЧЁТ О НЕВЯЗКЕ ПУСТ: калибровка ничего не сверила -- это НЕ 'сошлось'"
            )
        out = ["ОТЧЁТ О НЕВЯЗКЕ (замер против модели)"]
        worst = 0.0
        for what, meas, mod in self.rows:
            d = 0.0 if meas == 0 else 100.0 * (mod - meas) / meas
            worst = max(worst, abs(d))
            out.append(
                "  %-40s замер %10.4g  модель %10.4g  невязка %+7.2f %%"
                % (what, meas, mod, d)
            )
        out.append("  максимум |невязки|: %.2f %%" % worst)
        out.append(
            "  НАПОМИНАНИЕ: невязка, обнулённая ПОДБОРОМ ставки, модель не подтверждает. "
            "Подтверждает предсказание на теле, которого модель не видела."
        )
        return "\n".join(out)
