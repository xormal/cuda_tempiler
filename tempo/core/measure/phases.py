#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ФАЗОВЫЙ ФАЛЬСИФИКАТОР: доля фазы = 1 - t(фаза снята) / t(база).

ЕДИНИЦА АНАЛИЗА -- ФАЗА, А НЕ КОМАНДА.  Фаза -- это множество операций, занимающее канал на
промежутке.  Счёт команд четырежды дал правку, уронившую счёт и НЕ уронившую время (одна --
на 4.8 % ХУЖЕ), и противоположный порядок вариантов.  Сперва разложение по фазам, потом счёт.

ЧЕСТНОСТЬ РАЗЛОЖЕНИЯ.  Доли, полученные СНЯТИЕМ, -- НИЖНИЕ ОЦЕНКИ: снятие меняет перекрытие
и освобождает общий канал.  Поэтому отчёт обязан печатать не только сумму, но и НЕВЯЗКУ:

    1 = сумма долей + ПЕРЕКРЫТИЕ + НЕНАЗВАННОЕ

и пока нет варианта «снять ВСЁ», перекрытие и неназванное НЕ РАЗДЕЛЕНЫ.  Замеренный пример:
сумма названных фаз 77.7 %, невязка 22.3 % не разделена.

Реализация одна -- `tools/phaseprof.py` (947 строк, арифметика долей проверена якорем 5/5).
Здесь -- стадия и арифметика невязки.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
TOOL = os.path.join(ROOT, "tools", "phaseprof.py")


def tool():
    spec = importlib.util.spec_from_file_location("tempo_phaseprof", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@dataclass
class Decomposition:
    base_time: float
    phases: dict = field(default_factory=dict)  # имя -> время БЕЗ этой фазы
    all_removed: float = float("nan")  # время варианта «снять ВСЁ», если он есть

    def share(self, name: str) -> float:
        return 1.0 - self.phases[name] / self.base_time

    def shares(self) -> dict:
        return {k: self.share(k) for k in self.phases}

    def residual(self) -> float:
        return 1.0 - sum(self.shares().values())

    def split_residual(self):
        """Разделить невязку на ПЕРЕКРЫТИЕ и НЕНАЗВАННОЕ.  Возможно ТОЛЬКО при «снять ВСЁ»."""
        if self.all_removed != self.all_removed:
            return None, None
        named_total = 1.0 - self.all_removed / self.base_time
        overlap = named_total - sum(self.shares().values())
        unnamed = 1.0 - named_total
        return overlap, unnamed

    def render(self) -> str:
        out = ["РАЗЛОЖЕНИЕ ПО ФАЗАМ (доля = 1 - t(снята)/t(база))"]
        for k, v in sorted(self.shares().items(), key=lambda kv: -kv[1]):
            out.append("  %-24s %6.1f %%" % (k, 100.0 * v))
        out.append(
            "  %-24s %6.1f %%"
            % ("сумма названных", 100.0 * sum(self.shares().values()))
        )
        ov, un = self.split_residual()
        if ov is None:
            out.append(
                "  НЕВЯЗКА %.1f %% НЕ РАЗДЕЛЕНА: варианта «снять ВСЁ» нет, поэтому перекрытие "
                "и неназванное не различены. Доли -- НИЖНИЕ оценки."
                % (100.0 * self.residual())
            )
        else:
            out.append("  %-24s %6.1f %%" % ("перекрытие", 100.0 * ov))
            out.append("  %-24s %6.1f %%" % ("неназванное", 100.0 * un))
        return "\n".join(out)
