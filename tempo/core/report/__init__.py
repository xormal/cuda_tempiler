#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ОТЧЁТ.  Две планки ВСЕГДА; статус каждой ставки; состояние карты; и умение МОЛЧАТЬ.

ЧЕТЫРЕ ПРАВИЛА, КОТОРЫЕ ОТЧЁТ ВЫНУЖДАЕТ СОБЛЮДАТЬ (иначе он не собирается):
  1. ДВЕ ПЛАНКИ.  «Во сколько раз обогнали вход» без второй планки -- самообман; вторая без
     первой не отвечает, продукт ли это сделал.
  2. ТРЕТЬЕ ЧИСЛО -- ДОЛЯ РАБОТЫ В СТЕНЕ.  Ядро дало x1.36 изолированно и 0.6 % сквозняком.
  3. ПРОИСХОЖДЕНИЕ.  Число без метки происхождения -> отчёт НЕ СОБИРАЕТСЯ.
  4. МОДЕЛЬ ВПРАВЕ МОЛЧАТЬ.  Если связывающий ресурс не представлен, печатается
     «МОДЕЛЬ МОЛЧИТ», а НЕ тихо неверная граница.  Долг оформляется с формой лечения.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tempo.core.measure.baseline import Comparison, two_bars_required, wall_share_required
from tempo.core.report.provenance import require_provenance


@dataclass
class Section:
    title: str
    lines: list = field(default_factory=list)

    def add(self, s):
        self.lines.append(s)
        return self


@dataclass
class Report:
    title: str
    plugin_id: str
    card: object = None
    sections: list = field(default_factory=list)
    comparisons: list = field(default_factory=list)
    wall_share: float = None
    bound: object = None
    rates_used: list = field(default_factory=list)

    def section(self, title) -> Section:
        s = Section(title)
        self.sections.append(s)
        return s

    def compare(self, c: Comparison):
        self.comparisons.append(c)
        return self

    def render(self) -> str:
        require_provenance(self)
        two_bars_required(self.comparisons)
        out = ["=" * 96, self.title, "=" * 96]

        # ШАПКА: карта, частоты, соседи. Без них число недействительно.
        if self.card is None:
            out.append("КАРТА НЕ УКАЗАНА -- всякое число ниже НЕДЕЙСТВИТЕЛЬНО как замер")
        else:
            out.append(
                "карта %s, частота %s МГц, чужих процессов %s, дата %s"
                % (self.card.index, self.card.clock_mhz, self.card.foreign_procs, self.card.date)
            )
            if self.card.foreign_procs:
                out.append("ЗАМЕР С СОСЕДОМ НА КАРТЕ НЕДЕЙСТВИТЕЛЕН")
        out.append("плагин: %s" % self.plugin_id)

        # СТАВКИ, на которые опирается вывод
        nm = [r for r in self.rates_used if getattr(r, "status", "") == "NOT_MEASURED"]
        if nm:
            out.append(
                "ВЫВОД ОПИРАЕТСЯ НА НЕ ЗАМЕРЕННЫЕ СТАВКИ: " + ", ".join(r.symbol for r in nm)
            )

        # ГРАНИЦА (или её отсутствие)
        if self.bound is not None:
            out.append("")
            out.append(self.bound.render())

        # ДВЕ ПЛАНКИ
        out.append("")
        out.append("ПЛАНКИ:")
        for c in self.comparisons:
            out.append("  " + c.render())
        out.append("  " + wall_share_required(self.wall_share))

        for s in self.sections:
            out.append("")
            out.append(s.title)
            for ln in s.lines:
                out.append("  " + str(ln))
        return "\n".join(out)


DEBTS = {
    "latency_at_low_occupancy": (
        "КАНАЛА «задержка при низкой занятости» В МОДЕЛИ НЕТ, а при малом числе строк выхода "
        "связывает именно он. ФОРМА ЛЕЧЕНИЯ НАЗВАНА: граница Литтла T >= L*N_треб/MLP_max; "
        "ставка NOT_MEASURED, калибруется первым же замером в этом режиме."
    ),
    "conflict_input": (
        "Конфликтность -- ВХОД из карты адресов, а не константа. Для порождаемого ядра модель "
        "точна (адреса известны эмиттеру); для чтения ЧУЖОГО кода постфактум -- нет."
    ),
    "phase_residual": (
        "Невязка фазового разложения не разделена на перекрытие и неназванное, пока нет "
        "варианта «снять ВСЁ». Доли -- НИЖНИЕ оценки."
    ),
}
