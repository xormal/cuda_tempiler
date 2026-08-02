#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ГРАНИЦА ПЕРИОДА.  Ни одного имени архитектуры, ни одного вшитого числа.

    ResMII = max по каналам ( нагрузка канала / его ёмкость )   -- «объём не влезает»
    RecMII = max по циклам графа зависимостей ( сумма задержек / число оборотов )
    T >= max(ResMII, RecMII)

Это в точности нижняя граница циклического расписания.  ДВА ПРЕДУПРЕЖДЕНИЯ, без которых
число врёт:

1. МОДЕЛЬ -- ГРАНИЦА, А НЕ ПРЕДСКАЗАНИЕ.  Замерено: боевое ядро идёт на 61 % СВОЕГО ЖЕ
   счётного потолка; четыре правки уронили счёт команд и НЕ уронили время.  Вердикт всегда
   двойной: граница + фазовый замер.

2. МОДЕЛЬ ОБЯЗАНА УМЕТЬ МОЛЧАТЬ.  Если связывающий ресурс не представлен ни одним каналом
   (замеренный случай: при малом числе строк выхода связывает ЗАДЕРЖКА при низкой занятости,
   а такого канала в модели нет), функция возвращает `silent=True`, и отчёт печатает
   «МОДЕЛЬ МОЛЧИТ», а не тихо неверную границу.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from tempo.plugins.base import Rate, spec_forbidden_in_time_bound


@dataclass
class Bound:
    T: float
    res_mii: float
    rec_mii: float
    binding: str  # имя канала либо "RECURRENCE" либо "" при молчании
    per_channel: dict = field(default_factory=dict)
    silent: bool = False
    silent_reason: str = ""
    uses_not_measured: tuple = ()

    def render(self) -> str:
        if self.silent:
            return "МОДЕЛЬ МОЛЧИТ: %s" % self.silent_reason
        head = "T >= %.2f такта (ResMII %.2f, RecMII %.2f), связывает %s" % (
            self.T,
            self.res_mii,
            self.rec_mii,
            self.binding,
        )
        if self.uses_not_measured:
            head += (
                "\n  ВНИМАНИЕ: вывод опирается на НЕ ЗАМЕРЕННЫЕ ставки: "
                + ", ".join(self.uses_not_measured)
            )
        return head


def res_mii(atoms, channels, warps_per_sm: int = 1, schedulers: int = 1):
    """Нагрузка канала / его ёмкость.

    scope="sm" -- ресурс ВСЕГО мультипроцессора: его нагрузка домножается на число варпов.
    scope="sched" -- ресурс планировщика: ёмкость объявлена в тактах НА КОМАНДУ, поэтому
    нагрузка есть произведение числа команд на ставку.

    ЧТО ЗДЕСЬ НЕЛЬЗЯ ПРОЧЕСТЬ КАК ВЫВОД (закон L-OCCUPANCY-MOVES-BINDING, data/laws/method.json).
    Прежняя редакция этой строки утверждала, что связывающий ресурс МЕНЯЕТСЯ с занятостью.
    Замер это ОПРОВЕРГ: прогон одного тела при 8/16/32 варпах не сдвинул доли каналов ни на
    процент, потому что в модели стенда ОБА вида канала линейны по числу варпов и их отношение
    постоянно.  Здесь арифметика ДРУГАЯ (варпы входят только в канал процессора), и это
    расхождение двух моделей одного дерева -- НАЗВАННЫЙ ДОЛГ, а не подтверждение старого
    вывода: см. запись L-TWO-MODELS-OCCUPANCY.  Связывающий ресурс открывают, а не назначают.
    """
    from tempo.core.ir.atom import channel_load

    load = channel_load(atoms)
    per = {}
    not_measured = []
    for name, cyc in load.items():
        ch = channels.get(name)
        if ch is None:
            continue
        cap = ch.capacity
        spec_forbidden_in_time_bound(cap)
        if cap.status == "NOT_MEASURED":
            not_measured.append(cap.symbol)
            continue
        if ch.scope == "sm":
            # один ресурс на все планировщики: нагрузка умножается на число варпов
            per[name] = (cyc * max(1, warps_per_sm)) / max(1e-12, cap.value)
        else:
            per[name] = cyc * cap.value
    return per, tuple(not_measured)


def rec_mii(atoms) -> float:
    """Максимум по циклам графа зависимостей отношения (сумма задержек)/(число оборотов).

    Это МАКСИМАЛЬНЫЙ СРЕДНИЙ ЦИКЛ.  Точное решение при фиксированной сериализации -- Карп;
    здесь используется реализация из tempo/core/graph/graphs.py (46/46, дифф-тест против
    перебора ВСЕХ простых контуров).
    """
    idx = {a.uid: i for i, a in enumerate(atoms)}
    edges = []
    for a in atoms:
        for d in a.deps:
            if d.src_uid not in idx:
                continue
            # ребро src -> a весом (задержка src + минимальный зазор), с оборотом distance
            src = atoms[idx[d.src_uid]]
            w = float(src.latency) + float(d.min_gap)
            edges.append((idx[d.src_uid], idx[a.uid], w, max(0, int(d.distance))))
    if not edges:
        return 0.0
    cycles_edges = [
        (u, v, Fraction(w).limit_denominator(10**6))
        for u, v, w, dist in edges
        if dist > 0
    ]
    if not cycles_edges:
        return 0.0
    try:
        from tempo.core.graph.graphs import max_mean_cycle
    except Exception:
        return 0.0
    n = len(atoms)
    ew = [(u, v) for u, v, _ in cycles_edges]
    w = [float(x) for _, _, x in cycles_edges]
    try:
        lam, _cycle = max_mean_cycle(n, ew, weight=w)
    except Exception:
        return 0.0
    return float(lam) if lam is not None else 0.0


def bound(atoms, channels, warps_per_sm: int = 1, schedulers: int = 1) -> Bound:
    per, nm = res_mii(atoms, channels, warps_per_sm, schedulers)
    if not per:
        return Bound(
            T=float("nan"),
            res_mii=float("nan"),
            rec_mii=float("nan"),
            binding="",
            silent=True,
            silent_reason="связывающий ресурс не представлен ни одним каналом с ЗАМЕРЕННОЙ ставкой"
            + (" (не замерены: %s)" % ", ".join(nm) if nm else ""),
            uses_not_measured=nm,
        )
    binding = max(per, key=per.get)
    r = per[binding]
    rec = rec_mii(atoms)
    T = max(r, rec)
    return Bound(
        T=T,
        res_mii=r,
        rec_mii=rec,
        binding=binding if r >= rec else "RECURRENCE",
        per_channel=per,
        uses_not_measured=nm,
    )


def headroom(measured_cycles: float, b: Bound) -> float:
    """ЗАПАС: во сколько раз замер выше границы.  Это и есть «разрыв до планировщика».

    Замеренный порядок величины запаса на боевых телах -- 1.6x (ядро на 61 % своего потолка).
    Запас < 1 означает НЕ «модель побеждена», а ошибку в единице: ровно так был пойман
    вчетверо завышенный закон.
    """
    if b.silent or not b.T or b.T != b.T:
        return float("nan")
    return measured_cycles / b.T
