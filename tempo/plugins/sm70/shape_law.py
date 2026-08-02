#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""sm_70: ЗАКОНЫ ФОРМЫ -- ЧТО СВЯЗЫВАЕТ при данном M и ДО СКОЛЬКИХ БИТ сжимать вес.

ТРИ ВЕРДИКТА, И КАЖДЫЙ ОТКАЗЫВАЕТ ВНЕ СВОЕЙ ОБЛАСТИ:
    binding(shape, M)        LAW=L-MRIDGE-128    ПОЛОСА / СЧЁТ / МОЛЧИТ
    bit_floor(shape, M, c)   LAW=L-BIT-FLOOR     дно разрядности, ДВА коэффициента
    narrow_ok(shape, M)      LAW=L-NARROW-0644   применим ли узкий формат

ПОЧЕМУ ЭТО ЖИВЁТ В ПЛАГИНЕ, А НЕ В КОНВЕЙЕРЕ.  Это ВЫВОДЫ, полученные на конкретных телах
конкретной машины, и через границу они не ходят (`docs/CONTRACT.md`, раздел «что через границу
НЕ ходит»).  Правило «при малом M предпочитать узкий вес» внутри `core/search/prune.py` было бы
протечкой и ловилось бы гейтами G6+G8.  Здесь они -- данные плагина с ОБЛАСТЬЮ.

ПОЧЕМУ ВЕРДИКТ, А НЕ ОСЬ ОТСЕЧЕНИЯ.  Канала внекристальной подачи в таблице ставок НЕТ
(`docs/NOT_YET.md` §14), поэтому отсекать по этому закону значило бы подменить отсутствующий
канал эвристикой -- ровно то, за что в этом дереве отвечает слово «назначить».  Вердикт
ГОВОРИТ и требует замера; отсечение МОЛЧА ВЫБРАСЫВАЕТ.  Разница дороже удобства.

ОБЛАСТЬ (нарушать её -- главная ошибка дня, повторившаяся четырежды):
  * формы -- ПЯТЬ боевых матриц; для ОДНОЙ матрицы берётся ЕЁ строка, а не строка `gate,up`;
  * M -- замерено в четырёх точках {32, 64, 128, 4096}; между ними ИНТЕРПОЛЯЦИЯ, и она
    называется вслух; вне [32, 4096] -- ОТКАЗ, а не экстраполяция;
  * дно разрядности замерено при M<=64; при M=128 оно уже около 16 бит, то есть не окупается
    даже байт -- и это независимо воспроизводит перелом M=128.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from ..base import PluginCapabilityError

_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "machine", "shape.json"
)
_D = None


def data() -> dict:
    global _D
    if _D is None:
        with open(_DATA, encoding="utf-8") as f:
            _D = json.load(f)
    return _D


def shapes() -> tuple:
    return tuple(data()["shapes"])


def ridge_m() -> int:
    return int(data()["ridge_m"]["value"])


def _shape(name: str) -> dict:
    d = data()["shapes"].get(name)
    if d is None:
        raise PluginCapabilityError(
            "формы %r среди замеренных нет; замерены %s. Перенос закона на незамеренную форму "
            "-- ровно та ошибка, которая за сутки повторилась четырежды: величина верна, "
            "область не та." % (name, ", ".join(shapes()))
        )
    return d


def _interp(tbl: dict, M: float):
    """Линейная интерполяция по ЛОГАРИФМУ M между замеренными точками.

    Возвращает (значение, точно_ли_замерено).  Вне замеренного размаха -- ОТКАЗ: экстраполяция
    закона за его точки и есть способ получить уверенное неверное число.
    """
    xs = sorted(int(k) for k in tbl)
    if M < xs[0] or M > xs[-1]:
        raise PluginCapabilityError(
            "M=%g вне замеренного размаха [%d, %d]: закон формы за своими точками НЕ "
            "экстраполируется. Домерить -- один прогон развёртки по M на свободной карте."
            % (M, xs[0], xs[-1])
        )
    for x in xs:
        if abs(x - M) < 1e-9:
            return float(tbl[str(x)]), True
    import math

    lo = max(x for x in xs if x < M)
    hi = min(x for x in xs if x > M)
    t = (math.log(M) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return float(tbl[str(lo)]) * (1 - t) + float(tbl[str(hi)]) * t, False


class ShapeVerdict(object):
    """Вердикт формы.  `kind` -- ASCII-код, `explain` -- по-русски, для отчёта."""

    KINDS = ("ПОЛОСА", "СЧЁТ", "МОЛЧИТ")

    def __init__(self, kind, bw, comp, exact, explain):
        if kind not in self.KINDS:
            raise PluginCapabilityError("код вердикта %r вне %r" % (kind, self.KINDS))
        self.kind, self.bw, self.comp, self.exact, self.explain = (
            kind,
            bw,
            comp,
            exact,
            explain,
        )

    def __repr__(self):
        return "ShapeVerdict(%s, полоса=%.3f, счёт=%.3f, %s)" % (
            self.kind,
            self.bw,
            self.comp,
            "замер" if self.exact else "интерполяция",
        )


# ЕДИНСТВЕННОЕ МЕСТО, ГДЕ РЕШАЕТСЯ «ЗНАЧИМА ЛИ РАЗНИЦА ДОЛЕЙ».
# Величина та же, что у переукладчика (разрешающая способность модели 14.7 %): доли,
# расходящиеся меньше, чем модель различает, НЕ дают права называть связывающий ресурс.
BAND = 0.147


def binding(shape: str, M: float) -> ShapeVerdict:
    """LAW=L-MRIDGE-128.  Что связывает линейный слой при данном числе строк выхода.

    При M<=64 замерено 78-90 % достижимой полосы при 12-47 % счёта; при M>=256 -- счёт;
    перелом на M=128.  ВАЖНО: у `k,v` доли сравнялись уже при M=128 (40.4 против 42.4), то есть
    перелом -- свойство НАБОРА форм, а на отдельной форме он сдвинут. Поэтому вердикт даётся ПО
    ДОЛЯМ ЭТОЙ формы, а не по сравнению M со 128.
    """
    s = _shape(shape)
    bw, e1 = _interp(s["bw_share"], M)
    comp, e2 = _interp(s["comp_share"], M)
    exact = e1 and e2
    tail = (
        ""
        if exact
        else (
            " ВНИМАНИЕ: M=%g между замеренными точками, доли ИНТЕРПОЛИРОВАНЫ по логарифму -- "
            "это оценка, а не замер." % M
        )
    )
    if bw > comp * (1.0 + BAND):
        return ShapeVerdict(
            "ПОЛОСА",
            bw,
            comp,
            exact,
            "%s при M=%g: %.1f %% достижимой полосы против %.1f %% счёта -- "
            "связывает ЧТЕНИЕ ВЕСОВ. Узкий вес здесь ПЛАТИТ.%s"
            % (shape, M, 100 * bw, 100 * comp, tail),
        )
    if comp > bw * (1.0 + BAND):
        return ShapeVerdict(
            "СЧЁТ",
            bw,
            comp,
            exact,
            "%s при M=%g: %.1f %% счёта против %.1f %% полосы -- связывает СЧЁТ. "
            "Сужение формата здесь платит ВРЕМЕНЕМ, а не покупает его.%s"
            % (shape, M, 100 * comp, 100 * bw, tail),
        )
    return ShapeVerdict(
        "МОЛЧИТ",
        bw,
        comp,
        exact,
        "%s при M=%g: полоса %.1f %% против счёта %.1f %% -- разрыв меньше "
        "разрешающей способности (%.0f %%). МОДЕЛЬ МОЛЧИТ: назвать связывающий "
        "ресурс здесь значит назначить его. Домерить парными отношениями.%s"
        % (shape, M, 100 * bw, 100 * comp, 100 * BAND, tail),
    )


def bit_floor(shape: str, M: float, chain_ops: float = 0.0):
    """LAW=L-BIT-FLOOR.  Дно разрядности: ниже него время НЕ падает.

    Возвращает (b_форма, b_калиброванная_модель, b_грубая_крыша, пояснение).

    ТРИ ЧИСЛА, А НЕ ОДНО, И ЭТО НЕ ИЗБЫТОЧНОСТЬ.  Первое -- ЗАМЕР ЭТОЙ формы (брать его).
    Второе -- откалиброванная модель выдачи (наклон 0.1236), третье -- грубая крыша без модели
    выдачи вовсе (наклон 0.144).  Два наклона расходятся на 16 %, и оба обязаны быть названы:
    сузить их до одного значило бы выдать незнание за точность.
    """
    s = _shape(shape)
    m = data()["bit_floor_model"]
    cal = m["calibrated"]["slope"] * M + m["calibrated"]["chain_coeff"] * chain_ops
    crude = m["crude"]["slope"] * M
    try:
        own, exact = _interp(s["bit_floor"], M)
        src = "ЗАМЕР этой формы" if exact else "интерполяция по замерам этой формы"
    except PluginCapabilityError:
        own, src = None, "вне размаха замеров дна (32..64) -- своей строки НЕТ"
    expl = (
        "%s при M=%g: дно %s (%s); модель выдачи даёт %.2f бита, грубая крыша -- %.2f "
        "(наклоны 0.1236 и 0.144 расходятся на %.1f %%, оба названы намеренно). "
        "ОБЛАСТЬ: дно посчитано для ядра, берущего ТУ ЖЕ долю обеих крыш, что сильная "
        "библиотека; у ПЛОХОГО ядра дно НИЖЕ, и это дефект реализации, а не ресурс."
        % (
            shape,
            M,
            ("%.2f бита" % own) if own is not None else "не замерено",
            src,
            cal,
            crude,
            m["disagreement_pct"],
        )
    )
    return own, cal, crude, expl


def narrow_ok(shape: str, M: float, compute_cost: Optional[float] = None):
    """LAW=L-NARROW-0644.  Применим ли формат УЖЕ половинной точности к этой фазе.

    Правило одной строкой: **узкий формат применим тогда и только тогда, когда фазу связывает
    ПОЛОСА.** Числом: новое время = max(0.75*T_памяти, 1.165*T_счёта), значит выигрыш есть,
    пока T_счёта/T_памяти < 0.644.

    Возвращает (можно ли, отношение, пояснение).  `None` в первом поле означает МОЛЧАНИЕ: доли
    внутри разрешающей способности, и ответа у модели нет.
    """
    nf = data()["narrow_format"]
    thr = nf["threshold"]["value"]
    cost = nf["compute_cost"]["value"] if compute_cost is None else compute_cost
    v = binding(shape, M)
    ratio = (v.comp / v.bw) if v.bw > 0 else float("inf")
    if v.kind == "МОЛЧИТ":
        return (
            None,
            ratio,
            (
                "МОДЕЛЬ МОЛЧИТ: %s Пока не названо, ЧТО связывает фазу, вердикта о формате нет -- "
                "и это отказ, а не осторожность: за сутки трижды сужение формата не купило времени "
                "именно там, где связывающий ресурс не был назван." % v.explain
            ),
        )
    ok = ratio < thr
    return (
        ok,
        ratio,
        (
            "%s Отношение счёт/полоса = %.3f против порога %.3f => узкий формат %s. "
            "Экономия байт 0.750, цена тензорной фазы x%.3f (при MB=4). ОГОВОРКА: порог СДВИГАЕТСЯ "
            "с ростом плитки по M -- распаковка амортизируется как 1/M (LAW=L-UNPACK-AMORT-1M)."
            % (
                v.explain,
                ratio,
                thr,
                "ПРИМЕНИМ" if ok else "ПЛАТИТ ВРЕМЕНЕМ, а не покупает его",
                cost,
            )
        ),
    )
