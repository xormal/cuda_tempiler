#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""sm_70: ТЕНЗОРНЫЙ УЗЕЛ -- HMMA.884 и его ЗАМЕРЕННАЯ КАРТА ФРАГМЕНТА.

Карта нужна не для красоты: из неё core ВЫВОДИТ размер накопителя и закон плитки.  На Volta
это даёт «накопитель = MB*NB*8 float на поток» при плитке варпа 16*MB x 16*NB; на m16n8k16
(sm_80) число другое, и именно поэтому константы «*8» в core быть не может.

ДВЕ ЕДИНИЦЫ, И ОБЕ ОБЯЗАНЫ БЫТЬ НАЗВАНЫ (на этом наряд уже ошибся ВЧЕТВЕРО):
    SASS-единица  -- HMMA.884, одна КВАДРОПАРА 8x8x4, цена 2.00 такта на планировщик;
    ВАРПОВАЯ      -- одна mma.sync.m8n8k4, выданная варпом = ЧЕТЫРЕ квадропары = плотная
                     плитка 16x16, цена 8.00 такта на планировщик, 8 накопителей на полосу.
Ставка «тактов на команду» бессмысленна без имени единицы команды.  `TensorOp` описывает
ВАРПОВУЮ единицу (в ней сформулирован закон плитки); цена квадропары остаётся в CAP.TENSOR.

КАРТА ВЗЯТА ИЗ ОТГРУЖЕННОГО ЯДРА и проверена БИЕКЦИЕЙ (32 полосы x 8 накопителей = 256
различных ячеек = ровно 16x16), а не сверкой значений: наивная подача «полоса l даёт строку l»
считает ЧЕТВЕРТЬ плитки и ПРОХОДИТ сверку с torch, потому что сверка смотрит ровно те ячейки,
которые посчитаны.
"""

from __future__ import annotations

from ..base import FragmentMap, PluginCapabilityError, Rate, TensorOp
from .machine import Sm70Machine, data

_M = Sm70Machine()


# --------------------------------------------------------------------------------------------
# ЗАМЕРЕННАЯ КАРТА (volta_hmma.h боевого дерева; зонд tools/hmma_map_probe.py).
# Четыре квадропары ложатся на четыре квадранта ПЛОТНОЙ плитки 16x16.
# --------------------------------------------------------------------------------------------
def _a_row(l: int) -> int:
    return (l & 3) | ((l & 16) >> 2) | (l & 8)


def _b_col(l: int) -> int:
    return (l & 3) | ((l & 16) >> 2) | ((l & 4) << 1)


def _acc_row(l: int, r: int) -> int:
    return ((l & 1) | ((l & 16) >> 2) | (l & 8)) + 2 * ((r >> 1) & 1)


def _acc_col(l: int, r: int) -> int:
    return ((l & 2) | ((l & 4) << 1)) + (r & 1) + 4 * ((r >> 2) & 1)


def _frag_a(lane: int):
    """Полоса несёт 4 половинки подряд по k -- это ОДИН LDS.64; пара шагов даёт LDS.128."""
    r = _a_row(lane)
    return tuple((r, k, k) for k in range(4))


def _frag_b(lane: int):
    c = _b_col(lane)
    return tuple((k, c, k) for k in range(4))


def _frag_c(lane: int):
    """8 накопителей fp32 на полосу -- ИСТОЧНИК множителя '*8' в законе плитки."""
    return tuple((_acc_row(lane, r), _acc_col(lane, r), r) for r in range(8))


_FRAG = FragmentMap(a=_frag_a, b=_frag_b, c=_frag_c)


def bijection_ok() -> bool:
    """Покрытие плитки 16x16 картой C -- БИЕКЦИЯ.  Это и есть гейт корректности карты."""
    cells = {(_acc_row(l, r), _acc_col(l, r)) for l in range(32) for r in range(8)}
    return len(cells) == 256 and max(c[0] for c in cells) == 15 and max(c[1] for c in cells) == 15


def a_value_fanout() -> int:
    """Сколько полос получают ОДНО значение A. Замерено 2 -- отсюда неустранимый трафик Q."""
    from collections import Counter

    return max(Counter(_a_row(l) for l in range(32)).values())


def _warp_cost() -> Rate:
    w = data("tensor")["warp_op"]
    q = _M.rate("TENSOR.COST")
    return Rate(
        symbol="TENSOR.WARP_COST",
        value=float(w["cost_cycles_per_sched"]),
        units="такт/варповую-mma/планировщик",
        status=w["status"],
        prov=q.prov,
        note="%d квадропары x %.2f такта. %s" % (w["sass_per_op"], q.value, w["derivation"]),
    )


def _op() -> TensorOp:
    t = data("tensor")["op"]
    w = data("tensor")["warp_op"]
    return TensorOp(
        id=t["id"],
        m=w["m"],
        n=w["n"],
        k=w["k"],
        in_dtypes=tuple(t["in_dtypes"]),
        acc_dtype=t["acc_dtype"],
        frag=_FRAG,
        cost=_warp_cost(),
        operand_source=t["operand_source"],
        loader=t["loader"],
        exact_while=t["exact_while"],
    )


def _op_int8_as_fp16() -> TensorOp:
    """int8, уложенный в мантиссу fp16.  ТА ЖЕ ИНСТРУКЦИЯ, ТА ЖЕ ЦЕНА.

    Существует не ради ФЛОПов (их прибавки нет и быть не может: IMMA у sm_70 нет), а ради
    ТРАФИКА: байтовый вес вдвое режет чтение весов, а при M<=64 связывает именно чтение.
    Заявка «int8 быстрее fp16 по счёту» на этой машине ЛОЖНА -- отношение замерено 1.00
    на K = 256..16384.  Точность при этом ПОБИТОВАЯ: fp16 представляет целые до 2048 без
    ошибки, накопитель fp32 точен до 2**24.
    """
    t = data("tensor")["op"]
    w = data("tensor")["warp_op"]
    return TensorOp(
        id="HMMA.884.F32.F32/i8-in-mantissa",
        m=w["m"],
        n=w["n"],
        k=w["k"],
        in_dtypes=("int8", "fp16"),
        acc_dtype=t["acc_dtype"],
        frag=_FRAG,
        cost=_warp_cost(),
        operand_source=t["operand_source"],
        loader=t["loader"],
        exact_while=t["exact_while"],
    )


class Sm70TensorUnit:
    def ops(self):
        return (_op(), _op_int8_as_fp16())

    def select(self, in_dtypes, acc_dtype):
        want = tuple(in_dtypes)
        for o in self.ops():
            if o.in_dtypes == want and o.acc_dtype == acc_dtype:
                return o
        raise PluginCapabilityError(
            "sm_70 не имеет тензорной операции для операндов %r с накопителем %r; "
            "есть только %s. (IMMA появилась с sm_75; fp8/bf16 арифметики нет вовсе.)"
            % (want, acc_dtype, ", ".join(o.id for o in self.ops()))
        )
