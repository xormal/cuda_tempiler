#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""sm_70: ЛЕГАЛЬНЫЕ ГИПЕРФОРМЫ и ОСИ ПОИСКА.

Здесь живёт то, что core знать не имеет права: какие BM/BN/BK/варпы вообще собираются на
этой машине, и какие «стены» отсекают часть произведения ДО сборки.

СТЕНЫ cutlass -- СВОЙСТВО БИБЛИОТЕКИ, А НЕ ЖЕЛЕЗА, и потому стоят здесь ФИЛЬТРАМИ, а не
законами.  Замерено трижды: сообщение вида «Number of iterations must be non-zero» всегда
оказывалось ЦЕЛОЧИСЛЕННЫМ ДЕЛЕНИЕМ, дошедшим до нуля, и все три случая чинились заменой
константы на закон.  Правило: увидев такое сообщение, не искать «поддерживается ли форма», а
выписать формулу параметра и найти деление, давшее ноль.
"""

from __future__ import annotations

from typing import Iterator

from ..base import Axis, Hyperform, OpSpec, PluginCapabilityError
from .gemm_bound import Hyperform as GemmHyperform

# --------------------------------------------------------------------------------------------
# ОСИ.  ОТКРЫТЫЙ словарь: core перебирает их механически и НЕ знает, что каждая значит.
# На sm_80 сюда добавится async_depth -- ноль правок в core.
# --------------------------------------------------------------------------------------------
AXES = (
    Axis("block_tile", frozenset({"MIO", "smem", "hbm"})),
    Axis("warp_tile", frozenset({"MIO", "regs"})),
    Axis("stages", frozenset({"smem", "regs", "long_scoreboard"})),
    Axis("gstage", frozenset({"regs", "long_scoreboard"})),
    Axis("fpref", frozenset({"regs", "short_scoreboard"})),
    Axis("group", frozenset({"l2", "hbm"})),
    Axis("epilogue", frozenset({"MIO", "hbm"})),
    Axis("swizzle", frozenset({"MIO"})),
    Axis("pred", frozenset({"ISSUE"})),
    Axis("minb", frozenset({"regs", "occupancy"})),
)

# Значения осей, ЗАКОННЫЕ на этой машине.  Не «разумные», а собираемые.
DOMAINS = {
    "BM": (16, 32, 64, 128, 256),
    "BN": (128, 256),
    "BK": (32, 64),
    "WM": (1, 2, 4),
    "WN": (2, 4),
    "STAGES": (2,),  # 3/4 замерены: кадр стека 240-512 Б и обвал в 3-6 раз
    "GSTAGE": (1,),  # 2 замерено: то же
    "FPREF": (1, 2),
    "GROUP": (1, 8),  # GROUP=1 даёт 0.449 против 0.836 -- САМАЯ ДОРОГАЯ ОДИНОЧНАЯ ОСЬ
    "EPI": (0,),  # 1 (через разделяемую) замерен: кадр стека
    "SWZ": (0, 2),  # 2 = фазово-инъективный (пара разрядов {1,3}); 0 = как отгружено
    "PRED": (True,),
    "MINB": (1, 2),
}


def axes():
    return AXES


def _legal(h: GemmHyperform) -> bool:
    """Причина незаконности либо None.  Имя метода -- legal(), возвращает ПРИЧИНУ отказа."""
    return h.legal() is None


def variants(op: OpSpec) -> Iterator[Hyperform]:
    """Перечислить ЗАКОННЫЕ гиперформы под эту операцию.  Ни одной сборки."""
    if op.kind != "gemm":
        raise PluginCapabilityError(
            "скелеты sm_70 умеют только 'gemm'; запрошено %r. "
            "Скелетов FMHA нет -- это объявленная заглушка (declared_stubs)." % op.kind
        )
    N = op.shapes["N"]
    K = op.shapes["K"]
    for BM in DOMAINS["BM"]:
        for BN in DOMAINS["BN"]:
            if N % BN:
                continue
            for BK in DOMAINS["BK"]:
                if K % BK:
                    continue
                for WM in DOMAINS["WM"]:
                    for WN in DOMAINS["WN"]:
                        for ST in DOMAINS["STAGES"]:
                            for GS in DOMAINS["GSTAGE"]:
                                for FP in DOMAINS["FPREF"]:
                                    for GR in DOMAINS["GROUP"]:
                                        for EP in DOMAINS["EPI"]:
                                            for SW in DOMAINS["SWZ"]:
                                                for PR in DOMAINS["PRED"]:
                                                    for MB in DOMAINS["MINB"]:
                                                        h = GemmHyperform(
                                                            BM,
                                                            BN,
                                                            BK,
                                                            WM,
                                                            WN,
                                                            ST,
                                                            GS,
                                                            FP,
                                                            GR,
                                                            EP,
                                                            SW,
                                                            PR,
                                                            MB,
                                                        )
                                                        if not _legal(h):
                                                            continue
                                                        yield Hyperform(
                                                            plugin="sm_70",
                                                            params=dict(
                                                                BM=BM,
                                                                BN=BN,
                                                                BK=BK,
                                                                WM=WM,
                                                                WN=WN,
                                                                STAGES=ST,
                                                                GSTAGE=GS,
                                                                FPREF=FP,
                                                                GROUP=GR,
                                                                EPI=EP,
                                                                SWZ=SW,
                                                                PRED=PR,
                                                                MINB=MB,
                                                            ),
                                                            key=h.tag(),
                                                        )


def to_gemm(h: Hyperform) -> GemmHyperform:
    p = h.params
    return GemmHyperform(
        p["BM"],
        p["BN"],
        p["BK"],
        p["WM"],
        p["WN"],
        p["STAGES"],
        p["GSTAGE"],
        p["FPREF"],
        p["GROUP"],
        p["EPI"],
        p["SWZ"],
        p["PRED"],
        p["MINB"],
    )


# --------------------------------------------------------------------------------------------
# СТЕНЫ ЧУЖИХ БИБЛИОТЕК -- ТОЛЬКО ДЛЯ ПЕЧАТИ (core на этот текст не ветвится)
# --------------------------------------------------------------------------------------------
LIBRARY_WALLS = (
    "pitch_linear_thread_map.h:298 'Number of iterations must be non-zero': у операнда A одна "
    "непрерывная итерация => варпов <= kQ/8. Это свойство raked-раскладки cutlass, НЕ железа: "
    "рукописное 64/128 2x8 = 16 варпов при BQ=64 собирается и считает верно (relL2 2.5e-04).",
    "kBJ=32: плитка даёт 4 стодвадцативосьмибитных доступа, а WarpThreadArrangement<8,4> ставит "
    "8 нитей => 4/8 = 0. ИСПРАВЛЕНО (min(8, kN/kElementsPerAccess)), но сам kBJ=32 замеренно МЁРТВ.",
    "mma_tensor_op_sm70.h:136 'Shape % GemmShape<32,32,4>': плитка 32x16 не собирается, потому "
    "что раскладки операндов Volta организованы 32x32-чередованием. ЖЕЛЕЗО ЭТО УМЕЕТ (два 16x16x4).",
    "<256,64,256>: 256 доступов на 512 нитей. Не чинится дёшево, и ВСЁ РАВНО не влезает в 96 КБ "
    "-- стена там smem, а не шаблон.",
)
