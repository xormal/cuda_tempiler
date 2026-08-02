#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""sm_70: СИНХРОНИЗАЦИЯ И ТРАНЗАКЦИИ.

Тут объявлено ГЛАВНОЕ ВОЛТОВСКОЕ ДОПУЩЕНИЕ, из-за которого на A100 пришлось бы переписывать
конвейер, если бы оно жило в core:

    подача gmem->smem СЪЕДАЕТ РЕГИСТРЫ (LDG -> регистр -> STS), и потому глубина буфера
    ПЛАТИТ ЗАНЯТОСТЬЮ.

На sm_80 есть cp.async, который идёт мимо регистрового файла, и это допущение становится
ЛОЖНЫМ.  Поле consumes_registers делает его ДАННЫМИ.  Фальсификатор -- плагин
`nullarch_async`, отличающийся ровно этим полем; гейт G8 требует, чтобы вывод отсекателя
СДВИНУЛСЯ.
"""

from __future__ import annotations

from ..base import BarrierKind, PluginCapabilityError, Rate, TransactionKind

_GMEM_TO_SMEM = TransactionKind(
    id="gmem->smem",
    issue_op="LDG",
    wait_op=None,  # ожидание НЕЯВНОЕ: табло (scoreboard), отдельной инструкции нет
    granularity="instruction",
    in_flight_max=Rate(
        "XFER.IN_FLIGHT",
        float("nan"),
        "запрос",
        "NOT_MEASURED",
        note="ЗАМЕРЕНО КОСВЕННО: запросы в полёте и занятость -- ОДИН ресурс (конвейер страниц "
        "через регистры НЕ РАБОТАЕТ). Отдельного числа нет.",
    ),
    occupies={"LSU": 1.0, "ISSUE": 1.0, "MIO": 1.0},  # LDG в регистр + STS из него
    consumes_registers=True,  # <-- ПОЛЕ-ФАЛЬСИФИКАТОР (гейт G8)
    consumes_smem_staging=False,
    depth_axis="gstage",  # глубина глобального буфера -- ось поиска
)

_BARRIERS = (
    BarrierKind(
        id="cta",
        scope="cta",
        phased=False,
        counted=False,
        cost=Rate(
            "BARRIER.CTA",
            2.0,
            "% времени форварда",
            "MEASURED",
            note="фаза 'рандеву' = 2.0 % (разложение форварда). ЕДИНИЦА СОКРЫТИЯ ЗАДЕРЖКИ -- "
            "БЛОК: при одном резидентном блоке любой __syncthreads() останавливает SM целиком.",
        ),
    ),
    BarrierKind(
        id="warp",
        scope="warp",
        phased=False,
        counted=False,
        cost=Rate(
            "BARRIER.WARP",
            0.0,
            "такт",
            "MODEL",
            note="__syncwarp: на Volta независимое планирование полос делает его обязательным, "
            "но его цена стендом не выделена",
        ),
    ),
)


class Sm70Sync:
    def transactions(self):
        return (_GMEM_TO_SMEM,)

    def barriers(self):
        return _BARRIERS

    def rendezvous_cost(self, barrier_id: str, participants: int) -> Rate:
        for b in _BARRIERS:
            if b.id == barrier_id:
                return b.cost
        raise PluginCapabilityError(
            "sm_70 не имеет барьера %r; есть только %s. "
            "Фазный/счётный барьер (mbarrier) появляется с sm_80 и потребует tempo/arch/2."
            % (barrier_id, ", ".join(b.id for b in _BARRIERS))
        )
