#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ФАЛЬСИФИКАТОР ДОПУЩЕНИЯ -- гейт G8.  Главный гейт этого дерева.

Отличается от `nullarch` РОВНО ОДНИМ ПОЛЕМ:

    TransactionKind.consumes_registers:  True  ->  False

Больше ничем: ни одного другого числа, ни одной другой строки.  Проверяемое утверждение --
вывод отсекателя ОБЯЗАН СДВИНУТЬСЯ, потому что глубина подачи перестаёт стоить регистров, а
значит перестаёт стоить занятости.

ЗАЧЕМ ЭТО НУЖНО ИМЕННО СЕЙЧАС, БЕЗ ВТОРОГО ЖЕЛЕЗА.  Лексический гейт (G1) проверяет СЛОВАРЬ:
он поймает слово «cp.async» в конвейере.  Но допущение «подача съедает регистры» пишется БЕЗ
ЕДИНОГО ЗАПРЕТНОГО СЛОВА -- достаточно прибавить глубину буфера к оценке регистров прямо в
core.  Такая протечка невидима любому текстовому гейту и всплывёт только на A100, где она
даст ЗАНИЖЕННУЮ занятость и ОТСЕЧЁТ верный вариант.  Здесь она всплывает за 40 строк.

ПРАВИЛО, КОТОРОЕ ОТСЮДА СЛЕДУЕТ ДЛЯ ВСЕГО КОНТРАКТА:
    на каждое ДВУЗНАЧНОЕ поле контракта обязан существовать плагин-фальсификатор с
    противоположным значением и тест, что вывод сдвинулся.
"""

from __future__ import annotations

from ..base import CONTRACT, Report
from ..nullarch import (  # noqa: F401
    NullarchMachine,
    NullarchMemory,
    NullarchMeters,
    NullarchPlugin,
    NullarchResources,
    NullarchSkeletons,
    NullarchSync,
    NullarchTensor,
    NullarchToolchain,
    _demo_op,
)


class AsyncSync(NullarchSync):
    """ЕДИНСТВЕННОЕ ОТЛИЧИЕ ВО ВСЁМ ПЛАГИНЕ."""

    CONSUMES_REGISTERS = False


class NullarchAsyncPlugin(NullarchPlugin):
    id = "nullarch_async"
    contract = CONTRACT
    description = (
        "Подставная архитектура, отличающаяся от nullarch РОВНО полем "
        "TransactionKind.consumes_registers (True -> False). Фальсификатор гейта G8."
    )

    Sync = AsyncSync

    def declared_stubs(self):
        return ("вся архитектура подставная; отличие от nullarch -- одно поле",)

    def selftest(self) -> Report:
        r = super().selftest()
        tx = self.sync.transactions()[0]
        r.check("подача объявлена НЕ съедающей регистры", tx.consumes_registers is False)
        r.check("ожидание объявлено ЯВНЫМ", tx.wait_op is not None)

        # То самое, ради чего плагин существует: глубина перестала стоить регистров.
        from ..nullarch import load as load_base

        op = _demo_op()
        base = load_base()
        deep = [h for h in self.skeletons.variants(op) if h.params["depth"] == 3][0]
        regs_here = self.skeletons.resources_of(op, deep)[0]
        regs_base = base.skeletons.resources_of(op, deep)[0]
        r.check("глубина подачи БОЛЬШЕ НЕ СТОИТ регистров", regs_here < regs_base,
                "%d против %d" % (regs_here, regs_base))
        return r


_P = None


def load():
    global _P
    if _P is None:
        _P = NullarchAsyncPlugin()
    return _P
