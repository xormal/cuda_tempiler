#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""sm_70: КЛАССИФИКАТОР ISA -- ОПЦИОНАЛЬНАЯ возможность плагина.

Тело -- `isa_sass.py` в этом же каталоге (бывший vendor/issue_slots.py, 1576 строк, перенесён
ЦЕЛИКОМ и без правок; его самопроверка -- сверка с ручным счётом 2026-07-31).

ГРАНИЦА СОБЛЮДЕНА ТЕМ, ЧТО ВЕСЬ ФАЙЛ ЛЕЖИТ НА СТОРОНЕ ПЛАГИНА.  Плагин вправе содержать
общий код; core не вправе содержать архитектурный.  Расщепление CFG/доминаторов/циклов в
core/ir/ делается ПОЗЖЕ и только тогда, когда у core появится ВТОРОЙ потребитель.

ЧЕГО ЗДЕСЬ НЕТ И НЕ БУДЕТ В v1: `encode_control` и доказательство round-trip.  Причина не
в трудоёмкости: верификатора зависимостей (T3) не построено, а неверный `wait` даёт не
падение, а ТИХО НЕВЕРНЫЙ ОТВЕТ.  `control_fields` (ЧТЕНИЕ stall/yield) есть -- оно безопасно.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from ..base import AtomClass, NotSupported, PluginCapabilityError

_HERE = os.path.dirname(os.path.abspath(__file__))


def isa():
    """Разборщик SASS.  Грузится по пути: он остаётся запускаемым файлом."""
    p = os.path.join(_HERE, "isa_sass.py")
    spec = importlib.util.spec_from_file_location("tempo_isa_sass", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@dataclass(frozen=True)
class ControlFields:
    """То, что мы УМЕЕМ ЧИТАТЬ из управляющего слова sm_70.  Остальное честно отсутствует."""

    stall: int  # биты 41-44 старшего слова
    yield_: bool  # бит 45
    write_barrier: None = None  # НЕ РАЗОБРАН
    read_barrier: None = None  # НЕ РАЗОБРАН
    wait_mask: None = None  # НЕ РАЗОБРАН (6 бит)
    reuse: None = None  # НЕ РАЗОБРАН


class Sm70Classifier:
    def classify(self, instr_text: str) -> AtomClass:
        m = isa()
        op = instr_text.strip().split()[0] if instr_text.strip() else ""
        base = op.split(".")[0]
        cls = None
        for fn in ("classify", "classify_op", "op_class"):
            if hasattr(m, fn):
                try:
                    cls = getattr(m, fn)(instr_text)
                    break
                except Exception:
                    continue
        channel = _CHANNEL.get(base, "OTHER")
        width = 4
        for w in (128, 64, 32):
            if (".%d" % w) in op:
                width = w // 8
                break
        return AtomClass(
            channel=channel if cls is None else str(cls),
            cycles=_CYCLES.get(channel, 1.0),
            width_bytes=width,
            latency_class=base,
            predicated=instr_text.strip().startswith("@"),
        )

    def decode(self, binary: Path, kernel_regex: str):
        m = isa()
        for fn in ("parse_sass", "load", "decode"):
            if hasattr(m, fn):
                return getattr(m, fn)(str(binary), kernel_regex)
        raise PluginCapabilityError("isa_sass.py не отдаёт разборщик под известным именем")

    def control_fields(self, word: int) -> ControlFields:
        """ЧТЕНИЕ управляющих полей.  Запись НЕ реализована намеренно (см. шапку)."""
        return ControlFields(stall=(word >> 41) & 0xF, yield_=bool((word >> 45) & 1))

    def encode_control(self, *a, **k):
        raise NotSupported(
            "обратная запись управляющих полей SASS в контракт v1 НЕ ВХОДИТ: нет верификатора "
            "зависимостей (T3), а неверный wait даёт не падение, а тихо неверный ответ"
        )


_CHANNEL = {
    "HMMA": "TENSOR",
    "IMMA": "TENSOR",
    "LDS": "MIO",
    "STS": "MIO",
    "SHFL": "MIO",
    "LDG": "LSU",
    "STG": "LSU",
    "RED": "LSU",
    "ATOM": "LSU",
    "FFMA": "FPU",
    "FADD": "FPU",
    "FMUL": "FPU",
    "FSETP": "FPU",
    "MUFU": "SFU",
    "IADD3": "ALU",
    "IMAD": "ALU",
    "LOP3": "ALU",
    "SHF": "ALU",
    "PRMT": "ALU",
    "SEL": "ALU",
    "MOV": "ALU",
    "ISETP": "ALU",
    "BRA": "BRANCH",
    "BSSY": "BRANCH",
    "BSYNC": "BRANCH",
    "EXIT": "BRANCH",
    "BAR": "BRANCH",
}
_CYCLES = {"TENSOR": 2.0, "ALU": 2.0, "FPU": 2.0, "SFU": 8.0, "MIO": 1.0, "LSU": 1.0,
           "BRANCH": 1.0, "OTHER": 1.0}
