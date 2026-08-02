#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""СПЕЦИФИКАЦИЯ ОПЕРАЦИИ.  Определение живёт в контракте; здесь -- разбор и проверка.

Продукт распознаёт ОБЪЯВЛЕННУЮ операцию, а не читает произвольный исходник.  Незнакомое
ядро получает ОТКАЗ, а не плохой код.  Это ограничение названо прямо в docs/NOT_YET.md и
является границей честности, а не недоделкой: парсера C++ здесь нет и в версии 1 не будет.
"""

from __future__ import annotations

from tempo.plugins.base import OpSpec  # noqa: F401  (единственное место определения)

__all__ = ["OpSpec", "validate", "describe", "flops"]

KNOWN_KINDS = ("gemm",)
KNOWN_DTYPES = ("fp16", "fp32", "int8", "uint8", "q7", "bf16")


def validate(op: OpSpec) -> None:
    if op.kind not in KNOWN_KINDS:
        raise ValueError("операция %r не объявлена; известны: %s" % (op.kind, ", ".join(KNOWN_KINDS)))
    for name in ("dtype_a", "dtype_b", "dtype_c", "dtype_acc"):
        v = getattr(op, name)
        if v not in KNOWN_DTYPES:
            raise ValueError("тип %r=%r не объявлен" % (name, v))
    if op.kind == "gemm":
        for k in ("M", "N", "K"):
            if k not in op.shapes:
                raise ValueError("для 'gemm' обязательна форма %r" % k)
            if int(op.shapes[k]) <= 0:
                raise ValueError("форма %s = %r" % (k, op.shapes[k]))
    if op.tol_rel_l2 < 0:
        raise ValueError("допуск отрицателен")


def flops(op: OpSpec) -> float:
    if op.kind != "gemm":
        raise ValueError("счёт работы объявлен только для 'gemm'")
    s = op.shapes
    return 2.0 * s["M"] * s["N"] * s["K"]


def describe(op: OpSpec) -> str:
    s = op.shapes
    return "%s %sx%sx%s  %s*%s->%s (накопитель %s), раскладки %s/%s/%s" % (
        op.kind, s.get("M"), s.get("N"), s.get("K"),
        op.dtype_a, op.dtype_b, op.dtype_c, op.dtype_acc,
        op.layout_a, op.layout_b, op.layout_c,
    )
