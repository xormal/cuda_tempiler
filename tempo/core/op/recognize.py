#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""РАСПОЗНАВАНИЕ: наивный исходник -> OpSpec.

ЕДИНСТВЕННЫЙ ИСТОЧНИК СПЕЦИФИКАЦИИ -- блок `TEMPO-OP` ВНУТРИ самого наивного исходника.

Почему не отдельный файл описания рядом.  Потому что это ВТОРОЕ МЕСТО, и оно разъедется с
первым: сигнатура в исходнике поменяется, описание останется.  Правило проекта («одна
реализация закона на два инструмента») применяется и здесь: описание живёт там же, где код,
и сверяется с сигнатурой пуска.

ФОРМАТ (строки внутри комментария, порядок произволен):
    TEMPO-OP: gemm
    dtype: a=fp16 b=fp16 c=fp16 acc=fp32
    layout: a=k b=k c=n
    shape: M=* N=* K=*
    tol_rel_l2: 1e-3
    entry: naive_gemm_fp16
    signature: (const __half* A, const __half* B, __half* C, int M, int N, int K)

`*` в форме означает «задаётся при запуске».  Отсутствие блока -- ОТКАЗ, а не догадка.
"""

from __future__ import annotations

import re

from .spec import OpSpec, validate

BLOCK_START = re.compile(r"TEMPO-OP\s*:\s*(\w+)")
KV = re.compile(r"^\s*(?://|#|\*)?\s*([A-Za-z_][\w]*)\s*:\s*(.+?)\s*$")


class NotRecognised(Exception):
    """Незнакомое ядро получает ОТКАЗ, а не плохой код."""


def parse_block(text: str) -> dict:
    m = BLOCK_START.search(text)
    if not m:
        raise NotRecognised(
            "в исходнике нет блока TEMPO-OP. Продукт распознаёт ОБЪЯВЛЕННУЮ операцию и НЕ "
            "читает произвольный C++ -- парсера здесь нет и в версии 1 не будет."
        )
    kind = m.group(1)
    lines = text[m.end() :].split("\n")
    out = {"kind": kind}
    for ln in lines[:40]:
        if "TEMPO-OP-END" in ln:
            break
        km = KV.match(ln)
        if not km:
            continue
        key, val = km.group(1).lower(), km.group(2)
        if key in ("dtype", "layout", "shape"):
            sub = {}
            for pair in val.replace(",", " ").split():
                if "=" in pair:
                    a, b = pair.split("=", 1)
                    sub[a.strip().lower()] = b.strip()
            out[key] = sub
        elif key in ("tol_rel_l2", "entry", "signature", "coverage", "scale"):
            out[key] = val
    return out


def to_spec(block: dict, shapes: dict = None) -> OpSpec:
    d = block.get("dtype", {})
    l = block.get("layout", {})
    s = dict(block.get("shape", {}))
    shapes = shapes or {}
    resolved = {}
    for k in ("M", "N", "K"):
        v = shapes.get(k, s.get(k, s.get(k.lower())))
        if v is None or str(v) == "*":
            if k in shapes:
                v = shapes[k]
            else:
                raise NotRecognised(
                    "форма %s объявлена как '*' (задаётся при запуске), но при запуске не задана"
                    % k
                )
        resolved[k] = int(v)
    op = OpSpec(
        kind=block["kind"],
        dtype_a=d.get("a", "fp16"),
        dtype_b=d.get("b", "fp16"),
        dtype_c=d.get("c", "fp16"),
        dtype_acc=d.get("acc", "fp32"),
        layout_a=l.get("a", "k"),
        layout_b=l.get("b", "k"),
        layout_c=l.get("c", "n"),
        shapes=resolved,
        scale=block.get("scale"),
        tol_rel_l2=float(block.get("tol_rel_l2", 0.0)),
        coverage=block.get("coverage", "atomic_stamp"),
    )
    validate(op)
    return op


def check_signature(block: dict, text: str) -> None:
    """СВЕРКА С СИГНАТУРОЙ ПУСКА -- то, ради чего блок живёт в исходнике, а не рядом.

    Если объявленная точка входа не встречается в тексте, значит описание отстало от кода.
    Это ловится ЗДЕСЬ, а не после часа замеров не того ядра.
    """
    entry = block.get("entry")
    if not entry:
        raise NotRecognised("блок TEMPO-OP не объявил точку входа (entry:)")
    # Искать ОБЪЯВЛЕНИЕ, а не вхождение имени: имя встречается и в самом блоке TEMPO-OP,
    # то есть проверка «имя есть в тексте» была бы ТОЖДЕСТВЕННО ИСТИННОЙ и ничего не ловила.
    # Это ровно та симметрия, из-за которой две сверки соглашаются, будучи обе неверны.
    if not re.search(re.escape(entry) + r"\s*\(", text):
        raise NotRecognised(
            "точка входа %r объявлена в блоке TEMPO-OP, но объявления с таким именем в "
            "исходнике нет: описание отстало от кода" % entry
        )
    sig = block.get("signature")
    if sig:
        args = [a.strip() for a in sig.strip("()").split(",") if a.strip()]
        # проверяем ЧИСЛО аргументов у объявления с этим именем
        m = re.search(re.escape(entry) + r"\s*\(([^)]*)\)", text)
        if m:
            real = [a.strip() for a in m.group(1).split(",") if a.strip()]
            if len(real) != len(args):
                raise NotRecognised(
                    "сигнатура в блоке TEMPO-OP объявляет %d аргументов, в исходнике %d"
                    % (len(args), len(real))
                )


def from_file(path: str, shapes: dict = None) -> OpSpec:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    block = parse_block(text)
    check_signature(block, text)
    return to_spec(block, shapes)
