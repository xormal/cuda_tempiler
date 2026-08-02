#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""РЕЕСТР «форма -> вариант -> откат».  Сериализуемый, читается диспетчером поставки.

ПОЧЕМУ ДИСПЕТЧЕР ОБЯЗАТЕЛЕН, А НЕ УДОБЕН (замерено на 35 боевых точках):
    лучшая ЕДИНАЯ гиперформа на все точки .... геосреднее 0.746 к сопернику
    выбор по полосе входной размерности ....... геосреднее 0.885
Разница 0.139 БОЛЬШЕ всего, что удалось выиграть внутри одной гиперформы.

И ОТКАТ ОБЯЗАТЕЛЕН ВСЕГДА: покрытие предиката НИКОГДА не полное. У плотного тела есть
требование делимости, у пути малых размерностей своё; непокрытое обязано уходить на штатный
путь, иначе продукт ломает чужой сервер на первой нестандартной форме.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class Entry:
    predicate: dict  # что обязано выполниться (dtype, раскладки, делимость, границы)
    kernel_id: str
    params: dict = field(default_factory=dict)
    note: str = ""


@dataclass
class Registry:
    op: str
    arch: str
    entries: list = field(default_factory=list)
    fallback: str = "штатный путь вызывающего"

    def select(self, **kw):
        for e in self.entries:
            if _match(e.predicate, kw):
                return e
        return None

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema": "tempo/registry/1",
                "op": self.op,
                "arch": self.arch,
                "fallback": self.fallback,
                "entries": [
                    {
                        "predicate": e.predicate,
                        "kernel_id": e.kernel_id,
                        "params": e.params,
                        "note": e.note,
                    }
                    for e in self.entries
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def _match(pred, kw) -> bool:
    for k, v in pred.items():
        if k.endswith("_max"):
            if kw.get(k[:-4]) is None or kw[k[:-4]] > v:
                return False
        elif k.endswith("_min"):
            if kw.get(k[:-4]) is None or kw[k[:-4]] < v:
                return False
        elif k.endswith("_divides"):
            base = kw.get(k[:-8])
            if base is None or base % v:
                return False
        else:
            if kw.get(k) != v:
                return False
    return True
