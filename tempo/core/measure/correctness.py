#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ГЕЙТ КОРРЕКТНОСТИ РАНЬШЕ СЕКУНДОМЕРА.  Порядок не обсуждается.

Два правила, оба оплачены:
  1. НЕ ОПТИМИЗИРОВАТЬ НЕВЕРНЫЙ ОТВЕТ.  Гейт стоит ПЕРЕД замером, а не после.
  2. ЛОМАТЬ СИММЕТРИЮ.  Две сверки, читающие ОДИН промежуточный буфер, соглашаются, будучи
     ОБЕ неверны.  Симметрия ломается подменой на ЧУЖОЙ путь по одному узлу.

Арифметика -- в core/op/oracle.py; здесь стадия и её порядок.
"""

from __future__ import annotations

from tempo.core.op.coverage import check as coverage_check  # noqa: F401
from tempo.core.op.oracle import OracleResult, gate, rel_l2  # noqa: F401

ORDER = (
    "1. собрать вариант",
    "2. проверить ПОКРЫТИЕ атомарным штампом (ловит недосчёт, который сверка значений пропускает)",
    "3. сверить ЗНАЧЕНИЯ с эталоном повышенной точности (не с половинной!)",
    "4. сломать симметрию независимой сверкой по ЧУЖОМУ пути",
    "5. и только теперь -- секундомер",
)


def must_pass_before_timing(res: OracleResult) -> None:
    if not res.ok:
        raise AssertionError(
            "секундомер ЗАПРЕЩЁН: гейт корректности не пройден.\n" + res.render()
        )
