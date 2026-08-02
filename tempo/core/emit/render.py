#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ЭМИССИЯ: скелет + параметры -> текст.  БЕЗ ЛОГИКИ ЖЕЛЕЗА.

Конвейер здесь ничего не решает: он берёт `Rendered` у плагина, приклеивает шапку с ШТАМПОМ
ПОРОЖДЕНИЯ и пишет файл.  Чем тупее эмиттер, тем меньше мест, где заведётся расхождение
между тем, что посчитал отсекатель, и тем, что собралось.

ЧЕГО ЗДЕСЬ НЕТ И НЕ БУДЕТ В ВЕРСИИ 1: эмиссии РАСПИСАНИЯ.  Порядок команд и распределение
регистров остаются у стороннего компилятора; разрыв «граница модели против него» ИЗМЕРЯЕТСЯ,
а не закрывается.  Причина не в трудоёмкости: верификатора зависимостей нет, а неверное
ожидание даёт не падение, а ТИХО НЕВЕРНЫЙ ОТВЕТ.
"""

from __future__ import annotations

import os

from tempo.core.report.provenance import stamp

HEADER = (
    "// SPDX-License-Identifier: LicenseRef-TRL-1.0\n"
    "// Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>\n"
    "// Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.\n"
)


def emit(plugin, op, hyper, out_dir, version="0.1", data_hash="-", date="") -> str:
    r = plugin.skeletons.render(op, hyper)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "kernel.cuh")
    text = HEADER + "// " + stamp(version, plugin.id, data_hash, date) + "\n" + r.source
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
