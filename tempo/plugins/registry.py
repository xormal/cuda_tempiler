#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""Загрузка плагина по arch-id.  Неизвестный МАЖОР контракта -> отказ БЕЗ попытки работать.

Почему отказ, а не «попробуем»: плагин с контрактом другого мажора отдаёт поля, которые
конвейер прочтёт по-своему.  Это не падение, а ТИХО НЕВЕРНЫЙ вердикт -- самый дорогой из
исходов.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict

from . import base
from .base import ContractError, Plugin

_CACHE: Dict[str, object] = {}


def available() -> list:
    """Имена подкаталогов tempo/plugins/*, у которых есть __init__.py с load()."""
    import tempo.plugins as pkg

    out = []
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.ispkg:
            out.append(m.name)
    return sorted(out)


def _major(contract: str) -> str:
    parts = contract.rsplit("/", 1)
    return parts[0] if len(parts) == 2 else contract


def load(arch: str):
    """arch -- имя каталога плагина ("sm70", "sm80", "nullarch", "null"...)."""
    if arch in _CACHE:
        return _CACHE[arch]
    try:
        mod = importlib.import_module("tempo.plugins.%s" % arch)
    except ImportError as e:
        raise ContractError(
            "плагин %r не найден; известны: %s" % (arch, ", ".join(available()))
        ) from e
    if not hasattr(mod, "load"):
        raise ContractError("плагин %r не объявляет load()" % arch)
    p = mod.load()
    contract = getattr(p, "contract", None)
    if contract is None:
        raise ContractError("плагин %r не объявил contract" % arch)
    if _major(contract) != base.CONTRACT_MAJOR:
        raise ContractError(
            "плагин %r объявил контракт %r; конвейер понимает только мажор %r"
            % (arch, contract, base.CONTRACT_MAJOR)
        )
    _validate_rates(p, arch)
    _CACHE[arch] = p
    return p


def _validate_rates(p, arch: str) -> None:
    """Правила §3.1, проверяемые ЗАГРУЗЧИКОМ: MEASURED обязан нести карту и 0 соседей."""
    try:
        table = p.machine.symbols()
    except Exception as e:  # плагин-заглушка вправе не иметь машины вовсе
        raise ContractError("плагин %r: symbols() упал: %s" % (arch, e)) from e
    for sym, r in table.items():
        if not isinstance(r, base.Rate):
            raise ContractError("плагин %r: ставка %r не Rate" % (arch, sym))
        r.check()


def by_arch_id(arch_id: str):
    """Найти плагин по его ЗАЯВЛЕННОМУ arch-id ("sm_70"), а не по имени каталога."""
    for name in available():
        try:
            p = load(name)
        except ContractError:
            continue
        if getattr(p, "id", None) == arch_id:
            return p
    raise ContractError("плагина с arch-id %r нет" % arch_id)
