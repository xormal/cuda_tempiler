#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ПУСТОЙ ПЛАГИН -- фальсификатор гейта G2.

Он не умеет НИЧЕГО.  Проверяемое утверждение: конвейер, напоровшись на такой плагин, обязан
выдать СТРУКТУРНЫЙ ОТКАЗ (`PluginCapabilityError`) с названной причиной, а не упасть с
`AttributeError`/`KeyError` где-то в середине.

Разница не косметическая.  `AttributeError` в середине конвейера означает, что часть работы
уже сделана по частично прочитанному плагину, и её результат МОЖЕТ БЫТЬ НАПЕЧАТАН.
"""

from __future__ import annotations

from ..base import CONTRACT, PluginCapabilityError, Report


class _Refuses:
    """Любое обращение -- структурный отказ с указанием, чего именно нет."""

    def __init__(self, section):
        self._section = section

    def __getattr__(self, name):
        def _fail(*a, **k):
            raise PluginCapabilityError(
                "пустой плагин: раздел %r не реализован (спрошено %r)"
                % (self._section, name)
            )

        return _fail


class _NoMachine(_Refuses):
    def __init__(self):
        super().__init__("machine")

    def symbols(self):
        return {}  # ЗАКРЫТАЯ И ПУСТАЯ таблица: любой символ -> UnknownSymbol у вызывающего

    def arch(self):
        return "null"

    def channels(self):
        raise PluginCapabilityError("пустой плагин: каналов нет")


class NullPlugin:
    id = "null"
    contract = CONTRACT
    description = "Пустой плагин. Не реализует ничего. Существует ради гейта G2."

    def __init__(self):
        self.machine = _NoMachine()
        self.memory = _Refuses("memory")
        self.tensor = _Refuses("tensor")
        self.resources = _Refuses("resources")
        self.sync = _Refuses("sync")
        self.classifier = None
        self.skeletons = _Refuses("skeletons")
        self.toolchain = _Refuses("toolchain")
        self.meters = _Refuses("meters")

    def declared_stubs(self):
        return ("всё",)

    def selftest(self) -> Report:
        r = Report("null")
        r.check("объявляет контракт", self.contract == CONTRACT)
        r.check("таблица ставок пуста и ЗАКРЫТА", self.machine.symbols() == {})
        for name, fn in (
            ("channels", lambda: self.machine.channels()),
            ("skeletons.variants", lambda: self.skeletons.variants(None)),
            ("resources.verdict", lambda: self.resources.verdict(1, 1, 1, 1)),
        ):
            try:
                fn()
                r.check("%s отказывает СТРУКТУРНО" % name, False)
            except PluginCapabilityError:
                r.check("%s отказывает СТРУКТУРНО" % name, True)
            except Exception as e:
                r.check("%s отказывает СТРУКТУРНО" % name, False, type(e).__name__)
        return r


_P = None


def load():
    global _P
    if _P is None:
        _P = NullPlugin()
    return _P
