#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ОТСЕЧЕНИЕ БЕЗ СБОРКИ: сперва РЕСУРСНЫЙ вердикт, потом модельная граница.

Порядок не произволен.  Ресурсный вердикт -- ДА/НЕТ (не влезает = не соберётся или разольётся
и потеряет кратно), и он дешевле.  Граница -- это НЕПРЕДСКАЗАНИЕ: она умеет только сказать
«быстрее вот этого не будет», поэтому по ней отсекают лишь заведомо худшие варианты.

ПРАВИЛО ПРИЁМКИ ОТСЕКАТЕЛЯ (без него его нельзя включать):
    на подвыборке гиперформ собрать и замерить ВСЁ без отсечения; отсекатель обязан НЕ
    ВЫБРОСИТЬ фактического победителя.  Выбросил -- НАПЕЧАТАТЬ ЭТО, а не тихо ослабить порог.

ЧЕГО ЗДЕСЬ НЕТ И БЫТЬ НЕ ДОЛЖНО: правил вида «предпочитать низкую занятость».  Это ВЫВОД,
полученный на конкретных телах конкретной машины, и в конвейере он был бы протечкой границы
(ловят гейты G6 и G8).  Отсекатель знает только вердикт плагина и границу по каналам.
"""

from __future__ import annotations

from dataclasses import dataclass

from tempo.core.model.bound import Bound, bound


@dataclass
class Candidate:
    hyper: object
    verdict: object
    bound: object
    kept: bool
    reason: str

    def render(self) -> str:
        mark = "ОСТАВЛЕН" if self.kept else "ОТСЕЧЁН "
        return "%s %-42s %s" % (mark, getattr(self.hyper, "key", "?"), self.reason)


def prune(plugin, op, hyperforms, keep_silent: bool = True, bound_slack: float = 4.0):
    """Вернуть список Candidate.  Ничего не собирает и не запускает.

    keep_silent=True: варианты, по которым МОДЕЛЬ МОЛЧИТ, ОСТАВЛЯЮТСЯ.  Отсекать по молчащей
    модели значит отсекать по незнанию -- ровно то, чего делать нельзя.
    bound_slack: во сколько раз хуже лучшей границы вариант ещё оставляют.  Запас берётся с
    большим коэффициентом, потому что граница -- не предсказание (замеренный разрыв 1.6x).
    """
    channels = plugin.machine.channels()
    out = []
    best = None
    rows = []
    for h in hyperforms:
        try:
            regs, max_live, smem = plugin.skeletons.resources_of(op, h)
            v = plugin.resources.verdict(
                regs, max_live, smem, plugin.skeletons.launch_of(op, h).threads
            )
        except Exception as e:
            out.append(
                Candidate(h, None, None, False, "ресурсный вердикт недоступен: %s" % e)
            )
            continue
        if not v.ok:
            out.append(Candidate(h, v, None, False, "%s: %s" % (v.code, v.explain)))
            continue
        try:
            atoms = plugin.skeletons.estimate_atoms(op, h)
            b = bound(atoms, channels, warps_per_sm=v.occ.warps_per_sm)
        except Exception as e:
            out.append(
                Candidate(
                    h,
                    v,
                    None,
                    keep_silent,
                    "граница недоступна (%s) -- вариант ОСТАВЛЕН, отсекать по незнанию нельзя"
                    % e,
                )
            )
            continue
        rows.append((h, v, b))
        if not b.silent and (best is None or b.T < best):
            best = b.T

    for h, v, b in rows:
        if b.silent:
            out.append(
                Candidate(h, v, b, keep_silent, "МОДЕЛЬ МОЛЧИТ: " + b.silent_reason)
            )
            continue
        if best is not None and b.T > best * bound_slack:
            out.append(
                Candidate(
                    h,
                    v,
                    b,
                    False,
                    "граница %.2f хуже лучшей (%.2f) более чем в %.1f раза"
                    % (b.T, best, bound_slack),
                )
            )
        else:
            out.append(
                Candidate(
                    h,
                    v,
                    b,
                    True,
                    "%s; T >= %.2f, связывает %s" % (v.code, b.T, b.binding),
                )
            )
    return out


def kept(cands):
    return [c for c in cands if c.kept]


def summary(cands) -> str:
    k = sum(1 for c in cands if c.kept)
    silent = sum(1 for c in cands if c.bound is not None and c.bound.silent)
    return (
        "гиперформ %d, оставлено %d, отсечено %d (из них по молчащей модели -- 0, молчащих %d)"
        % (len(cands), k, len(cands) - k, silent)
    )
