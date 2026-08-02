#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ПРАВИЛО ПРИЁМКИ ОТСЕКАТЕЛЯ, ИСПОЛНЯЕМОЕ.  Карта не нужна.

Правило записано в `tempo/core/search/prune.py` и до сих пор НИ РАЗУ НЕ ИСПОЛНЯЛОСЬ:

    на подвыборке гиперформ собрать и замерить ВСЁ без отсечения; отсекатель обязан НЕ
    ВЫБРОСИТЬ фактического победителя.  Выбросил -- НАПЕЧАТАТЬ ЭТО, а не тихо ослабить порог.

Сборка и замер здесь не нужны: они УЖЕ СДЕЛАНЫ и лежат в паспорте поставки
`kernels/shipped/gemm_fp16/sm_70/manifest.json` -- 35 строк, у каждой замеренный победитель.
Точная конфигурация победителя берётся из `configs.inc` того же скелета, которым мерили.
Это и есть подвыборка «собрано и замерено ВСЁ»: 12 гиперформ на 35 боевых точках.

ЧТО ЗДЕСЬ ПРОВЕРЯЕТСЯ (падает) И ЧТО ТОЛЬКО ПЕЧАТАЕТСЯ (не падает):
  * ПАДАЕТ  -- отсекатель выбросил хоть одного замеренного победителя.  Это отказ продукта:
              вариант, который СОБРАЛСЯ и ОБОГНАЛ сильную библиотеку, объявлен безнадёжным
              БЕЗ СБОРКИ.
  * ПАДАЕТ  -- точной конфигурации победителя нет в перечислении вовсе (тогда отсекатель ни
              при чём, а дыра в осях, и это ещё хуже).
  * ПЕЧАТАЕТСЯ -- РАНГ победителя по модельной границе.  Это КАЧЕСТВО, а не корректность:
              модель -- граница, а не предсказание, и требовать от неё первого места нельзя.
              Ранг разделён по полосе M намеренно: при M <= 128 замерено, что связывает
              ЧТЕНИЕ ВЕСОВ (78-90 % полосы при 12-47 % счёта), а канала внекристальной
              подачи в таблице ставок НЕТ -- см. docs/NOT_YET.md.  В этой полосе модель
              ранжировать НЕ МОЖЕТ, и число это показывает.
  * ПЕЧАТАЕТСЯ -- ТРЕБУЕМЫЙ ЗАПАС `bound_slack`: во сколько раз граница замеренного
              победителя хуже лучшей границы.  Если он вырастет выше умолчания отсекателя,
              отсекатель начнёт терять победителей ГРАНИЦЕЙ, а не вердиктом.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tempo.core.search.prune import prune  # noqa: E402
from tempo.plugins import registry  # noqa: E402
from tempo.plugins.base import OpSpec  # noqa: E402

MANIFEST = os.path.join(ROOT, "kernels/shipped/gemm_fp16/sm_70/manifest.json")
CONFIGS = os.path.join(ROOT, "tempo/plugins/sm70/skeletons/gemm_hmma884/configs.inc")

# Порядок полей макроса CFG в harness.cu -- это ЕДИНСТВЕННОЕ место, где он назван в тестах.
CFG_FIELDS = (
    "BM",
    "BN",
    "BK",
    "WM",
    "WN",
    "STAGES",
    "GSTAGE",
    "FPREF",
    "GROUP",
    "EPI",
    "SWZ",
    "PRED",
    "MINB",
)
# Полоса M, ниже которой связывающий канал в модели ОТСУТСТВУЕТ (замер: EV_mridge, перелом 128).
M_RIDGE = 128


def _configs():
    out = {}
    with open(CONFIGS, encoding="utf-8") as f:
        for line in f:
            m = re.match(r'\s*CFG\("([^"]+)",\s*(.+)\),\s*$', line)
            if not m:
                continue
            v = [x.strip() for x in m.group(2).split(",")]
            vals = [int(x) for x in v[0:11]] + [v[11] == "true", int(v[12])]
            out[m.group(1)] = dict(zip(CFG_FIELDS, vals))
    return out


def _params(h):
    p = dict(h.params)
    p["PRED"] = bool(p["PRED"])
    return {k: p[k] for k in CFG_FIELDS}


def run(verbose: bool = True):
    cfg = _configs()
    with open(MANIFEST, encoding="utf-8") as f:
        man = json.load(f)
    plugin = registry.load("sm70")

    lost, missing = [], []
    rank_decode, rank_prefill, need_slack = [], [], 0.0
    firsts = set()
    if verbose:
        print(
            "ПРАВИЛО ПРИЁМКИ ОТСЕКАТЕЛЯ: %d замеренных точек, победители из паспорта поставки"
            % len(man["rows"])
        )
        print(
            "  %-6s %-6s %-6s %-14s %6s %6s %8s %8s"
            % ("K", "N", "M", "победитель", "всего", "остав", "ранг", "из")
        )

    for row in man["rows"]:
        M, N, K, name = row["M"], row["N"], row["K"], row["winner"]
        want = cfg.get(name)
        if want is None:
            missing.append("%s: нет строки в configs.inc" % name)
            continue
        op = OpSpec(
            "gemm",
            "fp16",
            "fp16",
            "fp16",
            "fp32",
            "k",
            "k",
            "n",
            {"M": M, "N": N, "K": K},
            tol_rel_l2=1e-3,
        )
        hyper = list(plugin.skeletons.variants(op))
        cands = prune(plugin, op, hyper)
        mine = [c for c in cands if _params(c.hyper) == want]
        if not mine:
            missing.append(
                "K%d N%d M%d: конфигурации %s НЕТ в перечислении" % (K, N, M, name)
            )
            continue
        dropped = [c for c in mine if not c.kept]
        if dropped:
            lost.append(
                "K%d N%d M%d победитель %s ВЫБРОШЕН: %s"
                % (K, N, M, name, dropped[0].reason)
            )

        ranked = sorted(
            [c for c in cands if c.kept and c.bound is not None and not c.bound.silent],
            key=lambda c: c.bound.T,
        )
        if ranked:
            firsts.add(ranked[0].hyper.key)
        pos = [i for i, c in enumerate(ranked) if _params(c.hyper) == want]
        rank = (min(pos) + 1) if pos else None
        if rank is not None:
            (rank_decode if M <= M_RIDGE else rank_prefill).append(rank)

        # ТРЕБУЕМЫЙ ЗАПАС: считается БЕЗ отсечения по границе, иначе он себя же и обрежет.
        wide = [
            c
            for c in prune(plugin, op, hyper, bound_slack=float("inf"))
            if c.bound is not None and not c.bound.silent
        ]
        w_mine = [c for c in wide if _params(c.hyper) == want]
        if wide and w_mine:
            need_slack = max(
                need_slack,
                min(c.bound.T for c in w_mine) / min(c.bound.T for c in wide),
            )
        if verbose:
            print(
                "  %-6d %-6d %-6d %-14s %6d %6d %8s %8d%s"
                % (
                    K,
                    N,
                    M,
                    name,
                    len(cands),
                    sum(1 for c in cands if c.kept),
                    rank if rank else "-",
                    len(ranked),
                    "   <-- ПОТЕРЯН" if dropped else "",
                )
            )

    if verbose:
        print()
        for r in lost + missing:
            print("  ОТКАЗ: " + r)
        print(
            "  ВЫБРОШЕНО ЗАМЕРЕННЫХ ПОБЕДИТЕЛЕЙ: %d из %d"
            % (len(lost), len(man["rows"]))
        )
        print(
            "  РАЗНЫХ ПЕРВЫХ ВАРИАНТОВ ПО МОДЕЛИ НА %d ФОРМАХ: %d  (1 = модель формы НЕ РАЗЛИЧАЕТ)"
            % (len(man["rows"]), len(firsts))
        )
        for label, arr in (
            ("префилл  M > %d" % M_RIDGE, rank_prefill),
            ("декод    M <= %d" % M_RIDGE, rank_decode),
        ):
            if arr:
                print(
                    "  РАНГ победителя, %s: %d..%d, медиана %.0f (точек %d)"
                    % (label, min(arr), max(arr), statistics.median(arr), len(arr))
                )
        print("  ТРЕБУЕМЫЙ bound_slack: %.2f" % need_slack)
        print(
            "  ЧЕГО МОДЕЛЬ НЕ УМЕЕТ И ЭТО НАЗВАНО: канала внекристальной подачи в таблице "
            "ставок нет,\n  поэтому в полосе M <= %d ранг победителя -- НЕ показатель "
            "качества модели, а признак\n  отсутствующего канала (docs/NOT_YET.md)."
            % M_RIDGE
        )
    return lost, missing


def main():
    lost, missing = run(verbose=True)
    bad = len(lost) + len(missing)
    print("\nИТОГ ПРИЁМКИ: %s" % ("ПРОЙДЕНО" if bad == 0 else "ОТКАЗ (%d)" % bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
