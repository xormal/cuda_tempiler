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


class _ОбаКраяОптимистичны:
    """ПРЕЖНЯЯ РЕДАКЦИЯ ОЦЕНКИ, обёрткой: на вопрос «разольётся ли» отвечает ОПТИМИСТИЧНЫЙ край.

    Чужой объект не правится -- он оборачивается: подменить поведение плагина внутри его же
    класса значило бы менять то, что проверяется.  LAW=L-ESTIMATOR-AND-DECIDER-COUPLED.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def resources_of(self, op, h):
        regs, _max_live, smem = self._inner.resources_of(op, h)
        return regs, regs, smem


class _ПлагинСДругойОценкой:
    def __init__(self, inner):
        self._inner = inner
        self.skeletons = _ОбаКраяОптимистичны(inner.skeletons)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def run(verbose: bool = True, optimistic: bool = False):
    cfg = _configs()
    with open(MANIFEST, encoding="utf-8") as f:
        man = json.load(f)
    plugin = registry.load("sm70")
    if optimistic:
        plugin = _ПлагинСДругойОценкой(plugin)

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


# ================================================================================================
# ЧАСТЬ 2. ГЕЙТ РЕГИСТРОВ ОТСЕКАТЕЛЯ (задача 144).  LAW=L-PRUNE-REGS-PESSIMISTIC
#
# ЗАЧЕМ ЗАВЕДЁН.  Задача 141 починила счёт регистров в АНАЛИЗАТОРЕ (он читает машинный код) и
# завела ему гейт на пяти телах с известными числами ptxas.  ОТСЕКАТЕЛЬ этой правки не получил
# и СВОЕГО ГЕЙТА НЕ ИМЕЛ ВОВСЕ: он считает регистры ДО СБОРКИ, по структуре скелета, и до этого
# гейта ни одна его цифра ни разу не сверялась с ptxas.  А выбрасывает варианты ИМЕННО ОН.
#
# ПРЕДМЕТ ЗДЕСЬ ДРУГОЙ, И ЭТО ГЛАВНОЕ ОГРАНИЧЕНИЕ.  Анализатор читает ТЕЛО (живость по ширине,
# циклически, плюс уже разлитое из опкодов) -- у него есть чем ошибиться на 7 регистров.  У
# отсекателя тела ЕЩЁ НЕТ: он складывает объявленные скелетом массивы.  Такой счёт не может
# попасть в ptxas точно ни при какой правке -- сборщик перевыражает значения, размножает адреса
# и раскручивает подачу.  ЗАМЕР ПОЛОСЫ (ниже, `_band`) даёт разброс 0.65..1.20 против ptxas.
# Поэтому гейт требует НЕ ТОЧНОСТИ, А НАПРАВЛЕНИЯ ОШИБКИ:
#
#   А1. тело, которое ptxas РАЗЛИЛ, отсекателю ЗАПРЕЩЕНО называть «влезает» (FITS).
#       Это и есть та ложь, ради которой гейт заведён: по ней продукт выбирает вариант.
#   А2. тело, которое ptxas собрал БЕЗ разлива, отсекателю ЗАПРЕЩЕНО ВЫБРАСЫВАТЬ.
#       Ошибаться в сторону «не влезет» разрешено -- но тогда вариант обязан дойти до сборки.
#   Б0. БЮДЖЕТ РЕГИСТРОВ -- ОБЪЯВЛЕННЫЙ СБОРЩИКУ, а не выведенный из своей же оценки.
#       Проверяется фактом: ptxas НИКОГДА не превышает объявленный бюджет, и упирается в него
#       РОВНО на разлившихся телах.  Прежний бюджет (из модельной занятости) этот факт нарушает.
#   Б.  ВЕЛИЧИНА: верхний край полосы обязан НЕ ЗАНИЖАТЬ ptxas на телах без разлива.
#   В.  ФАЛЬСИФИКАТОР ГЕЙТА: прежняя редакция (ОПТИМИСТИЧНЫЙ край полосы в роли потребности)
#       обязана гейт ПРОВАЛИТЬ.  Гейт, который проходит обеими редакциями, не проверяет ничего.
#
# ЧИСЛА ptxas БЕРУТСЯ ИЗ ПАСПОРТА ПОСТАВКИ (kernels/shipped/.../manifest.json, раздел "build"):
# двенадцать гиперформ ЭТОГО ЖЕ скелета, у каждой регистры / кадр стека / разделяемая.  Ни
# сборки, ни карты гейт не требует.
# ================================================================================================
# Форма операции, на которой считается вердикт.  Регистровая часть вердикта от M/N/K НЕ зависит
# (она читает только гиперформу), а форма нужна лишь чтобы гиперформа была законной.
GATE_OP = {"M": 2048, "N": 3840, "K": 4096}


def _gate_bodies():
    """(тег, params, ptxas-регистры, кадр стека, разделяемая) для двенадцати собранных тел."""
    cfg = _configs()
    with open(MANIFEST, encoding="utf-8") as f:
        man = json.load(f)
    out = []
    for tag, b in man["build"].items():
        want = cfg.get(tag)
        if want is None:
            continue
        out.append((tag, want, int(b["regs"]), int(b["frame"]), int(b["smem"])))
    return out


def _band(rows):
    """ПОЛОСА оценки против ptxas, посчитанная ТОЛЬКО на телах БЕЗ разлива.

    ПОЧЕМУ ТОЛЬКО НА НИХ -- это третья из трёх ошибок задачи 141, и отсекатель повторял её в
    СВОЕЙ КАЛИБРОВКЕ.  У тела, которое разлило, ptxas печатает не потребность, а ПОТОЛОК
    (объявленный бюджет): потребность ушла в локальную память и в числе не видна.  Взять такое
    число за «истинную потребность» значит записать в полосу точку, которая занижена ровно на
    разлитое, -- то есть подвинуть полосу в сторону «влезает».  Разлившиеся тела входят сюда
    НЕРАВЕНСТВОМ (потребность > бюджета), и это ОТДЕЛЬНАЯ, более сильная проверка: край полосы,
    посчитанный без них, обязан их разлив ОБЪЯСНИТЬ.
    """
    clean = [(est / pt, tag) for tag, est, pt, fr, _b in rows if fr == 0]
    clean.sort()
    return clean[0], clean[-1]


def regs_gate(verbose: bool = True):
    """-> список претензий.  Карта не нужна и сборка не нужна: числа ptxas уже известны."""
    from tempo.plugins.base import Hyperform

    plugin = registry.load("sm70")
    op = OpSpec(
        "gemm",
        "fp16",
        "fp16",
        "fp16",
        "fp32",
        "k",
        "k",
        "n",
        dict(GATE_OP),
        tol_rel_l2=1e-3,
    )
    tpw = 32  # нитей в варпе -- единица, в которой объявлен бюджет; не число железа, а единица
    bad, rows, table = [], [], []
    for tag, want, ptxas, frame, smem in _gate_bodies():
        h = Hyperform(plugin="sm_70", params=dict(want), key=tag)
        spilled = frame > 0
        try:
            est, max_live, smem_m = plugin.skeletons.resources_of(op, h)
            need = plugin.resources.spill_threshold(max_live)
            cand = prune(plugin, op, [h])[0]
        except Exception as e:  # noqa: BLE001 -- гейт обязан назвать отказ, а не упасть
            bad.append("%s: вердикт недоступен: %s" % (tag, e))
            continue
        warps_cta = max(1, want["WM"] * want["WN"])
        declared = plugin.resources.reg_budget(warps_cta * want["MINB"])
        modelled = plugin.resources.reg_budget(
            plugin.resources.occupancy(est, smem_m, warps_cta * tpw).warps_per_sm
        )
        code = cand.verdict.code if cand.verdict is not None else "?"
        why = []
        # Б0. БЮДЖЕТ -- ФАКТ, А НЕ МОДЕЛЬ
        if ptxas > declared:
            why.append(
                "ptxas взял %d регистров при ОБЪЯВЛЕННОМ бюджете %d -- бюджет посчитан неверно"
                % (ptxas, declared)
            )
        if spilled and ptxas != declared:
            why.append(
                "разлившееся тело обязано упереться в объявленный бюджет: ptxas %d, бюджет %d"
                % (ptxas, declared)
            )
        # А1. ЛОЖЬ В СТОРОНУ «ВЛЕЗАЕТ»
        if spilled and code == "FITS":
            why.append(
                "ptxas разлил (кадр %d Б), а отсекатель говорит FITS: потребность %d при "
                "бюджете %d" % (frame, need, declared)
            )
        # А2. СЛЕПОЕ ВЫБРАСЫВАНИЕ
        if not spilled and not cand.kept:
            why.append(
                "тело собралось БЕЗ разлива (%d регистров, кадр 0), а отсекатель его ВЫБРОСИЛ: %s"
                % (ptxas, cand.reason)
            )
        # Б. ВЕЛИЧИНА.  Число, которым отсекатель ОБЪЯВЛЯЕТ потребность, обязано НЕ ЗАНИЖАТЬ
        # то, что сборщик взял на самом деле.  Проверяется только там, где ptxas напечатал
        # ПОТРЕБНОСТЬ; на разлившихся он печатает ПОТОЛОК, и требовать попадания в него значило
        # бы проверять по обрезанной величине (та же оговорка, что и у гейта анализатора).
        if not spilled and need < ptxas:
            why.append(
                "объявленная потребность %d ЗАНИЖАЕТ взятое сборщиком %d -- это и есть ошибка "
                "в сторону «влезает»" % (need, ptxas)
            )
        # Б2. НЕЗАВИСИМОЕ ПОДТВЕРЖДЕНИЕ: край полосы считается ТОЛЬКО на телах без разлива
        # (см. `_band`), и обязан ОБЪЯСНИТЬ разлив тех двух, что в расчёт не входили.
        if spilled and need <= declared:
            why.append(
                "потребность %d не превышает бюджет %d -- край полосы, посчитанный на телах "
                "без разлива, разлив ЭТОГО тела не объясняет" % (need, declared)
            )
        if why:
            bad.append("%s: %s" % (tag, "; ".join(why)))
        rows.append((tag, est, ptxas, frame, declared))
        table.append(
            (tag, ptxas, frame, declared, modelled, est, need, code, cand.kept, not why)
        )

    lo, hi = _band(rows)
    # В. ФАЛЬСИФИКАТОР ГЕЙТА: ОПТИМИСТИЧНЫЙ край полосы в роли потребности -- ровно прежняя
    # редакция.  Число берётся из самого отсекателя (`resources_of` отдаёт его первым полем:
    # это край, который вариант СОХРАНЯЕТ, и он законен для занятости и для границы времени --
    # но не для вопроса «разольётся ли»).
    falsified = [
        tag
        for tag, est, ptxas, frame, declared in rows
        if frame > 0 and est <= declared
    ]

    if verbose:
        print(
            "\nГЕЙТ РЕГИСТРОВ ОТСЕКАТЕЛЯ: %d собранных тел, у каждого регистры и кадр от ptxas"
            % len(table)
        )
        print(
            "  %-16s %6s %5s | %6s %6s | %6s %6s %-10s %-8s %s"
            % (
                "тело",
                "ptxas",
                "кадр",
                "объявл",
                "модель",
                "оценка",
                "нужно",
                "вердикт",
                "оставлен",
                "итог",
            )
        )
        for t in table:
            print(
                "  %-16s %6d %5d | %6d %6d | %6d %6d %-10s %-8s %s"
                % (
                    t[0][:16],
                    t[1],
                    t[2],
                    t[3],
                    t[4],
                    t[5],
                    t[6],
                    t[7],
                    "да" if t[8] else "НЕТ",
                    "ПРОЙДЕНО" if t[9] else "ПРОВАЛ",
                )
            )
        n_sp = sum(1 for r in rows if r[3] > 0)
        print(
            "\n  А. ВЕРДИКТ: тел %d, разлившихся по ptxas %d; названо «влезает» из разлившихся %d; "
            "выброшено из НЕ разлившихся %d."
            % (
                len(table),
                n_sp,
                sum(1 for t in table if t[2] > 0 and t[7] == "FITS"),
                sum(1 for t in table if t[2] == 0 and not t[8]),
            )
        )
        print(
            "  Б0. БЮДЖЕТ: объявленный сборщику против выведенного из МОДЕЛЬНОЙ занятости -- "
            "расходятся на %d телах из %d;\n      ptxas превысил модельный на %d телах "
            "(объявленный -- ни на одном, это и делает его ФАКТОМ, а модельный -- догадкой)."
            % (
                sum(1 for t in table if t[3] != t[4]),
                len(table),
                sum(1 for t in table if t[1] > t[4]),
            )
        )
        print(
            "  Б. ПОЛОСА, ЗАМЕРЕННАЯ ЗДЕСЬ ЖЕ (только тела БЕЗ разлива, где ptxas печатает "
            "ПОТРЕБНОСТЬ, а не потолок):\n      край полосы / ptxas = %.3f (%s) .. %.3f (%s), то "
            "есть верхняя граница потребности = край x %.3f."
            % (lo[0], lo[1], hi[0], hi[1], 1.0 / lo[0])
        )
        print(
            "      Множитель посчитан БЕЗ разлившихся тел и обязан их разлив ОБЪЯСНИТЬ (Б2) -- "
            "это независимое подтверждение, а не подгонка."
        )
        print(
            "  В. ФАЛЬСИФИКАТОР: прежняя редакция (ОПТИМИСТИЧНЫЙ край в роли потребности) "
            "называет влезающими %d разлившихся тела%s"
            % (len(falsified), (": " + ", ".join(falsified)) if falsified else "")
        )
        for r in bad:
            print("  ОТКАЗ: " + r)
    if not falsified:
        bad.append(
            "ФАЛЬСИФИКАТОР ГЕЙТА ПУСТ: прежняя редакция модели гейт ПРОХОДИТ, значит гейт не "
            "проверяет ничего"
        )
    return bad


# ================================================================================================
# ЧАСТЬ 3. ГЕЙТ СЦЕПЛЕННОСТИ.  LAW=L-ESTIMATOR-AND-DECIDER-COUPLED
#
# ЧТО ПРОВЕРЯЕТСЯ.  Модель здесь и ОЦЕНИВАЕТ (сколько регистров нужно), и РЕШАЕТ (выбросить ли
# вариант).  Половины сцеплены: край оценки выбран ПЕССИМИСТИЧНЫМ намеренно, и это законно РОВНО
# ПОТОМУ, что жёсткий рез по нему снят.  Починить одну половину без другой ХУЖЕ, чем не чинить
# обе: пессимистичный край + прежний жёсткий рез выбрасывает замеренных победителей, которых
# оптимистичный край + жёсткий рез оставлял.
#
# ГЕЙТ ПАДАЕТ В ОБЕ СТОРОНЫ, и это делает его проверкой, а не иллюстрацией:
#   * если продукт (пессимистичный край + СОВЕЩАТЕЛЬНЫЙ вердикт) теряет хоть одного победителя
#     -- отказ продукта;
#   * если контрфакт (та же оценка + ЖЁСТКИЙ рез) НИКОГО не теряет -- значит сцепленности нет,
#     и утверждение закона ложно; тогда падает закон, а не продукт.
# ================================================================================================
def coupling_gate(verbose: bool = True):
    """-> список претензий.  Ни карты, ни сборки: контрфакт считается на тех же 35 точках."""
    from tempo.core.search import prune as prune_mod
    from tempo.plugins.base import RESOURCE_CODES

    было = prune_mod.HARD_CODES
    try:
        lost_now, _ = run(verbose=False)
        # КОНТРФАКТ: «правка одной половины» -- край оценки пессимистичный (как сейчас), а рез
        # ЖЁСТКИЙ (как было до задачи 144: резал ЛЮБОЙ вердикт «не ok», а не только структурный
        # отказ запуска).  Подозрение на разлив снова выбрасывает вариант БЕЗ сборки.
        prune_mod.HARD_CODES = frozenset(c for c in RESOURCE_CODES if c != "FITS")
        lost_hard, _ = run(verbose=False)
        # ВТОРАЯ ПОЛОВИНА КОНТРФАКТА: тот же жёсткий рез, но по ОПТИМИСТИЧНОМУ краю -- прежняя
        # редакция ЦЕЛИКОМ.  Она теряет МЕНЬШЕ, и это и есть «одна правка хуже обеих неправок».
        lost_hard_optimistic, _ = run(verbose=False, optimistic=True)
    finally:
        prune_mod.HARD_CODES = было

    with open(MANIFEST, encoding="utf-8") as f:
        всего = len(json.load(f)["rows"])
    bad = []
    if lost_now:
        bad.append(
            "продукт (пессимистичный край + СОВЕЩАТЕЛЬНЫЙ вердикт) потерял %d победителей из %d"
            % (len(lost_now), всего)
        )
    if not lost_hard:
        bad.append(
            "СЦЕПЛЕННОСТИ НЕТ: жёсткий рез по пессимистичному краю не теряет НИ ОДНОГО "
            "победителя из %d -- тогда утверждение «правка одной половины хуже обеих неправок» "
            "ложно, и закон L-ESTIMATOR-AND-DECIDER-COUPLED надо снимать, а не продукт чинить"
            % всего
        )
    if len(lost_hard) <= len(lost_hard_optimistic):
        bad.append(
            "ОДНА ПРАВКА НЕ ХУЖЕ ДВУХ НЕПРАВОК: пессимистичный край с жёстким резом теряет %d, "
            "прежняя редакция целиком -- %d. Тогда половины НЕ сцеплены, и закон ложен"
            % (len(lost_hard), len(lost_hard_optimistic))
        )
    if verbose:
        print(
            "\nГЕЙТ СЦЕПЛЕННОСТИ ОЦЕНЩИКА И РЕШАТЕЛЯ: %d замеренных точек паспорта"
            % всего
        )
        print(
            "  как в продукте (край ПЕССИМИСТИЧНЫЙ, вердикт СОВЕЩАТЕЛЬНЫЙ): потеряно %d"
            % len(lost_now)
        )
        print(
            "  контрфакт «починили ОДНУ половину» (край ПЕССИМИСТИЧНЫЙ, рез ЖЁСТКИЙ): потеряно %d"
            % len(lost_hard)
        )
        print(
            "  обе НЕправки вместе (край ОПТИМИСТИЧНЫЙ, рез ЖЁСТКИЙ -- как было): потеряно %d"
            % len(lost_hard_optimistic)
        )
        print(
            "  ЧИТАЕТСЯ ТАК: точность оценки и право резать -- ОДНО решение, а не два. "
            "Пессимизм оценки\n  законен ровно потому, что жёсткий рез снят; вернуть рез, не "
            "вернув оптимистичный край,\n  значит выбросить %d замеренных победителя из %d -- "
            "БОЛЬШЕ, чем теряла прежняя редакция целиком (%d)."
            % (len(lost_hard), всего, len(lost_hard_optimistic))
        )
        for r in bad:
            print("  ОТКАЗ: " + r)
    return bad


def main():
    lost, missing = run(verbose=True)
    bad = regs_gate(verbose=True)
    bad = bad + coupling_gate(verbose=True)
    n = len(lost) + len(missing) + len(bad)
    print(
        "\nИТОГ ПРИЁМКИ: %s  (приёмка %d, гейт регистров %d)"
        % (
            "ПРОЙДЕНО" if n == 0 else "ОТКАЗ (%d)" % n,
            len(lost) + len(missing),
            len(bad),
        )
    )
    return 0 if n == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
