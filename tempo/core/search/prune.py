#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ОТСЕЧЕНИЕ БЕЗ СБОРКИ: сперва РЕСУРСНЫЙ вердикт, потом модельная граница.

Порядок не произволен.  Ресурсный вердикт дешевле, и он единственный, кто вправе сказать
«этого не будет».  Граница -- это НЕПРЕДСКАЗАНИЕ: она умеет только сказать «быстрее вот этого
не будет», поэтому по ней отсекают лишь заведомо худшие варианты.

РЕСУРСНЫЙ ВЕРДИКТ НЕ ОДИН, А ДВА РАЗНЫХ УТВЕРЖДЕНИЯ, И РЕЖЕТ ТОЛЬКО ОДНО (задача 144).
Прежняя редакция резала по любому «не ok», и это было выбрасыванием ВСЛЕПУЮ: половина этого
вердикта -- ОЦЕНКА РЕГИСТРОВ БЕЗ СБОРКИ, у которой замеренная полоса против сборщика
0.747..1.254, то есть точности нет в принципе.  Разбор -- у HARD_CODES ниже.

ПРАВИЛО ПРИЁМКИ ОТСЕКАТЕЛЯ (без него его нельзя включать):
    на подвыборке гиперформ собрать и замерить ВСЁ без отсечения; отсекатель обязан НЕ
    ВЫБРОСИТЬ фактического победителя.  Выбросил -- НАПЕЧАТАТЬ ЭТО, а не тихо ослабить порог.

ЧЕГО ЗДЕСЬ НЕТ И БЫТЬ НЕ ДОЛЖНО: правил вида «предпочитать низкую занятость».  Это ВЫВОД,
полученный на конкретных телах конкретной машины, и в конвейере он был бы протечкой границы
(ловят гейты G6 и G8).  Отсекатель знает только вердикт плагина и границу по каналам.
"""

from __future__ import annotations

from dataclasses import dataclass

from tempo.core.model.bound import bound

# КАКИЕ ВЕРДИКТЫ РЕЖУТ ЖЁСТКО, А КАКИЕ ТОЛЬКО СОВЕТУЮТ.  LAW=L-PRUNE-REGS-PESSIMISTIC
#
# Различение не косметическое, оно ровно того же рода, что различение «граница» и
# «предсказание» выше по файлу.  Коды вердикта распадаются на ДВА РАЗНЫХ УТВЕРЖДЕНИЯ:
#
#   ОТКАЗ ЗАПУСКА -- «этого не будет».  Запрошено больше разделяемой, чем процессор отдаёт
#     блоку; ни один блок не резидентен.  Это НЕ оценка: у обоих чисел есть ПОТОЛОК, и он
#     проверяется точным сравнением, а не полосой.  Такой вариант выбрасывают, и правильно.
#
#   ПОДОЗРЕНИЕ НА РАЗЛИВ -- «по нашей оценке не влезет».  Регистры до сборки СЧИТАЮТСЯ ПО
#     СТРУКТУРЕ, и точности у такого счёта нет ни при какой правке (замеренная полоса против
#     сборщика -- 0.747..1.254).  Раз точности нет, край берётся ПЕССИМИСТИЧНЫЙ: ложь «влезает»
#     тихо теряет победителя, ложь «не влезет» стоит одной сборки.  Но жёсткий рез по
#     намеренно завышенному краю превращает эту дешёвую ошибку обратно в дорогую -- замерено:
#     ДВАДЦАТЬ ПЯТЬ замеренных победителей из тридцати пяти выбрасываются, и все они
#     собрались БЕЗ единого разлива.  Поэтому вариант ОСТАВЛЯЮТ С ПОМЕТКОЙ: пусть его
#     соберут и ПРОВЕРЯТ -- разлив увидит сборка, а не догадка.
#
# Отсекатель на СМЫСЛ кодов не смотрит -- он смотрит на их принадлежность к этому списку;
# сами коды объявлены контрактом (docs/CONTRACT.md, раздел Resources), а не этим файлом.
HARD_CODES = frozenset({"WALL_SMEM", "NO_BUDGET"})


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
            lau = plugin.skeletons.launch_of(op, h)
            v = plugin.resources.verdict(
                regs, max_live, smem, lau.threads, lau.min_ctas_per_sm
            )
        except Exception as e:
            out.append(
                Candidate(h, None, None, False, "ресурсный вердикт недоступен: %s" % e)
            )
            continue
        if not v.ok and v.code in HARD_CODES:
            out.append(Candidate(h, v, None, False, "%s: %s" % (v.code, v.explain)))
            continue
        advice = (
            ""
            if v.ok
            else "ПОДОЗРЕНИЕ НА РАЗЛИВ (оценка БЕЗ сборки, край полосы) -- вариант ОСТАВЛЕН, "
            "разлив обязана показать СБОРКА; "
        )
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
                    advice
                    + "граница недоступна (%s) -- вариант ОСТАВЛЕН, отсекать по незнанию нельзя"
                    % e,
                )
            )
            continue
        rows.append((h, v, b, advice))
        if not b.silent and (best is None or b.T < best):
            best = b.T

    for h, v, b, advice in rows:
        if b.silent:
            out.append(
                Candidate(
                    h, v, b, keep_silent, advice + "МОДЕЛЬ МОЛЧИТ: " + b.silent_reason
                )
            )
            continue
        if best is not None and b.T > best * bound_slack:
            out.append(
                Candidate(
                    h,
                    v,
                    b,
                    False,
                    advice
                    + "граница %.2f хуже лучшей (%.2f) более чем в %.1f раза"
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
                    advice + "%s; T >= %.2f, связывает %s" % (v.code, b.T, b.binding),
                )
            )
    return out


def kept(cands):
    return [c for c in cands if c.kept]


def summary(cands) -> str:
    k = sum(1 for c in cands if c.kept)
    silent = sum(1 for c in cands if c.bound is not None and c.bound.silent)
    # ОСТАВЛЕННЫЕ ПОД ПОДОЗРЕНИЕМ СЧИТАЮТСЯ ОТДЕЛЬНО: это не «прошли», это «модель против, но
    # выбрасывать по ней запрещено».  Число обязано быть видно, иначе совещательность вырождается
    # в молчаливое «всё влезает».
    risky = sum(
        1 for c in cands if c.kept and c.verdict is not None and not c.verdict.ok
    )
    return (
        "гиперформ %d, оставлено %d (из них под подозрением на разлив %d), отсечено %d "
        "(из них по молчащей модели -- 0, молчащих %d)"
        % (len(cands), k, risky, len(cands) - k, silent)
    )
