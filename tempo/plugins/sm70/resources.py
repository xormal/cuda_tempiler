#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""sm_70: РЕСУРСЫ И ЗАНЯТОСТЬ.  Вердикт БЕЗ СБОРКИ -- это и есть отсекатель конвейера.

Три числа, каждое замерено и каждое НЕ выводится из спецификации:
    Q(W) = min(255, 8*floor(256/W))   -- совпало с прямым опросом ptxas 4/4;
    R_треб = MaxLive + 7              -- точно ДО ЕДИНИЦЫ на трёх бюджетах;
    первые ДВА разлитых значения БЕСПЛАТНЫ -- следствие РАЗНОСТИ двух порогов.

И одно поведение компилятора, без которого модель врёт: БЕЗ ОБЪЯВЛЕННОЙ ЗАНЯТОСТИ ptxas НЕ
РАЗЛИВАЕТ ВООБЩЕ -- он отдаёт отказ запуска вместо замедления.  Поэтому регистры имеют цену
ТОЛЬКО там, где бюджет сообщён компилятору (__launch_bounds__).
"""

from __future__ import annotations

from ..base import Occupancy, Rate, ResourceVerdict
from .machine import Sm70Machine, data

_M = Sm70Machine()


class Sm70Resources:
    # -- ЗАКОНЫ ------------------------------------------------------------------------------
    def reg_budget(self, warps_per_sm: int) -> int:
        """Q(W) = min(255, 8*floor(256/W)).  ЗАМЕР, а не расчёт из размера файла."""
        if warps_per_sm <= 0:
            return int(_M.rate("GEOM.REG_ISA_LIMIT").value)
        isa = int(_M.rate("GEOM.REG_ISA_LIMIT").value)
        gran = int(_M.rate("GEOM.REG_ALLOC_GRANULARITY").value)
        return min(isa, gran * (256 // warps_per_sm))

    def spill_threshold(self, max_live: int) -> int:
        """Требуемых регистров = MaxLive + REG_OVERHEAD (замерено до единицы)."""
        return int(max_live + _M.rate("REG.OVERHEAD").value)

    def free_spills(self) -> int:
        return int(_M.rate("REG.FREE_SPILLS").value)

    def spill_cost(self, n: int) -> Rate:
        """СТУПЕНЬКА, не прямая: первые два бесплатны, дальше x5.1, у края x75.7."""
        free = self.free_spills()
        if n <= free:
            return Rate(
                "SPILL.COST",
                1.0,
                "кратность",
                "MEASURED",
                note="первые %d разлитых значения бесплатны (их LDL прячется)" % free,
            )
        step = _M.rate("REG.SPILL_STEP")
        edge = _M.rate("REG.SPILL_EDGE")
        val = step.value if n < 16 else edge.value
        return Rate(
            "SPILL.COST", val, "кратность", "MEASURED", prov=step.prov, note=step.note
        )

    def occupancy(self, regs: int, smem_bytes: int, threads: int) -> Occupancy:
        """варпов/SM = floor(65536/рег/32); ФОРМА CTA ЕГО НЕ МЕНЯЕТ."""
        words = int(_M.rate("GEOM.REGFILE_WORDS_PER_SM").value)
        tpw = int(_M.rate("GEOM.THREADS_PER_WARP").value)
        slots = int(_M.rate("GEOM.WARP_SLOTS_PER_SM").value)
        smem_sm = int(_M.rate("GEOM.SMEM_PER_SM").value)

        by_reg = words // max(1, regs) // tpw
        warps_per_cta = max(1, threads // tpw)
        by_smem = (
            (smem_sm // max(1, smem_bytes)) * warps_per_cta if smem_bytes else slots
        )
        w = min(slots, by_reg, by_smem)
        limiter = "regs"
        if by_smem < by_reg and by_smem <= slots:
            limiter = "smem"
        if slots <= min(by_reg, by_smem):
            limiter = "warp_slots"
        ctas = max(0, w // warps_per_cta)
        # ВАРПОВ РОВНО СТОЛЬКО, СКОЛЬКО ИХ В РЕЗИДЕНТНЫХ БЛОКАХ.  LAW=L-PRUNE-REGS-PESSIMISTIC
        # Резидентность -- целое число БЛОКОВ, а не варпов: варп чужого блока на процессор не
        # въезжает. Прежняя редакция отдавала остаток от деления (20 варпов при двух блоках по
        # восемь), и число, которого на процессоре не бывает, шло дальше в бюджет регистров:
        # Q(20)=96 вместо Q(16)=128. Замерено: ровно по этому 96 отсекатель выбросил
        # m64x128k32, который собрался на 112 регистрах БЕЗ единого разлива.
        w = ctas * warps_per_cta
        # ПОРОГ ВТОРОГО БЛОКА: ОБА условия обязаны выполниться одновременно.
        if ctas >= 2:
            if smem_bytes > int(_M.rate("REG.SECOND_CTA_SMEM").value) or regs > int(
                _M.rate("REG.SECOND_CTA_REGS").value
            ):
                ctas = 1
                w = warps_per_cta
                limiter = "second_cta_threshold"
        return Occupancy(
            warps_per_sm=int(w),
            ctas_per_sm=int(ctas),
            regs_per_thread=int(regs),
            limiter=limiter,
        )

    # -- ВЕРДИКТ БЕЗ СБОРКИ ------------------------------------------------------------------
    def verdict(
        self,
        regs: int,
        max_live: int,
        smem_bytes: int,
        threads: int,
        min_ctas_per_sm: int = 1,
    ) -> ResourceVerdict:
        """LAW=L-PRUNE-REGS-PESSIMISTIC

        ТРИ ВХОДА, И КАЖДЫЙ ОТВЕЧАЕТ ЗА СВОЁ:
          * `regs` -- ОПТИМИСТИЧНЫЙ край оценки. Им считается РЕЗИДЕНТНОСТЬ, потому что
            резидентность идёт в НИЖНЮЮ границу времени, а нижнюю границу занижать нельзя.
          * `max_live` -- ПЕССИМИСТИЧНЫЙ край. Им отвечают «разольётся ли»: ошибка в сторону
            «влезает» ТИХО теряет победителя, ошибка в сторону «не влезет» стоит сборки.
          * `min_ctas_per_sm` -- ОБЪЯВЛЕННАЯ СБОРЩИКУ резидентность, то есть то, что сборщик
            УСЛЫШАЛ. Бюджет регистров назначает ОНА. Прежняя редакция выводила бюджет из
            собственной оценки занятости -- круг: оценка проверялась бюджетом, посчитанным по
            ней же. Круг опровергнут прямым замером: на трёх телах из двенадцати выведенный
            бюджет оказался НИЖЕ того, что ptxas реально взял (объявленный -- ни на одном).
        """
        isa = int(_M.rate("GEOM.REG_ISA_LIMIT").value)
        smem_max = int(_M.rate("GEOM.SMEM_PER_CTA_MAX").value)
        tpw = int(_M.rate("GEOM.THREADS_PER_WARP").value)
        occ = self.occupancy(regs, smem_bytes, threads)

        if smem_bytes > smem_max:
            return ResourceVerdict(
                False,
                "WALL_SMEM",
                occ,
                "разделяемой нужно %d Б при потолке %d Б (потолок найден ИЗ ОТКАЗА ЗАПУСКА, "
                "а не из спецификации)" % (smem_bytes, smem_max),
            )
        if occ.warps_per_sm <= 0 or occ.ctas_per_sm <= 0:
            return ResourceVerdict(
                False, "NO_BUDGET", occ, "ни один блок не резидентен"
            )
        need = self.spill_threshold(max_live)
        declared_warps = max(1, threads // tpw) * max(1, int(min_ctas_per_sm))
        budget = min(isa, self.reg_budget(declared_warps))
        if need > isa:
            return ResourceVerdict(
                False,
                "WALL_REG",
                occ,
                "MaxLive %d + %d = %d > потолка ISA %d: не лечится занятостью вовсе "
                "(число -- ПЕССИМИСТИЧНЫЙ край оценки, поэтому вывод СОВЕЩАТЕЛЬНЫЙ)"
                % (max_live, need - max_live, need, isa),
            )
        if need > budget:
            over = need - budget
            k = self.spill_cost(over)
            return ResourceVerdict(
                over <= self.free_spills(),
                "SPILL",
                occ,
                "разлив %d значений (нужно %d, ОБЪЯВЛЕННЫЙ бюджет %d при %d варпах); цена x%.1f%s"
                % (
                    over,
                    need,
                    budget,
                    declared_warps,
                    k.value,
                    "" if over > self.free_spills() else " -- в пределах бесплатных",
                ),
            )
        # «ВЛЕЗАЕТ» СКАЗАНО ПО ПЕССИМИСТИЧНОМУ КРАЮ -- и только поэтому это утверждение, а не
        # пожелание: если даже верхний край полосы уместился в объявленный бюджет, уместится и
        # тело.  Обратное («разольётся») по тому же краю -- лишь ПОДОЗРЕНИЕ, и режет по нему
        # не вердикт, а сборка.
        return ResourceVerdict(
            True,
            "FITS",
            occ,
            "нужно %d регистров даже по ВЕРХНЕМУ краю оценки, ОБЪЯВЛЕННЫЙ бюджет %d при %d "
            "варпах (%d блок(ов) на SM, ограничивает %s)"
            % (need, budget, declared_warps, occ.ctas_per_sm, occ.limiter),
        )

    def wave_quantum(self, occ: Occupancy) -> int:
        """Волна = число SM x блоков на SM.  Полезные сетки КРАТНЫ ЭТОМУ, а не 64/128."""
        return int(_M.rate("GEOM.SMS").value) * max(1, occ.ctas_per_sm)

    def declared_bounds_required(self) -> bool:
        """True: без __launch_bounds__ ptxas не разливает, и модель разлива неприменима."""
        return True

    # -- то, что НЕ переносится ---------------------------------------------------------------
    def notes(self) -> tuple:
        r = data("regs")
        return (
            r["occupancy_is_not_a_lever"]["note"],
            "ПЕРЕНОСИМОСТЬ: " + r["occupancy_is_not_a_lever"]["portability"],
            r["two_optima_diverge"]["note"],
            r["spill_cost"]["gotcha"],
        )
