#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ПЛАГИН Ampere / sm_80 (A100) -- КАРКАС.  НИ ОДНОГО ЗАМЕРА, И ЭТО НАПИСАНО ПРЯМО.

ЗАЧЕМ КАРКАС ЗАВОДИТСЯ СЕЙЧАС, А НЕ КОГДА ПОЯВИТСЯ A100.  Каркас, заведённый ПОСЛЕ, не
проверяет ничего: граница, у которой одна реализация, -- не граница, а привычка.  Этот
каркас обязан проходить `selftest()` НА ОДНИХ NOT_MEASURED и обязан отказывать структурно
там, где ставки нет.  Ровно это и делает его инструментом приёмки: он не даёт конвейеру
посчитать что-нибудь по волтовским числам.

ЧЕГО ЗДЕСЬ НЕТ И ЧЕГО ЗДЕСЬ НЕ БУДЕТ БЕЗ ЖЕЛЕЗА (пустых обещаний не пишем):
  * ни одной ёмкости канала, ни одной задержки, ни одной полосы -- всё NOT_MEASURED;
  * скелетов нет вовсе: тела пишутся под форму mma, а m16n8k16 -- другая карта фрагмента;
  * тулчейн и приборы отвечают структурным отказом, а не «примерно как на Volta».

ЧТО ПРИНЦИПИАЛЬНО ИНАЧЕ, А НЕ «ДРУГОЕ ЧИСЛО» (полный разбор -- data/machine/porting.json):
  1. cp.async -- НОВЫЙ КАНАЛ; волтовское допущение «подача съедает регистры» становится
     ЛОЖНЫМ.  Здесь оно уже объявлено как False, и это единственная ставка, которую каркас
     утверждает без замера, -- потому что это факт ISA, а не величина.
  2. IMMA -- у int8 появляется СОБСТВЕННАЯ тензорная операция, и волтовский вывод
     «int8 не даёт ФЛОПов» перестаёт быть верным.  Приём «int8 в мантиссе fp16 + ранг-1
     смещение» на sm_80 просто НЕ НУЖЕН.
  3. m16n8k16 -- закон плитки тот же алгебраически, но «накопитель = MB*NB*8» -- нет.
  4. L2 6 -> 40 МБ + управление резидентностью: ЕДИНСТВЕННОЕ место, где перенос потребует
     НОВОГО КОДА в core/model/residency.py, а не новых данных.
  5. «Занятость -- не рычаг» -- вывод из наших тел на Volta.  На A100 он ОБЯЗАН быть
     перевыведен фальсификатором.  Механический перенос был бы ложью.

ЗАПУСК САМОПРОВЕРКИ:
    python3 -c "from tempo.plugins import registry; print(registry.load('sm80').selftest().render())"
"""

from __future__ import annotations

import json
import os
from typing import Mapping

from ..base import (
    CONTRACT,
    BarrierKind,
    Channel,
    Layout,
    MemLevel,
    Occupancy,
    PluginCapabilityError,
    Rate,
    Report,
    ResourceVerdict,
    TransactionKind,
    UnknownSymbol,
    closed_table_get,
    not_measured,
)

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "machine")
_CACHE = {}


def data(name: str) -> dict:
    if name not in _CACHE:
        with open(os.path.join(_DATA, name + ".json"), encoding="utf-8") as f:
            _CACHE[name] = json.load(f)
    return _CACHE[name]


# --------------------------------------------------------------------------------------------
# ТАБЛИЦА СТАВОК.  Каждая -- SPEC (паспорт, запрещён в границе времени) либо NOT_MEASURED.
# Ни одной MEASURED: их появление обязано сопровождаться картой и нулём чужих процессов.
# --------------------------------------------------------------------------------------------
_CHANNEL_SYMBOLS = (
    "TENSOR",
    "ALU",
    "FPU",
    "SFU",
    "ISSUE",
    "BRANCH",
    "LSU",
    "MIO",
    "ASYNC",
)
_LATENCY_SYMBOLS = (
    "FFMA",
    "IADD3",
    "LOP3",
    "IMAD",
    "MUFU",
    "LDS",
    "SHFL",
    "STS",
    "STG",
    "LDC",
    "S2R",
    "LDG",
    "HMMA",
    "LDSM",
    "CPASYNC",
)


def _build_symbols() -> Mapping[str, Rate]:
    out = {}
    for c in _CHANNEL_SYMBOLS:
        out["CAP." + c] = not_measured(
            "CAP." + c,
            "такт/команду/планировщик",
            "ПЕРЕМЕРИТЬ стендом. Форма закона та же, число -- нет. Канал ASYNC на Volta "
            "не существует вовсе.",
        )
    for l in _LATENCY_SYMBOLS:
        out["LATENCY." + l] = not_measured(
            "LATENCY." + l,
            "такт",
            "ПЕРЕМЕРИТЬ лестницей зависимых цепей. FFMA ожидаемо 4, LDS/LDG -- нет.",
        )
    m = data("machine")
    for name, g in m["geometry"].items():
        out["GEOM." + name.upper()] = Rate(
            symbol="GEOM." + name.upper(),
            value=float(g["value"]),
            units=g["units"],
            status=g["status"],
            note=g.get("note", ""),
        )
    for name, p in m["peak"].items():
        out["PEAK." + name.upper()] = Rate(
            symbol="PEAK." + name.upper(),
            value=float(p["value"]),
            units=p["units"],
            status=p["status"],
            note=p.get("note", ""),
        )
    for s, note in (
        (
            "REG.OVERHEAD",
            "на Volta ровно 7 и до единицы; здесь -- перемерить компиляторным свидетелем",
        ),
        ("REG.FREE_SPILLS", "на Volta 2 (разность двух порогов); здесь -- перемерить"),
        ("REG.SPILL_STEP", "на Volta x5.1"),
        ("REG.SPILL_EDGE", "на Volta x75.7"),
        ("REG.SECOND_CTA_SMEM", "форма smem/2 та же, число другое (164 КБ вместо 96)"),
        ("REG.SECOND_CTA_REGS", "перемерить"),
        (
            "MIO.WAVEFRONT_BYTES",
            "банков те же 32x4 Б, но ЦЕНА вайвфронта -- перемерить",
        ),
        ("MIO.CONFLICT", "вход из карты адресов, не константа железа"),
        ("WAVE.QUANTUM", "108 SM вместо 80 -- но подтвердить замером, а не паспортом"),
        ("ISSUE.SLOT_PRICE_IDLE", "метод переносится, число -- нет"),
        ("ISSUE.SLOT_PRICE_INCHAIN", "метод переносится, число -- нет"),
        ("TENSOR.COST", "форма mma другая (m16n8k16) -- и единица команды другая"),
        ("TENSOR.WARP_COST", "вывести из НОВОЙ карты фрагмента"),
    ):
        out[s] = not_measured(s, "-", note)
    return out


_SYMBOLS = None


class Sm80Machine:
    def arch(self):
        return "sm_80"

    def symbols(self):
        global _SYMBOLS
        if _SYMBOLS is None:
            _SYMBOLS = _build_symbols()
        return _SYMBOLS

    def rate(self, symbol):
        return closed_table_get(self.symbols(), symbol)

    def channels(self):
        return {
            c: Channel(
                name=c,
                scope=("sm" if c in ("MIO", "ASYNC") else "sched"),
                capacity=self.rate("CAP." + c),
            )
            for c in _CHANNEL_SYMBOLS
        }

    def latency(self, atom_class):
        return self.rate("LATENCY." + atom_class)

    def sms(self):
        return self.rate("GEOM.SMS")

    def schedulers_per_sm(self):
        return self.rate("GEOM.SCHEDULERS_PER_SM")

    def warp_slots_per_sm(self):
        return self.rate("GEOM.WARP_SLOTS_PER_SM")

    def threads_per_warp(self):
        return self.rate("GEOM.THREADS_PER_WARP")

    def smem_per_sm(self):
        return self.rate("GEOM.SMEM_PER_SM")

    def smem_per_cta_max(self):
        return self.rate("GEOM.SMEM_PER_CTA_MAX")

    def regfile_words_per_sm(self):
        return self.rate("GEOM.REGFILE_WORDS_PER_SM")

    def clock_mhz(self):
        return self.rate("GEOM.CLOCK_MHZ")

    def peak(self, kind):
        return self.rate("PEAK." + kind.upper())


class Sm80Memory:
    def levels(self):
        nm = lambda s, u: Rate(s, float("nan"), u, "NOT_MEASURED", note="ПЕРЕМЕРИТЬ")
        return (
            MemLevel(
                "smem",
                nm("MEM.smem.BYTES", "байт"),
                nm("MEM.smem.BW", "ГБ/с"),
                nm("MEM.smem.LAT", "такт"),
            ),
            MemLevel(
                "l2",
                nm("MEM.l2.BYTES", "байт"),
                nm("MEM.l2.BW", "ГБ/с"),
                nm("MEM.l2.LAT", "такт"),
            ),
            MemLevel(
                "hbm",
                nm("MEM.hbm.BYTES", "байт"),
                nm("MEM.hbm.BW", "ГБ/с"),
                nm("MEM.hbm.LAT", "такт"),
            ),
        )

    def wavefronts(self, lane_words, width_bytes):
        """ФОРМА закона переносится (32 банка по 4 Б те же), ЦЕНА -- нет.

        Считать по этой форме МОЖНО, но отчёт обязан пометить вывод как опирающийся на
        NOT_MEASURED (цена вайвфронта).
        """
        from ..base import WavefrontCost

        per_bank = {}
        for w in lane_words:
            if w is None:
                continue
            per_bank.setdefault(int(w) % 32, set()).add(int(w))
        degree = float(max((len(s) for s in per_bank.values()), default=1))
        floor = max(width_bytes / 8.0, 1.0)
        return WavefrontCost(degree=degree, floor=floor, wavefronts=max(degree, floor))

    def alignment_rule(self, width_bytes):
        return int(width_bytes)

    def pad_family(self, layout: Layout):
        raise PluginCapabilityError(
            "семейство дополнений sm_80 не объявлено: калибровочная поверхность пуста даже на "
            "sm_70 (замеренных кривых 0), а угадывать аргминимум ЗАПРЕЩЕНО замером "
            "(для шага 68 слов аргминимум по семейству карт = {0,1,2}, единого нет)"
        )

    def swizzle_family(self, layout: Layout):
        raise PluginCapabilityError(
            "свизлы sm_80 не объявлены: с ldmatrix путь smem->фрагмент перестраивается "
            "целиком, и волтовские свизлы к нему не относятся"
        )

    def residency_policy(self):
        return (
            "lru+window"  # cudaAccessPolicyWindow -- рычаг, которого на Volta НЕТ ВОВСЕ
        )


class Sm80TensorUnit:
    def ops(self):
        return ()

    def select(self, in_dtypes, acc_dtype):
        raise PluginCapabilityError(
            "тензорные операции sm_80 не объявлены. Их нельзя объявить без КАРТЫ ФРАГМЕНТА "
            "m16n8k16: из неё выводятся и закон плитки, и размер накопителя, и копировать "
            "волтовское «накопитель = MB*NB*8» ЗАПРЕЩЕНО -- это свойство m8n8k4."
        )


class Sm80Resources:
    def reg_budget(self, warps_per_sm):
        raise PluginCapabilityError(
            "Q(W) на sm_80 не перемерена. Форма min(255, 8*floor(256/W)) ОЖИДАЕТСЯ той же "
            "(файл 65536, гранулярность 256/варп), но ожидание -- не замер."
        )

    def spill_threshold(self, max_live):
        raise PluginCapabilityError(
            "REG.OVERHEAD на sm_80 не перемерен (на sm_70 ровно 7)"
        )

    def free_spills(self):
        raise PluginCapabilityError("число бесплатных разлитых на sm_80 не перемерено")

    def spill_cost(self, n):
        raise PluginCapabilityError("ступенька цены разлива на sm_80 не перемерена")

    def occupancy(self, regs, smem_bytes, threads):
        raise PluginCapabilityError("занятость sm_80 не перемерена")

    def verdict(self, regs, max_live, smem_bytes, threads, min_ctas_per_sm=1):
        raise PluginCapabilityError(
            "ресурсный вердикт sm_80 недоступен: все три закона (Q(W), MaxLive+overhead, порог "
            "второго блока) требуют ЗАМЕРА. Отсекатель обязан МОЛЧАТЬ, а не считать по Volta."
        )

    def wave_quantum(self, occ):
        raise PluginCapabilityError(
            "квант волны sm_80 не подтверждён замером (паспорт 108 SM)"
        )

    def declared_bounds_required(self):
        return True


class Sm80Sync:
    """ЕДИНСТВЕННОЕ, ЧТО КАРКАС УТВЕРЖДАЕТ БЕЗ ЗАМЕРА -- факты ISA, а не величины."""

    def transactions(self):
        return (
            TransactionKind(
                id="gmem->smem",
                issue_op="cp.async.cg.shared.global",
                wait_op="cp.async.wait_group",  # ожидание ЯВНОЕ, в отличие от табло Volta
                granularity="group",
                in_flight_max=not_measured("XFER.IN_FLIGHT", "запрос", "ПЕРЕМЕРИТЬ"),
                occupies={"ASYNC": 1.0, "ISSUE": 1.0},
                consumes_registers=False,  # <-- ФАКТ ISA: cp.async идёт МИМО регистрового файла
                consumes_smem_staging=True,
                depth_axis="async_depth",
            ),
        )

    def barriers(self):
        return (
            BarrierKind(
                "cta",
                "cta",
                phased=False,
                counted=False,
                cost=not_measured("BARRIER.CTA", "такт", "ПЕРЕМЕРИТЬ"),
            ),
            BarrierKind(
                "mbarrier",
                "cta",
                phased=True,
                counted=True,
                cost=not_measured(
                    "BARRIER.MBARRIER",
                    "такт",
                    "ФАЗНЫЙ И СЧЁТНЫЙ -- категории, которой на Volta нет",
                ),
            ),
        )

    def rendezvous_cost(self, barrier_id, participants):
        raise PluginCapabilityError("цена рандеву на sm_80 не перемерена")


class Sm80Skeletons:
    def ops(self):
        return ()

    def axes(self):
        from ..base import Axis

        # Ось объявлена ЗАРАНЕЕ: она и есть доказательство, что ось -- открытый словарь.
        return (Axis("async_depth", frozenset({"ASYNC", "smem"})),)

    def variants(self, op):
        raise PluginCapabilityError(
            "скелетов sm_80 нет. Тело пишется под форму mma, а m16n8k16 -- другая карта "
            "фрагмента, другой loader (ldmatrix) и другая подача (cp.async)."
        )

    def estimate_atoms(self, op, h):
        raise PluginCapabilityError("скелетов sm_80 нет")

    def resources_of(self, op, h):
        raise PluginCapabilityError("скелетов sm_80 нет")

    def launch_of(self, op, h):
        raise PluginCapabilityError("скелетов sm_80 нет")

    def render(self, op, h):
        raise PluginCapabilityError("скелетов sm_80 нет")

    def entry_probe(self, h):
        raise PluginCapabilityError("скелетов sm_80 нет")

    def capabilities(self):
        return frozenset()


class Sm80Toolchain:
    def arch_flags(self):
        return ["-arch=sm_80", "-std=c++17", "-O3"]

    def compile(self, source, out, mode="cubin", extra=None, build_dir=None):
        raise PluginCapabilityError(
            "тулчейн sm_80 не проверен на этой машине: собирать нечего (скелетов нет), а "
            "сборка ради сборки даст бинарь, который никто не запускал"
        )

    def disasm(self, binary):
        raise PluginCapabilityError("разбор SASS sm_80 не реализован (другая ISA)")

    def requirements(self):
        return []


class Sm80Meters:
    def counters(self, kind):
        raise PluginCapabilityError(
            "имена счётчиков sm_80 не выверены. Часть имён l1tex__* совпадает с sm_70, но "
            "совпадение имени НЕ ЕСТЬ совпадение смысла -- проверять на железе."
        )

    def profile(self, binary, kernel, counters):
        raise PluginCapabilityError("приборов sm_80 нет")

    def clock_lock(self, card, mhz):
        raise PluginCapabilityError("фиксация частот sm_80 не проверена")


class Sm80Plugin:
    id = "sm_80"
    contract = CONTRACT
    description = (
        "Ampere / A100 -- КАРКАС. Ни одного замера. Все ставки NOT_MEASURED либо SPEC. "
        "Существует, чтобы граница проверялась ДВУМЯ реализациями, а не одной."
    )

    def __init__(self):
        self.machine = Sm80Machine()
        self.memory = Sm80Memory()
        self.tensor = Sm80TensorUnit()
        self.resources = Sm80Resources()
        self.sync = Sm80Sync()
        self.classifier = None  # опциональная возможность; на sm_80 не реализована
        self.skeletons = Sm80Skeletons()
        self.toolchain = Sm80Toolchain()
        self.meters = Sm80Meters()

    def declared_stubs(self):
        p = data("porting")
        out = ["ВЕСЬ ПЛАГИН -- ЗАГЛУШКА: ни одного замера на железе."]
        out += [
            "ПЕРЕМЕРИТЬ (та же форма закона, другое число): " + x["law"]
            for x in p["class_A_same_law_other_numbers"]
        ]
        out += [
            "ПЕРЕПИСАТЬ (новая категория, не число): %s -- %s" % (x["what"], x["dies"])
            for x in p["class_B_new_category_needs_code"]
        ]
        out += [
            "ПЕРЕВЫВЕСТИ ФАЛЬСИФИКАТОРОМ: %s -- %s" % (x["conclusion"], x["action"])
            for x in p["class_C_conclusions_that_must_be_rederived"]
        ]
        return tuple(out)

    def selftest(self) -> Report:
        r = Report("sm_80 (каркас)")
        M = self.machine
        syms = M.symbols()
        r.check("таблица ставок объявлена", len(syms) > 0, "%d символов" % len(syms))
        r.check(
            "НИ ОДНОЙ ставки MEASURED (иначе каркас врёт)",
            all(s.status != "MEASURED" for s in syms.values()),
        )
        r.check(
            "паспортные ставки помечены SPEC, а не выданы за замер",
            M.peak("hbm_spec").status == "SPEC",
        )
        r.check(
            "полоса, против которой планируют, объявлена НЕ ЗАМЕРЕННОЙ",
            M.peak("hbm_copy").status == "NOT_MEASURED",
        )
        try:
            M.rate("CAP.NONEXISTENT")
            r.check("закрытая таблица отказывает", False)
        except UnknownSymbol:
            r.check("закрытая таблица отказывает", True)

        # Отсекатель обязан МОЛЧАТЬ, а не считать по волтовским числам
        for name, fn in (
            ("ресурсный вердикт", lambda: self.resources.verdict(128, 100, 32768, 256)),
            (
                "тензорная операция",
                lambda: self.tensor.select(("fp16", "fp16"), "fp32"),
            ),
            ("скелеты", lambda: list(self.skeletons.variants(None))),
            ("счётчики", lambda: self.meters.counters("smem_wavefronts")),
            (
                "семейство дополнений",
                lambda: list(self.memory.pad_family(Layout("x", 32, 16, 2))),
            ),
        ):
            try:
                fn()
                r.check("%s отказывает СТРУКТУРНО, а не считает по Volta" % name, False)
            except PluginCapabilityError:
                r.check("%s отказывает СТРУКТУРНО, а не считает по Volta" % name, True)

        # ФАКТЫ ISA, которые каркас вправе утверждать без замера
        tx = self.sync.transactions()[0]
        r.check(
            "cp.async объявлен НЕ съедающим регистры (факт ISA, не величина)",
            tx.consumes_registers is False,
        )
        r.check(
            "ожидание объявлено ЯВНЫМ (в отличие от табло Volta)",
            tx.wait_op == "cp.async.wait_group",
        )
        r.check(
            "ось async_depth объявлена -- ось это ОТКРЫТЫЙ словарь",
            any(a.name == "async_depth" for a in self.skeletons.axes()),
        )
        r.check(
            "mbarrier объявлен ФАЗНЫМ и СЧЁТНЫМ",
            any(b.phased and b.counted for b in self.sync.barriers()),
        )
        r.check(
            "политика резидентности отличается от волтовской (окно L2)",
            self.memory.residency_policy() != "lru",
        )

        # Честность каркаса
        st = self.declared_stubs()
        r.check(
            "перечень заглушек начинается с признания, что замеров нет",
            st[0].startswith("ВЕСЬ ПЛАГИН"),
        )
        r.check(
            "перечень разделён на ПЕРЕМЕРИТЬ / ПЕРЕПИСАТЬ / ПЕРЕВЫВЕСТИ",
            any(s.startswith("ПЕРЕМЕРИТЬ") for s in st)
            and any(s.startswith("ПЕРЕПИСАТЬ") for s in st)
            and any(s.startswith("ПЕРЕВЫВЕСТИ") for s in st),
        )
        return r


_P = None


def load():
    global _P
    if _P is None:
        _P = Sm80Plugin()
    return _P
