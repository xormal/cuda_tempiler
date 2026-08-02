#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ПОДСТАВНАЯ АРХИТЕКТУРА -- фальсификатор гейтов G3 и G6.

Ни одно её число не совпадает с настоящими.  Это и есть смысл: если конвейер где-то ЗНАЕТ
про 32 банка, 80 процессоров, 255 регистров или про то, что накопитель равен произведению
плиток на восемь, -- он ошибётся ЗДЕСЬ, и ошибётся видимо.

    процессоров            13   (а не 80 и не 108)
    каналы                 X, Y, Z  (Z -- ресурс всего процессора, как и MIO у настоящих)
    банков                 7 по 8 байт  (а не 32 по 4)
    тензорная форма        m4n4k2, 2 накопителя на полосу
    бюджет регистров       Q(W) = min(99, 6*floor(128/W))
    опкоды                 AAA / BBB / CCC
    планировщиков          3
    варп                   11 полос  (чтобы «32» нигде не подразумевалось)

ПЕРИОД СЧИТАН РУКОЙ.  Тело: 6 атомов AAA (канал X, 3 такта каждый) + 4 атома BBB (канал Y,
5 тактов) + 2 атома CCC (канал Z, 7 тактов, ресурс процессора, 2 варпа).
    X: 6*3  = 18
    Y: 4*5  = 20
    Z: (2*7)*2 варпа / 1.0 = 28     <- связывает
    T >= 28, связывающий канал Z.
Гейт G3 требует буквально этого числа и этого имени канала.
"""

from __future__ import annotations

import itertools

from ..base import (
    CONTRACT,
    Atom,
    AtomKind,
    Axis,
    BarrierKind,
    Channel,
    FragmentMap,
    Hyperform,
    Launch,
    Layout,
    MemLevel,
    Occupancy,
    PluginCapabilityError,
    Rate,
    Rendered,
    Report,
    ResourceVerdict,
    TensorOp,
    TransactionKind,
    UnknownSymbol,
    WavefrontCost,
    acc_regs_per_thread,
    closed_table_get,
    operand_loads_per_mma,
)

# --------------------------------------------------------------------------------------------
# ЧИСЛА ПОДСТАВНОЙ МАШИНЫ.  Все MODEL: карты нет, значит MEASURED быть не может.
# --------------------------------------------------------------------------------------------
PROCS = 13
SCHEDULERS = 3
LANES = 11
WARP_SLOTS = 17
BANKS = 7
BANK_BYTES = 8
REG_FILE_WORDS = 4096
REG_LIMIT = 99
REG_GRAN = 6
REG_OVERHEAD = 3
FREE_SPILLS = 1
SMEM_PER_PROC = 5000
SMEM_PER_CTA_MAX = 4800

_ARCH = "nullarch"


def _r(sym, val, units, note=""):
    return Rate(sym, float(val), units, "MODEL", note=note or "подставная величина, замера нет")


_SYMBOLS = {
    "CAP.X": _r("CAP.X", 3.0, "такт/атом/планировщик"),
    "CAP.Y": _r("CAP.Y", 5.0, "такт/атом/планировщик"),
    "CAP.Z": _r("CAP.Z", 1.0, "единиц/такт на ПРОЦЕССОР"),
    "LATENCY.AAA": _r("LATENCY.AAA", 9.0, "такт"),
    "LATENCY.BBB": _r("LATENCY.BBB", 4.0, "такт"),
    "LATENCY.CCC": _r("LATENCY.CCC", 21.0, "такт"),
    "GEOM.SMS": _r("GEOM.SMS", PROCS, "процессор"),
    "GEOM.SCHEDULERS_PER_SM": _r("GEOM.SCHEDULERS_PER_SM", SCHEDULERS, "планировщик/процессор"),
    "GEOM.WARP_SLOTS_PER_SM": _r("GEOM.WARP_SLOTS_PER_SM", WARP_SLOTS, "слот"),
    "GEOM.THREADS_PER_WARP": _r("GEOM.THREADS_PER_WARP", LANES, "полоса/пучок"),
    "GEOM.SMEM_PER_SM": _r("GEOM.SMEM_PER_SM", SMEM_PER_PROC, "байт"),
    "GEOM.SMEM_PER_CTA_MAX": _r("GEOM.SMEM_PER_CTA_MAX", SMEM_PER_CTA_MAX, "байт"),
    "GEOM.REGFILE_WORDS_PER_SM": _r("GEOM.REGFILE_WORDS_PER_SM", REG_FILE_WORDS, "слово"),
    "GEOM.REG_ISA_LIMIT": _r("GEOM.REG_ISA_LIMIT", REG_LIMIT, "регистр"),
    "GEOM.REG_ALLOC_GRANULARITY": _r("GEOM.REG_ALLOC_GRANULARITY", REG_GRAN, "регистр"),
    "GEOM.CLOCK_MHZ": _r("GEOM.CLOCK_MHZ", 777, "МГц"),
    "PEAK.TENSOR_DENSE": _r("PEAK.TENSOR_DENSE", 3.5, "условных единиц"),
    "PEAK.HBM_COPY": _r("PEAK.HBM_COPY", 11.0, "условных единиц"),
    "REG.OVERHEAD": _r("REG.OVERHEAD", REG_OVERHEAD, "регистр"),
    "REG.FREE_SPILLS": _r("REG.FREE_SPILLS", FREE_SPILLS, "значение"),
    "TENSOR.COST": _r("TENSOR.COST", 7.0, "такт/атом/процессор"),
}


class NullarchMachine:
    def arch(self):
        return _ARCH

    def symbols(self):
        return _SYMBOLS

    def rate(self, s):
        return closed_table_get(_SYMBOLS, s)

    def channels(self):
        return {
            "X": Channel("X", "sched", self.rate("CAP.X")),
            "Y": Channel("Y", "sched", self.rate("CAP.Y")),
            "Z": Channel("Z", "sm", self.rate("CAP.Z")),
        }

    def latency(self, cls):
        return self.rate("LATENCY." + cls)

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


class NullarchMemory:
    def levels(self):
        return (
            MemLevel("scratch", _r("MEM.scratch.BYTES", SMEM_PER_PROC, "байт"),
                     _r("MEM.scratch.BW", 1.0, "ед/такт"), _r("MEM.scratch.LAT", 13.0, "такт")),
        )

    def wavefronts(self, lane_words, width_bytes):
        per = {}
        for w in lane_words:
            if w is None:
                continue
            per.setdefault(int(w) % BANKS, set()).add(int(w))
        degree = float(max((len(s) for s in per.values()), default=1))
        floor = max(width_bytes / float(BANK_BYTES), 1.0)
        return WavefrontCost(degree, floor, max(degree, floor))

    def alignment_rule(self, width_bytes):
        return int(width_bytes)

    def pad_family(self, layout):
        from tempo.core.model.banks import pad_candidates

        return pad_candidates(layout.row_words, BANKS)

    def swizzle_family(self, layout):
        return ("none", "rotate")

    def residency_policy(self):
        return "fifo"  # НЕ lru -- чтобы вывод, завязанный на lru, здесь заметно ошибся


def _fa(l):
    return ((l % 4, 0, 0),)


def _fb(l):
    return ((0, l % 4, 0),)


def _fc(l):
    """ДВА накопителя на полосу (а не восемь): любой, кто закэшировал '*8', ошибётся вчетверо."""
    return ((l % 4, (l // 4) % 4, 0), (l % 4, (l // 4) % 4 + 1, 1))


class NullarchTensor:
    def ops(self):
        return (
            TensorOp(
                id="CCC.m4n4k2",
                m=4, n=4, k=2,
                in_dtypes=("q7", "q7"),
                acc_dtype="w21",
                frag=FragmentMap(a=_fa, b=_fb, c=_fc),
                cost=_SYMBOLS["TENSOR.COST"],
                operand_source="smem_direct",
                loader=None,
                exact_while="|sum| < 3**7",
            ),
        )

    def select(self, in_dtypes, acc_dtype):
        for o in self.ops():
            if o.in_dtypes == tuple(in_dtypes) and o.acc_dtype == acc_dtype:
                return o
        raise PluginCapabilityError("nullarch: нет тензорной операции %r/%r" % (in_dtypes, acc_dtype))


class NullarchResources:
    def reg_budget(self, warps_per_sm):
        if warps_per_sm <= 0:
            return REG_LIMIT
        return min(REG_LIMIT, REG_GRAN * (128 // warps_per_sm))

    def spill_threshold(self, max_live):
        return int(max_live + REG_OVERHEAD)

    def free_spills(self):
        return FREE_SPILLS

    def spill_cost(self, n):
        return _r("SPILL.COST", 1.0 if n <= FREE_SPILLS else 2.5, "кратность")

    def occupancy(self, regs, smem_bytes, threads):
        warps_per_cta = max(1, threads // LANES)
        by_reg = REG_FILE_WORDS // max(1, regs) // LANES
        by_smem = (SMEM_PER_PROC // max(1, smem_bytes)) * warps_per_cta if smem_bytes else WARP_SLOTS
        w = max(0, min(WARP_SLOTS, by_reg, by_smem))
        limiter = "regs" if by_reg <= by_smem else "smem"
        return Occupancy(int(w), int(w // warps_per_cta), int(regs), limiter)

    def verdict(self, regs, max_live, smem_bytes, threads):
        occ = self.occupancy(regs, smem_bytes, threads)
        if smem_bytes > SMEM_PER_CTA_MAX:
            return ResourceVerdict(False, "WALL_SMEM", occ,
                                   "нужно %d Б при потолке %d Б" % (smem_bytes, SMEM_PER_CTA_MAX))
        if occ.warps_per_sm <= 0 or occ.ctas_per_sm <= 0:
            return ResourceVerdict(False, "NO_BUDGET", occ, "ни один блок не резидентен")
        need = self.spill_threshold(max_live)
        if need > REG_LIMIT:
            return ResourceVerdict(False, "WALL_REG", occ,
                                   "нужно %d > потолка %d" % (need, REG_LIMIT))
        budget = self.reg_budget(occ.warps_per_sm)
        if need > budget:
            over = need - budget
            return ResourceVerdict(over <= FREE_SPILLS, "SPILL", occ,
                                   "разлив %d значений (нужно %d, бюджет %d)" % (over, need, budget))
        return ResourceVerdict(True, "FITS", occ,
                               "нужно %d, бюджет %d при %d пучках" % (need, budget, occ.warps_per_sm))

    def wave_quantum(self, occ):
        return PROCS * max(1, occ.ctas_per_sm)

    def declared_bounds_required(self):
        return False  # ещё одно отличие от боевой машины


CONSUMES_REGISTERS = True  # <-- ПОЛЕ-ФАЛЬСИФИКАТОР; nullarch_async переопределяет его


def make_transaction(consumes_registers: bool):
    return TransactionKind(
        id="far->near",
        issue_op="FETCH",
        wait_op=None if consumes_registers else "FETCH.WAIT",
        granularity="instruction" if consumes_registers else "group",
        in_flight_max=_r("XFER.IN_FLIGHT", 4, "запрос"),
        occupies={"Z": 1.0},
        consumes_registers=consumes_registers,
        consumes_smem_staging=not consumes_registers,
        depth_axis="depth",
    )


class NullarchSync:
    CONSUMES_REGISTERS = True

    def transactions(self):
        return (make_transaction(self.CONSUMES_REGISTERS),)

    def barriers(self):
        return (BarrierKind("cta", "cta", False, False, _r("BARRIER.CTA", 6.0, "такт")),)

    def rendezvous_cost(self, barrier_id, participants):
        if barrier_id != "cta":
            raise PluginCapabilityError("nullarch: барьера %r нет" % barrier_id)
        return _r("BARRIER.CTA", 6.0 + participants, "такт")


# --------------------------------------------------------------------------------------------
# СКЕЛЕТ.  Одна ось (`depth`) и один параметр плитки -- достаточно, чтобы конвейер прошёл
# ЦЕЛИКОМ и принял решение, а гейт G8 увидел сдвиг вердикта.
# --------------------------------------------------------------------------------------------
AXES = (
    Axis("tile", frozenset({"X", "Y", "smem"})),
    Axis("depth", frozenset({"Z", "regs"})),
)


class NullarchSkeletons:
    SYNC = NullarchSync

    def ops(self):
        return ("gemm",)

    def axes(self):
        return AXES

    def capabilities(self):
        return frozenset({"handmade_period"})

    def variants(self, op):
        if op is None or getattr(op, "kind", None) != "gemm":
            raise PluginCapabilityError("nullarch умеет только 'gemm'")
        for tile, depth in itertools.product((4, 8), (1, 2, 3)):
            yield Hyperform(plugin=_ARCH, params={"tile": tile, "depth": depth},
                            key="t%dd%d" % (tile, depth))

    def estimate_atoms(self, op, h):
        """ТЕЛО, ПЕРИОД КОТОРОГО СЧИТАН РУКОЙ (см. шапку модуля).

        При tile=4, depth=1: 6 атомов AAA (X), 4 атома BBB (Y), 2 атома CCC (Z).
        """
        t = h.params["tile"]
        d = h.params["depth"]
        n_a = 6 * (t // 4)
        n_b = 4 * (t // 4)
        n_c = 2 * (t // 4)
        z_cost = _SYMBOLS["TENSOR.COST"].value  # 7 единиц канала Z на атом CCC
        # ЕДИНИЦЫ РАЗНЫЕ, И ЭТО НАМЕРЕННО (ровно на смешении единиц ломаются настоящие
        # плагины): у каналов планировщика ёмкость -- ТАКТОВ НА АТОМ, поэтому отпечаток
        # несёт ЧИСЛО АТОМОВ; у канала процессора ёмкость -- ЕДИНИЦ ЗА ТАКТ, поэтому
        # отпечаток несёт ЧИСЛО ЕДИНИЦ.
        return [
            Atom(0, AtomKind.COMPUTE, "AAA x%d" % n_a, {"X": float(n_a)}, 9.0),
            Atom(1, AtomKind.COMPUTE, "BBB x%d" % n_b, {"Y": float(n_b)}, 4.0),
            Atom(2, AtomKind.XFER_ISSUE, "CCC x%d" % (n_c * d),
                 {"Z": float(n_c * d) * z_cost}, 21.0),
        ]

    def resources_of(self, op, h):
        """Глубина подачи стоит регистров ТОЛЬКО если транзакция их съедает.

        Это и есть узел, который фальсифицирует G8: на nullarch_async та же глубина
        не стоит ни одного регистра, и вердикт обязан отличаться.
        """
        t, d = h.params["tile"], h.params["depth"]
        tx = self.SYNC().transactions()[0]
        base = 20 + 6 * t
        per_depth = 24 if tx.consumes_registers else 0
        regs = base + per_depth * (d - 1)
        smem = 400 * t + (300 * (d - 1) if tx.consumes_smem_staging else 0)
        return int(regs), int(regs - REG_OVERHEAD), int(smem)

    def launch_of(self, op, h):
        return Launch(grid_ctas=PROCS * 2, threads=LANES * 2, smem_bytes=self.resources_of(op, h)[2],
                      entry="nullarch_%s" % h.key)

    def render(self, op, h):
        return Rendered(source="// nullarch %s\n" % h.key, launch=self.launch_of(op, h),
                        includes=(), notes="подставная архитектура: собирать нечего")

    def entry_probe(self, h):
        return "nullarch_%s" % h.key


class NullarchToolchain:
    def arch_flags(self):
        return ["--nullarch"]

    def compile(self, *a, **k):
        raise PluginCapabilityError("nullarch: собирать нечего, это подставная архитектура")

    def disasm(self, b):
        raise PluginCapabilityError("nullarch: разбирать нечего")

    def requirements(self):
        return []


class NullarchMeters:
    def counters(self, kind):
        if kind != "z_units":
            raise PluginCapabilityError("nullarch: группы счётчиков %r нет" % kind)
        return ["nullarch__z_units.sum"]

    def profile(self, *a, **k):
        raise PluginCapabilityError("nullarch: приборов нет")

    def clock_lock(self, card, mhz):
        raise PluginCapabilityError("nullarch: карты нет")


class NullarchPlugin:
    id = _ARCH
    contract = CONTRACT
    description = "Подставная архитектура: 13 процессоров, каналы X/Y/Z, 7 банков по 8 Б, форма m4n4k2."

    Sync = NullarchSync
    Skeletons = NullarchSkeletons

    def __init__(self):
        self.machine = NullarchMachine()
        self.memory = NullarchMemory()
        self.tensor = NullarchTensor()
        self.resources = NullarchResources()
        self.sync = self.Sync()
        self.classifier = None
        self.skeletons = self.Skeletons()
        self.skeletons.SYNC = self.Sync
        self.toolchain = NullarchToolchain()
        self.meters = NullarchMeters()

    def declared_stubs(self):
        return ("вся архитектура подставная: сборки, приборов и карты нет по построению",)

    def selftest(self) -> Report:
        r = Report(self.id)
        from tempo.core.model.bound import bound

        op = _demo_op()
        h = next(iter(self.skeletons.variants(op)))
        atoms = self.skeletons.estimate_atoms(op, h)
        b = bound(atoms, self.machine.channels(), warps_per_sm=2)
        r.check("ПЕРИОД, СЧИТАННЫЙ РУКОЙ: T >= 28", abs(b.T - 28.0) < 1e-9, "получено %.2f" % b.T)
        r.check("связывающий канал -- Z (ресурс процессора)", b.binding == "Z", b.binding)
        r.check("ни одна ставка не выдана за замер",
                all(s.status != "MEASURED" for s in self.machine.symbols().values()))
        try:
            self.machine.rate("CAP.TENSOR")
            r.check("закрытая таблица отказывает на чужом символе", False)
        except UnknownSymbol:
            r.check("закрытая таблица отказывает на чужом символе", True)
        o = self.tensor.ops()[0]
        r.check("накопитель выводится из КАРТЫ: плитка 4x4 -> 2 на полосу",
                acc_regs_per_thread((4, 4), o) == 2)
        r.check("загрузок-на-mma при плитке 8x8 = 1.00",
                abs(operand_loads_per_mma((8, 8), o) - 1.0) < 1e-9)
        r.check("банков 7 по 8 Б: шаг 7 слов даёт ПОЛНЫЙ конфликт",
                self.memory.wavefronts([l * 7 for l in range(LANES)], 8).degree == float(LANES))
        r.check("Q(W): бюджет при 16 пучках", self.resources.reg_budget(16) == min(REG_LIMIT, 6 * 8))
        v = self.resources.verdict(60, 57, 2000, LANES * 2)
        r.check("ресурсный вердикт считается", v.code in ("FITS", "SPILL"), v.explain)
        return r


def _demo_op():
    from ..base import OpSpec

    return OpSpec(kind="gemm", dtype_a="q7", dtype_b="q7", dtype_c="q7", dtype_acc="w21",
                  layout_a="k", layout_b="k", layout_c="n", shapes={"M": 64, "N": 64, "K": 16},
                  tol_rel_l2=1e-3)


_P = None


def load():
    global _P
    if _P is None:
        _P = NullarchPlugin()
    return _P
