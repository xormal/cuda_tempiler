#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ГРАНИЦА ПЛАГИНА -- ЕДИНСТВЕННЫЙ ФАЙЛ, ГДЕ ОНА ОПИСАНА.

Конвейер (`tempo/core/`) не знает ни одного имени архитектуры.  Всё, что замерено на
конкретном железе, живёт в плагине (`tempo/plugins/<arch>/`).  Этот файл -- договор между
ними.

ПРАВИЛО ПРИЁМКИ КОНТРАКТА (оно механически режет спор «а не переусложнили ли»):

    Метод не входит в контракт, пока у него нет (а) вызывающего в core/ и
    (б) второй реализации (nullarch или sm80-заглушка), которая его исполняет.
    Метод контракта без вызывающего -- не граница, а пожелание.

ПРАВИЛО ФАЛЬСИФИКАТОРА (без него двузначное поле контракта -- украшение):

    На каждое двузначное поле контракта обязан существовать плагин-фальсификатор с
    ПРОТИВОПОЛОЖНЫМ значением и тест, что вывод конвейера СДВИНУЛСЯ.
    Пример в дереве: nullarch против nullarch_async, различие ровно в
    TransactionKind.consumes_registers (гейт G8).

ЯЗЫК И КОДИРОВКА.  Значения перечислений -- ASCII: они сравниваются в коде и уезжают в
manifest.json в чужое дерево, где действует правило «не-ASCII отвергается».  Человеческий
текст (`explain`, `note`, отчёты) -- по-русски.

ВЕРСИЯ.  CONTRACT = "tempo/arch/1".  Плагин с неизвестным МАЖОРОМ отвергается на загрузке,
без попытки работать.  Что потребует "tempo/arch/2": кластеры блоков (BarrierKind.scope
закрыт значениями warp|cta), обратная запись управляющих полей SASS, планировщик как
эмиттер.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

CONTRACT = "tempo/arch/1"
CONTRACT_MAJOR = "tempo/arch"

ArchId = str  # "sm_70".  core НИКОГДА не сравнивает эту строку с литералом.


# ============================================================================================
# ИСКЛЮЧЕНИЯ ГРАНИЦЫ
# ============================================================================================
class PluginError(Exception):
    """Общий предок.  Конвейер ловит ЭТИ, а не AttributeError/KeyError."""


class PluginCapabilityError(PluginError):
    """Плагин не умеет того, что просят.  СТРУКТУРНЫЙ отказ, а не падение (гейт G2)."""


class UnknownSymbol(PluginError):
    """Спросили ставку вне закрытой таблицы Machine.symbols() (гейт G4).

    Зачем отказ, а не умолчание: ставка, вернувшаяся по умолчанию, попадает в границу и
    ПРОТЕКАЕТ мимо учёта происхождения.  Замеренная цена такой протечки в проекте --
    сутки на поиск «почему модель врёт на 30 %», а виновата была ставка от ДРУГОГО ядра.
    """


class ContractError(PluginError):
    """Плагин нарушил договор (неизвестный мажор, MEASURED без карты и т.п.)."""


class NotSupported(PluginError):
    """Опциональная возможность плагина не реализована.  Законный ответ, не дефект."""


# ============================================================================================
# 3.1  ПРОИСХОЖДЕНИЕ ЛЮБОГО ЧИСЛА
# ============================================================================================
STATUS = ("MEASURED", "SPEC", "MODEL", "UPPER_BOUND", "NOT_MEASURED")


@dataclass(frozen=True)
class CardState:
    """Состояние карты в момент замера.  Без него число НЕДЕЙСТВИТЕЛЬНО.

    Замерено, почему: одна карта в сервере роняет частоту до 307 МГц против 1530 у соседней
    и даёт 60 % разброса, имитируя сверхлинейный эффект.  И: замер с чужим процессом на
    карте недействителен -- признак соседа берётся по МОЩНОСТИ, а не по utilization.gpu
    (последний бинарен и врёт).
    """

    index: int
    clock_mhz: int
    foreign_procs: int
    date: str


@dataclass(frozen=True)
class Provenance:
    tool: str
    counter: Optional[str]
    version: str
    date: str
    who: str
    observability: Literal["clean", "code_edit", "derived"]
    card: Optional[CardState] = None


@dataclass(frozen=True)
class Rate:
    """Ставка: число + ЕДИНИЦА + СТАТУС + происхождение.

    Число, у которого не названа база сравнения, -- не число.  Этот тип существует затем,
    чтобы «96.5» нельзя было положить рядом с «52.2», не назвав, что 52.2 -- это DP4A.
    """

    symbol: str
    value: float
    units: str
    status: Literal["MEASURED", "SPEC", "MODEL", "UPPER_BOUND", "NOT_MEASURED"]
    prov: Optional[Provenance] = None
    note: str = ""

    def __post_init__(self):
        if self.status not in STATUS:
            raise ContractError("статус %r вне %r" % (self.status, STATUS))

    @property
    def measured(self) -> bool:
        return self.status == "MEASURED"

    def check(self) -> None:
        """Правила, проверяемые загрузчиком.  ОТКАЗ, а не предупреждение."""
        if self.status == "MEASURED":
            if self.prov is None or self.prov.card is None:
                raise ContractError(
                    "%s: MEASURED без карты -- заявка на точность, которой у неё нет"
                    % self.symbol
                )
            if self.prov.card.foreign_procs != 0:
                raise ContractError(
                    "%s: MEASURED снят при %d чужих процессах на карте -- замер недействителен"
                    % (self.symbol, self.prov.card.foreign_procs)
                )


def spec_forbidden_in_time_bound(r: Rate) -> None:
    """SPEC запрещён в любом вычислении ГРАНИЦЫ ВРЕМЕНИ.

    Замерено: паспорт HBM2 -- 900 ГБ/с, достижимое чтение+запись -- 819 ГБ/с.  Граница,
    посчитанная по паспорту, объявляет ядро «на 47 % полосы» там, где оно на 52 %.
    """
    if r.status == "SPEC":
        raise ContractError(
            "%s: SPEC (паспорт) в границе времени запрещён; нужен MEASURED или UPPER_BOUND"
            % r.symbol
        )


# ============================================================================================
# 3.2  АТОМ -- ЕДИНСТВЕННОЕ IR КОНВЕЙЕРА В v1
# ============================================================================================
class AtomKind(str, Enum):
    COMPUTE = "COMPUTE"
    MEM_READ = "MEM_READ"
    MEM_WRITE = "MEM_WRITE"
    XFER_ISSUE = "XFER_ISSUE"  # РАСЩЕПЛЁННАЯ транзакция: выдача...
    XFER_WAIT = "XFER_WAIT"  # ...и ожидание (на Volta неявное, табло)
    BARRIER = "BARRIER"
    BRANCH = "BRANCH"
    SPILL = "SPILL"
    OTHER = "OTHER"


@dataclass(frozen=True)
class Dep:
    src_uid: int
    type: Literal["RAW", "WAR", "WAW", "CTRL", "XFER"]
    distance: int = 0
    min_gap: float = 0.0


@dataclass(frozen=True)
class Atom:
    """Единица нагрузки.  op_id для core НЕПРОЗРАЧЕН ("HMMA.884.F32.F32" -- строка и только).

    ПОЧЕМУ РАСЩЕПЛЁННАЯ ТРАНЗАКЦИЯ ЕСТЬ УЖЕ В v1, ХОТЯ НА VOLTA ЕЁ НЕТ.  На sm_70 подача
    gmem->smem это LDG в регистр и STS из него: она СЪЕДАЕТ РЕГИСТРЫ и потому платит
    занятостью.  На sm_80 cp.async идёт мимо регистров.  Если core закэширует волтовское
    допущение, глубина буфера на A100 будет посчитана дороже, чем она есть, и отсекатель
    выбросит верный вариант.  Поле token + kind XFER_* делает это ДАННЫМИ плагина.
    """

    uid: int
    kind: AtomKind
    op_id: str
    footprint: Mapping[str, float]  # имя канала -> занятых тактов
    latency: float = 0.0
    token: Optional[str] = None  # связывает XFER_ISSUE с его XFER_WAIT; None = табло
    deps: tuple = ()
    region: str = "body"  # "prologue"|"body"|"epilogue"|имя роли


# ============================================================================================
# 3.3  ВОСЕМЬ РАЗДЕЛОВ ПЛАГИНА
# ============================================================================================


# --- 1. МАШИНА -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Channel:
    """Канал -- то, что ЗАНИМАЕТСЯ на промежутке.  Имя НЕПРОЗРАЧНО для core.

    scope="sched" -- ресурс планировщика, "sm" -- ресурс всего SM.  Различие не косметика:
    единицы ёмкости у них РАЗНЫЕ (такт на команду против единиц за такт), и отпечаток обязан
    нести соответствующую величину.  Смешение единиц уже стоило занижения канала в 128 раз.

    ЧЕГО ЗДЕСЬ БОЛЬШЕ НЕ НАПИСАНО И ПОЧЕМУ (закон L-OCCUPANCY-MOVES-BINDING).  Прежняя
    редакция объясняла различие тем, что нагрузка на ресурс SM растёт с занятостью быстрее и
    потому связывающий ресурс с занятостью МЕНЯЕТСЯ.  Замер это опроверг: прогон одного тела
    при 8/16/32 варпах не сдвинул доли каналов ни на процент -- оба вида канала линейны по
    числу варпов, отношение постоянно.  Занятость входит в вердикт по-настоящему только через
    регистровый бюджет.  Связывающий ресурс открывают, а не назначают.
    """

    name: str
    scope: Literal["sched", "sm"]
    capacity: Rate


@runtime_checkable
class Machine(Protocol):
    def arch(self) -> ArchId: ...
    def symbols(
        self,
    ) -> Mapping[str, Rate]: ...  # ЗАКРЫТАЯ таблица; вне неё -> UnknownSymbol
    def channels(self) -> Mapping[str, Channel]: ...
    def latency(self, atom_class: str) -> Rate: ...
    def sms(self) -> Rate: ...
    def schedulers_per_sm(self) -> Rate: ...
    def warp_slots_per_sm(self) -> Rate: ...
    def threads_per_warp(self) -> Rate: ...
    def smem_per_sm(self) -> Rate: ...
    def smem_per_cta_max(self) -> Rate: ...
    def regfile_words_per_sm(self) -> Rate: ...
    def clock_mhz(self) -> Rate: ...
    def peak(self, kind: str) -> Rate: ...  # "tensor_dense","hbm_read","hbm_copy","l2"


# --- 2. ПАМЯТЬ -----------------------------------------------------------------------------
@dataclass(frozen=True)
class WavefrontCost:
    degree: float  # конфликтность
    floor: float  # пол: ширина/8 Б и единица
    wavefronts: float  # max(...)


@dataclass(frozen=True)
class MemLevel:
    name: str
    bytes_: Rate
    bandwidth: Rate
    latency: Rate


@dataclass(frozen=True)
class Layout:
    """Раскладка массива в разделяемой памяти -- ВХОД закона конфликтности."""

    name: str
    row_words: int  # шаг строки в СЛОВАХ (не в элементах и не в байтах)
    rows: int
    elem_bytes: int
    swizzle: Optional[str] = None


@runtime_checkable
class Memory(Protocol):
    def levels(self) -> tuple: ...
    def wavefronts(self, lane_words: Sequence, width_bytes: int) -> WavefrontCost: ...
    def alignment_rule(self, width_bytes: int) -> int: ...
    def pad_family(self, layout: Layout) -> Iterable[int]: ...
    def swizzle_family(self, layout: Layout) -> Iterable[str]: ...
    def residency_policy(self) -> str: ...  # "lru" | ... НЕПРОЗРАЧНО для core


# --- 3. ТЕНЗОРНЫЙ УЗЕЛ ---------------------------------------------------------------------
@dataclass(frozen=True)
class FragmentMap:
    """Карта фрагмента: полоса -> [(строка, столбец, слот регистра)].

    ЗАЧЕМ ОНА В КОНТРАКТЕ, А НЕ КОНСТАНТА В core.  Из карты ВЫВОДЯТСЯ и закон плитки
    (загрузок-фрагмента-на-mma = (MB+NB)/(MB*NB)), и размер накопителя.  На Volta
    (m8n8k4) накопитель = MB*NB*8 float на поток; на m16n8k16 -- ДРУГОЙ.  Кто закэширует
    «*8» в core, тот получит на A100 неверный ресурсный вердикт без единого сообщения.
    """

    a: Callable[[int], Sequence]
    b: Callable[[int], Sequence]
    c: Callable[[int], Sequence]


@dataclass(frozen=True)
class TensorOp:
    id: str
    m: int
    n: int
    k: int
    in_dtypes: tuple
    acc_dtype: str
    frag: FragmentMap
    cost: Rate
    operand_source: Literal["registers", "smem_direct"]
    loader: Optional[str] = None  # None | "manual_lds" | "ldmatrix"
    exact_while: Optional[str] = None  # домен точности, напр. "|sum| < 2**24"


@runtime_checkable
class TensorUnit(Protocol):
    def ops(self) -> tuple: ...
    def select(self, in_dtypes: tuple, acc_dtype: str) -> TensorOp: ...


def acc_regs_per_thread(warp_tile: tuple, op: TensorOp) -> int:
    """ВЫВОД из карты фрагмента, а НЕ константа.  core вправе звать это; плагин -- нет.

    warp_tile = (M, N) в элементах.  Считаем, сколько ячеек C приходится на одну полосу.
    """
    m, n = warp_tile
    per_op = len(op.frag.c(0))  # ячеек C на полосу в ОДНОЙ инструкции
    tiles = (m // op.m) * (n // op.n)
    return per_op * tiles


def operand_loads_per_mma(warp_tile: tuple, op: TensorOp) -> float:
    """Загрузок фрагмента на одну тензорную инструкцию: (MB+NB)/(MB*NB) в единицах op."""
    mb = warp_tile[0] // op.m
    nb = warp_tile[1] // op.n
    if mb <= 0 or nb <= 0:
        raise ContractError("плитка варпа %r мельче формы %s" % (warp_tile, op.id))
    return (mb + nb) / float(mb * nb)


# --- 4. РЕСУРСЫ И ЗАНЯТОСТЬ ----------------------------------------------------------------
RESOURCE_CODES = ("FITS", "SPILL", "WALL_REG", "WALL_SMEM", "NO_BUDGET")


@dataclass(frozen=True)
class Occupancy:
    warps_per_sm: int
    ctas_per_sm: int
    regs_per_thread: int
    limiter: str


@dataclass(frozen=True)
class ResourceVerdict:
    ok: bool
    code: Literal["FITS", "SPILL", "WALL_REG", "WALL_SMEM", "NO_BUDGET"]
    occ: Occupancy
    explain: str  # по-русски, ДЛЯ ОТЧЁТА; core на текст НЕ ветвится

    def __post_init__(self):
        if self.code not in RESOURCE_CODES:
            raise ContractError("код вердикта %r вне %r" % (self.code, RESOURCE_CODES))


@runtime_checkable
class Resources(Protocol):
    def reg_budget(self, warps_per_sm: int) -> int: ...
    def spill_threshold(self, max_live: int) -> int: ...
    def free_spills(self) -> int: ...
    def spill_cost(self, n: int) -> Rate: ...  # СТУПЕНЬКА, не прямая
    def occupancy(self, regs: int, smem_bytes: int, threads: int) -> Occupancy: ...
    def verdict(
        self, regs: int, max_live: int, smem_bytes: int, threads: int
    ) -> ResourceVerdict: ...
    def wave_quantum(self, occ: Occupancy) -> int: ...
    def declared_bounds_required(self) -> bool: ...


# --- 5. СИНХРОНИЗАЦИЯ И ТРАНЗАКЦИИ ---------------------------------------------------------
@dataclass(frozen=True)
class TransactionKind:
    """Как устроена подача данных.  ПОЛЕ-ФАЛЬСИФИКАТОР -- consumes_registers (гейт G8)."""

    id: str  # "gmem->smem"
    issue_op: str
    wait_op: Optional[str]  # None = ожидание неявное (табло Volta)
    granularity: Literal["instruction", "group", "barrier"]
    in_flight_max: Rate
    occupies: Mapping[str, float]  # какие каналы занимает ВЫДАЧА
    consumes_registers: bool
    consumes_smem_staging: bool
    depth_axis: Optional[str] = None  # имя оси поиска, если глубина настраиваема


@dataclass(frozen=True)
class BarrierKind:
    id: str
    scope: Literal["warp", "cta"]  # v1 закрыт; "cluster" потребует tempo/arch/2
    phased: bool
    counted: bool
    cost: Rate


@runtime_checkable
class Sync(Protocol):
    def transactions(self) -> tuple: ...
    def barriers(self) -> tuple: ...
    def rendezvous_cost(self, barrier_id: str, participants: int) -> Rate: ...


# --- 6. КЛАССИФИКАТОР ISA (ОПЦИОНАЛЬНАЯ возможность) ---------------------------------------
@dataclass(frozen=True)
class AtomClass:
    channel: str
    cycles: float
    width_bytes: int
    latency_class: str
    predicated: bool


@runtime_checkable
class Classifier(Protocol):
    def classify(self, instr_text: str) -> AtomClass: ...
    def decode(self, binary: Path, kernel_regex: str) -> Any: ...
    def control_fields(self, word: int) -> Any: ...  # вправе бросить NotSupported

    # encode_control / roundtrip_proof В КОНТРАКТ v1 НЕ ВХОДЯТ:
    # верификатора зависимостей (T3) нет, а неверный wait даёт не падение,
    # а ТИХО НЕВЕРНЫЙ ОТВЕТ.


# --- 7. СКЕЛЕТЫ И ОСИ ----------------------------------------------------------------------
@dataclass(frozen=True)
class Hyperform:
    plugin: ArchId
    params: Mapping[str, Any]  # НЕПРОЗРАЧНЫ для core
    key: str  # устойчивый хэш -> имя ядра и каталога поставки


@dataclass(frozen=True)
class Axis:
    """Ось поиска.  ОТКРЫТЫЙ словарь: sm_80 добавит async_depth без правок core."""

    name: str
    affects: frozenset


@dataclass(frozen=True)
class Launch:
    grid_ctas: int
    threads: int
    smem_bytes: int
    entry: str


@dataclass(frozen=True)
class Rendered:
    source: str
    launch: Launch
    includes: tuple = ()
    notes: str = ""
    # ИМЯ ФАЙЛА ВЫБИРАЕТ ПЛАГИН.  Расширение здесь -- НЕ КОСМЕТИКА: текст с ЯВНОЙ
    # ИНСТАНЦИАЦИЕЙ ядра обязан быть ЕДИНИЦЕЙ ТРАНСЛЯЦИИ (`.cu`), а не заголовком:
    # включённый дважды заголовок даёт дублирующиеся символы.  Умолчание сохраняет прежнее
    # поведение для плагинов, которым нечего инстанцировать.
    filename: str = "kernel.cuh"


@runtime_checkable
class Skeletons(Protocol):
    def ops(self) -> tuple: ...
    def axes(self) -> tuple: ...
    def variants(self, op: "OpSpec") -> Iterator[Hyperform]: ...
    def estimate_atoms(self, op: "OpSpec", h: Hyperform) -> list: ...  # БЕЗ СБОРКИ
    def resources_of(
        self, op: "OpSpec", h: Hyperform
    ) -> tuple: ...  # regs, max_live, smem
    def launch_of(self, op: "OpSpec", h: Hyperform) -> Launch: ...
    def render(self, op: "OpSpec", h: Hyperform) -> Rendered: ...
    def entry_probe(self, h: Hyperform) -> str: ...
    def capabilities(self) -> frozenset: ...  # ТОЛЬКО ДЛЯ ПЕЧАТИ; core НЕ ветвится


# --- 8. ТУЛЧЕЙН И ПРИБОРЫ ------------------------------------------------------------------
@dataclass(frozen=True)
class BuildResult:
    ok: bool
    binary: Optional[Path]
    regs: int
    spill_st: int
    spill_ld: int
    stack_frame: int  # ОТДЕЛЬНО от разлива: кадр бывает при НУЛЕ разливов по отчёту
    smem_static: int
    smem_dynamic: int
    ldl_stl_in_loops: Optional[int]
    log: str


@dataclass(frozen=True)
class EnvReq:
    name: str
    path: Optional[str]
    what: str


@runtime_checkable
class Toolchain(Protocol):
    def arch_flags(self) -> list: ...
    def compile(
        self, source: Path, out: Path, mode: str, extra: list, build_dir: Path
    ) -> BuildResult: ...
    def disasm(self, binary: Path) -> str: ...
    def requirements(self) -> list: ...


@runtime_checkable
class Meters(Protocol):
    def counters(self, kind: str) -> list: ...
    def profile(
        self, binary: Path, kernel: str, counters: list
    ) -> Mapping[str, float]: ...
    def clock_lock(self, card: int, mhz: int) -> ContextManager: ...


# --- СТОРОНА КОНВЕЙЕРА ---------------------------------------------------------------------
@dataclass(frozen=True)
class OpSpec:
    """ОБЪЯВЛЕННАЯ операция.  Продукт распознаёт её, а не читает произвольный C++."""

    kind: str  # "gemm"
    dtype_a: str
    dtype_b: str
    dtype_c: str
    dtype_acc: str
    layout_a: str
    layout_b: str
    layout_c: str
    shapes: Mapping[str, int]  # M, N, K
    scale: Optional[str] = None
    tol_rel_l2: float = 0.0
    coverage: str = "atomic_stamp"


# --- АГРЕГАТ -------------------------------------------------------------------------------
@dataclass
class Report:
    """Отчёт самопроверки плагина.  Плагин обязан уметь проверить СЕБЯ."""

    name: str
    ok: int = 0
    total: int = 0
    lines: list = field(default_factory=list)

    def check(self, label: str, cond: bool, note: str = "") -> bool:
        self.total += 1
        if cond:
            self.ok += 1
        self.lines.append(("ok   " if cond else "ПАДЁТ", label, note))
        return bool(cond)

    @property
    def green(self) -> bool:
        return self.total > 0 and self.ok == self.total

    def render(self) -> str:
        out = ["САМОПРОВЕРКА ПЛАГИНА %s" % self.name]
        for st, label, note in self.lines:
            out.append("  %-6s %s%s" % (st, label, ("  -- " + note) if note else ""))
        out.append("  ИТОГ: %d/%d" % (self.ok, self.total))
        return "\n".join(out)


@runtime_checkable
class Plugin(Protocol):
    id: ArchId
    contract: str
    description: str
    machine: Machine
    memory: Memory
    tensor: TensorUnit
    resources: Resources
    sync: Sync
    classifier: Optional[Classifier]
    skeletons: Skeletons
    toolchain: Toolchain
    meters: Meters

    def declared_stubs(self) -> tuple: ...  # ЧЕСТНЫЙ перечень нереализованного
    def selftest(self) -> Report: ...


# ============================================================================================
# ХЕЛПЕРЫ ДЛЯ ПЛАГИНОВ (не часть границы; чтобы не писать одно и то же пять раз)
# ============================================================================================
def rate(symbol, value, units, status, note="", prov=None) -> Rate:
    return Rate(
        symbol=symbol,
        value=float(value),
        units=units,
        status=status,
        note=note,
        prov=prov,
    )


def not_measured(symbol, units, note="") -> Rate:
    """Ставка, которой НЕТ.  Печатается как таковая и помечает вердикт в шапке отчёта."""
    return Rate(
        symbol=symbol, value=float("nan"), units=units, status="NOT_MEASURED", note=note
    )


def closed_table_get(table: Mapping[str, Rate], symbol: str) -> Rate:
    """Доступ к ЗАКРЫТОЙ таблице ставок: вне таблицы -- отказ, а не умолчание (гейт G4)."""
    try:
        return table[symbol]
    except KeyError:
        raise UnknownSymbol(
            "ставка %r не объявлена; закрытая таблица содержит %d символов"
            % (symbol, len(table))
        ) from None
