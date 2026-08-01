#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tempo -- МОДЕЛЬ ВРЕМЕННОГО ОТПЕЧАТКА для sm_70 (веха M1).

ЧТО ЭТО. Ядро представляется не последовательностью команд, а МНОЖЕСТВОМ АТОМОВ, каждый из
которых занимает набор ячеек (канал, такт).  Период установившегося цикла T -- минимальное
число тактов, за которое все атомы одной итерации помещаются в свои ячейки без наложений и с
соблюдением предшествований.  Модель даёт ДВЕ границы:

    ResMII = max по каналам ( нагрузка канала / его ёмкость )     -- "объём не влезает"
    RecMII = max по циклам графа зависимостей ( сумма задержек / число оборотов )
    T >= max(ResMII, RecMII)

Это в точности нижняя граница modulo-расписания; она ДОСТИЖИМА оптимальным расписанием для
класса задач, который назовёт §4 README, и НЕ достигается ptxas -- разрыв меряется отдельно и
печатается как "запас".

КАНАЛЫ И ИХ ЁМКОСТИ (все ЗАМЕРЕНЫ стендом tools/probe.cu, ни один не взят из документации):
    TENSOR  тензорный конвейер   2.00 такта на HMMA.884 на ПЛАНИРОВЩИК   [замер: 1.95 HMMA/такт/SM]
    ALU     целочисленный/логика 2.00 такта на команду на ПЛАНИРОВЩИК    [16 полос из 64 на SM]
    FPU     fp32                 2.00 такта на команду на ПЛАНИРОВЩИК
    SFU     MUFU/трансцендент    8.00 такта на команду на ПЛАНИРОВЩИК
    MIO     разделяемая + SHFL   128 БАЙТ за такт на ВЕСЬ SM            [замер: 6 точек, <1%]
    ISSUE   порт выдачи          1.00 такта на команду на ПЛАНИРОВЩИК

ВАЖНО, ЧТО MIO -- РЕСУРС SM, А НЕ ПЛАНИРОВЩИКА. Из-за этого нагрузка на него растёт с числом
варпов вчетверо быстрее, чем на прочие каналы, и связывающий ресурс МЕНЯЕТСЯ с занятостью.
Это то самое "связывающий ресурс надо открывать, а не назначать".

ЗАДЕРЖКИ (для RecMII), замерены лестницей цепей c1..c32 стенда:
    LOP3/IADD3/прочая целочисленная  4 такта
    FFMA/FADD/FMUL                   4 такта
    LDS                             26 тактов (замер s1x32 при одном варпе)
    HMMA                             8 тактов (оценка сверху; на период почти не влияет)

ЗАПУСК
    python3 tempo.py <cubin|so> --kernel <regex> --warps 8 [--json out.json]
    python3 tempo.py --selftest            # сверка модели с замерами стенда probe.cu
"""

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor")
)
import issue_slots as IS  # noqa: E402  (вендорено из fa2_sm70_cutlass_grade/tools/)

SCHEDULERS = 4

# --------------------------------------------------------------------------------------------
# РЕГИСТРОВЫЙ ПУЛ -- ЧЕТВЁРТЫЙ РЕСУРС (замерено 2026-08-01, tools/probe.cu, карта 1).
#
# Он НЕ канал с ёмкостью: превышение оплачивается не тактом, а РАЗЛИВОМ, и цена скачкообразна.
# Замер (лестница c1xNP: NP значений, живых через обратную дугу, при 13 бюджетах и 4 занятостях):
#
#   * БЮДЖЕТ НА НИТЬ задаётся ЗАНЯТОСТЬЮ:  Q(W) = min(255, 8*floor(256/W)).
#     Проверено прямым опросом ptxas при __launch_bounds__(W*32,1): W=12->168, 16->128, 24->80,
#     32->64 -- совпадение точное на всех четырёх.
#   * ИЗЛОМ по числу одновременно живых значений:  MaxLive* = Q - REG_OVERHEAD, REG_OVERHEAD = 7.
#     Найден компиляторным свидетелем (первое появление LDL/STL в теле) с точностью до ЕДИНИЦЫ:
#     минимальный бюджет без разлива равен NP+9 при NP = 40, 96, 192 (cap NP+8 -> 2 LDL, NP+9 -> 0),
#     а MaxLive, который считает наш планировщик по SASS, равен NP+2 на всех трёх -- отсюда
#     требуемых регистров = MaxLive + 7 РОВНО.  По секундомеру: при 13 бюджетах последняя чистая
#     NP = cap-8, первая с разливом NP = cap; при launch_bounds на W=12/16/24/32 то же.
#   * ЦЕНА ПЕРЕСЕЧЕНИЯ: скачок ~x5 на ОДНОМ шаге (cap=40: NP=32 -> 136.1 такта, NP=40 -> 697.9),
#     дальше рост сверхлинейный (cap=40, NP=240 -> 73391 против 969 без ограничения, x75).
#     Одно разлитое значение стоит 27-32 такта на итерацию против 4 тактов у неразлитого (x7-8).
#   * ПЛАТО ДО ИЗЛОМА -- РОВНО КАНАЛЬНАЯ МОДЕЛЬ: наклон 3.98 такта на живое значение при
#     предсказанных 4.00 (ALU 2 такта * 2 варпа на планировщик).
# --------------------------------------------------------------------------------------------
REG_ISA_LIMIT = 255
REG_OVERHEAD = 7  # регистров сверх MaxLive модели (замер: ровно 7 на трёх точках)


def reg_budget(warps_per_sm):
    """Бюджет регистров НА НИТЬ при заданной занятости.  Гранулярность Volta -- 256 регистров на
    варп, то есть 8 на нить; всего 65536 на SM.  Подтверждено ptxas на W=12/16/24/32."""
    w = max(1, int(warps_per_sm))
    return min(REG_ISA_LIMIT, 8 * (256 // w))


def reg_verdict(maxlive, warps_per_sm):
    """-> (бюджет, порог живых значений, влезает ли).  Порог -- ИЗМЕРЕННЫЙ излом, не оценка."""
    q = reg_budget(warps_per_sm)
    thr = q - REG_OVERHEAD
    return q, thr, (maxlive is not None and maxlive <= thr)


# --------------------------------------------------------------------------------------------
# 1. ОТПЕЧАТОК КОМАНДЫ: какие ячейки какого канала она занимает
# --------------------------------------------------------------------------------------------
# (канал, тактов на команду, "на планировщик" | "на SM")
CH_TENSOR, CH_ALU, CH_FPU, CH_SFU, CH_MIO, CH_LSU, CH_ISSUE, CH_BRANCH = (
    "TENSOR",
    "ALU",
    "FPU",
    "SFU",
    "MIO",
    "LSU",
    "ISSUE",
    "BRANCH",
)

CAP_PER_SCHED = {
    CH_TENSOR: 1.0,
    CH_ALU: 1.0,
    CH_FPU: 1.0,
    CH_SFU: 1.0,
    CH_ISSUE: 1.0,
    CH_LSU: 1.0,
    CH_BRANCH: 1.0,
}
MIO_BYTES_PER_CYCLE = 128.0  # на весь SM  [замерено]

# --------------------------------------------------------------------------------------------
# КАНАЛ MIO МЕРЯЕТСЯ НЕ БАЙТАМИ, А ВАЙВФРОНТАМИ.  [замерено секундомером И счётчиком ncu]
#
# Прежняя редакция этого файла содержала ПОДГОНКУ ("многоадресность даёт ровно x2 и насыщается").
# Она описывала данные и была НЕВЕРНА как механизм: ось шага (STRB) её опровергла, а счётчик
# l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld подтвердил настоящий закон с точностью до
# третьего знака на 17 телах:
#
#     ВАЙВФРОНТОВ НА КОМАНДУ = max( КОНФЛИКТНОСТЬ , ширина_на_полосу / 8 Б , 1 )
#     конвейер разделяемой памяти отдаёт РОВНО ОДИН ВАЙВФРОНТ ЗА ТАКТ НА SM
#
# где КОНФЛИКТНОСТЬ = максимум по 32 банкам числа РАЗЛИЧНЫХ 4-байтовых слов, запрошенных в
# этом банке.  Три слагаемых -- три разных предела:
#   * конфликтность -- пропускная банкового массива; РАССЫЛКА ОДНОГО АДРЕСА БЕСПЛАТНА ПОЛНОСТЬЮ
#     (при шаге 128 Б цена падает как 32/DUP от 2048 до 64.8 тактов -- выигрыш x32, а не x2);
#   * ширина/8 -- обратный путь в регистровый файл: LDS.128 стоит >= 2 вайвфронта ДАЖЕ при
#     полной рассылке (ncu: 2.000 при 32 полосах на одном адресе), LDS.64 >= 1, LDS.32 >= 1;
#   * 1 -- пол на команду.
#
# ЧЕМ ЭТО ПОДТВЕРЖДЕНО (ncu, вайвфронтов на команду; ожидание -> замер):
#   шаг 4/8/32/128 Б при DUP=1, LDS.32 (128 уникальных байт ВО ВСЕХ):  1/2/8/32 -> 1.000/2.000/8.000/32.000
#   шаг 132 Б (дополнение на слово):                                    1       -> 1.000
#   шаг 128 Б, DUP=32 (полная рассылка):                                1       -> 1.000
#   LDS.128 при полной рассылке:                                        2       -> 2.000
#
# ЧТО ИЗ ЭТОГО СЛЕДУЕТ ДЛЯ ГРАНИЦЫ.  Конфликтность из SASS без разбора адресов не выводится,
# поэтому в НИЖНЮЮ границу входит её минимум (=1), то есть цена max(ширина/8, 1) вайвфронта.
# Это ЗАКОННО и совпадает по числам с прежней "безопасной" ставкой -- но теперь у неё есть
# механизм, а не подгонка.  КОМПИЛЯТОР адреса ЗНАЕТ, поэтому для него модель ТОЧНА; неполной
# она является только для чтения чужого SASS постфактум.
#
# ПРОВЕРКА НА БОЕВЫХ ЯДРАХ (ncu, вайвфронтов на LDS):
#   volta_fwd_block  замер 2.054, ставка "все полосы различны" 2.054 -- СОВПАДЕНИЕ ТОЧНОЕ;
#   volta_fwd_ws     замер 2.221, та же ставка 1.874 -- замер ВЫШЕ на 18.5 % (банковые конфликты).
# То есть в наших ядрах СОВПАДАЮЩИХ АДРЕСОВ МЕЖДУ ПОЛОСАМИ НЕТ ВОВСЕ, и прежняя ставка
# "ширина*32 байт" верна для volta_fwd_block и ЗАНИЖАЕТ для volta_fwd_ws.
# --------------------------------------------------------------------------------------------
MIO_WAVEFRONT_BYTES = (
    128.0  # вайвфронт несёт не более 128 Б; конвейер -- 1 вайвфронт/такт/SM
)
MIO_SAFE = (
    True  # True -- ГРАНИЦА (конфликтность = 1); False -- ставка "все полосы различны"
)


def mio_wavefronts(width_bytes, degree=1):
    """Вайвфронтов на одну команду разделяемой памяти.  degree -- конфликтность (>=1)."""
    return max(float(degree), width_bytes / 8.0, 1.0)


# --------------------------------------------------------------------------------------------
# 1c. КАЛИБРОВКА: СТАВКИ ИЗ ЗАМЕРА ВМЕСТО ЗАШИТЫХ  (tools/calib.py, data/calib/*.json)
#
# ЗАЧЕМ. Модель укладывает операции в решётку (канал, такт), и для этого ей нужна ЦЕНА ЗАНЯТИЯ
# КАНАЛА. Замерено, что цена НЕ КОНСТАНТА: при неизменных 128 уникальных байтах один только шаг
# раскладки двигает её в 32 раза. Значит часть ставок обязана приходить ИЗ ЗАМЕРА, а модель обязана
# ГОВОРИТЬ, какие именно пришли, а какие остались её собственным допущением. Ставка без пометки
# происхождения -- это заявка на точность, которой у неё нет.
#
# ЧТО СТАВКА, А ЧТО НЕТ. Ниже перечислено ВСЁ, что модель считает ценой; поле "измеримо" говорит,
# существует ли для величины замер в принципе, а не есть ли он у нас сейчас.
# --------------------------------------------------------------------------------------------
MODEL_RATES = {
    # символ                    (описание,                                       измеримо)
    "SCHEDULERS": ("планировщиков на SM", True),
    "MIO_BYTES_PER_CYCLE": ("пропускная разделяемой памяти, Б/такт на SM", True),
    "MIO_WAVEFRONT_BYTES": ("байт в вайвфронте", True),
    "REG_ISA_LIMIT": ("потолок регистров ISA", True),
    "REG_OVERHEAD": ("регистров сверх MaxLive", True),
    "CAP_PER_SCHED.TENSOR": ("тензорный канал, тактов на команду", True),
    "CAP_PER_SCHED.ALU": ("целочисленный канал", True),
    "CAP_PER_SCHED.FPU": ("fp32-канал", True),
    "CAP_PER_SCHED.SFU": ("трансцендентный канал", True),
    "CAP_PER_SCHED.ISSUE": ("порт выдачи", True),
    "CAP_PER_SCHED.BRANCH": ("блок ветвлений", True),
    "LATENCY.LDS": ("задержка чтения разделяемой памяти", True),
    "LATENCY.SHFL": ("задержка обмена между полосами", True),
    "LATENCY.FFMA": ("задержка fp32-умножения-сложения", True),
    "LATENCY.IADD3": ("задержка целочисленного сложения", True),
    "LATENCY.HMMA": ("задержка тензорной команды (у нас ОЦЕНКА СВЕРХУ)", True),
    "LATENCY.LDG": ("задержка чтения глобальной памяти", True),
    "MIO_CONFLICT": ("КОНФЛИКТНОСТЬ раскладки -- из SASS НЕ ВЫВОДИТСЯ", True),
}
CALIB = {
    "loaded": False,
    "rates": {},
    "dir": None,
    "records": 0,
    "errors": {},
    "warnings": {},
}
MIO_WF_PER_INST = {}  # ядро -> замеренная цена команды разделяемой памяти (вайвфронтов)


def load_calibration(path=None, verbose=False):
    """Читает data/calib/*.json и ПОДМЕНЯЕТ ставки модели замеренными. -> сколько ставок взято.

    Отсутствие калибровки -- НЕ ошибка: модель обязана работать на своих ставках и честно об этом
    сообщать. Ошибка -- молча считать модельную ставку замеренной.
    """
    global \
        MIO_BYTES_PER_CYCLE, \
        MIO_WAVEFRONT_BYTES, \
        SCHEDULERS, \
        REG_ISA_LIMIT, \
        REG_OVERHEAD
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import calib as C
    except Exception as ex:
        CALIB["errors"] = {"<загрузчик>": ["модуль calib не подключился: %s" % ex]}
        return 0
    recs, errs, warns = C.load_dir(path)
    rr, clash = C.rates(recs)
    CALIB.update(
        {
            "loaded": True,
            "rates": rr,
            "dir": path or C.CALIB_DIR,
            "records": len(recs),
            "errors": dict(errs),
            "warnings": dict(warns),
            "clash": clash,
        }
    )
    took = 0
    for sym, v in rr.items():
        val = v["value"]
        if sym == "MIO_BYTES_PER_CYCLE":
            MIO_BYTES_PER_CYCLE = val
        elif sym == "MIO_WAVEFRONT_BYTES":
            MIO_WAVEFRONT_BYTES = val
        elif sym == "SCHEDULERS":
            SCHEDULERS = int(val)
        elif sym == "REG_ISA_LIMIT":
            REG_ISA_LIMIT = int(val)
        elif sym == "REG_OVERHEAD":
            REG_OVERHEAD = int(val)
        elif sym.startswith("CAP_PER_SCHED."):
            CAP_PER_SCHED[sym.split(".", 1)[1]] = val
        elif sym.startswith("LATENCY."):
            LATENCY[sym.split(".", 1)[1]] = val
        elif sym.startswith("MIO_WF_PER_INST."):
            MIO_WF_PER_INST[sym.split(".", 1)[1]] = val
        elif sym.startswith("MIO_CONFLICT_SHARE."):
            pass  # доля конфликтов -- диагностика, в ставку не входит (см. отчёт невязки)
        else:
            continue
        took += 1
    if verbose:
        calibration_report()
    return took


def calibration_report(out=print):
    """ПЕЧАТАЕТ, какие ставки взяты из замера, а какие остались модельными. Обязательный вывод."""
    rr = CALIB.get("rates") or {}
    out(
        "КАЛИБРОВКА: %s"
        % (CALIB["dir"] if CALIB["loaded"] else "НЕ ЗАГРУЖЕНА -- все ставки МОДЕЛЬНЫЕ")
    )
    if CALIB["loaded"]:
        out(
            "  записей %d, ставок из замера %d, негодных записей %d"
            % (CALIB["records"], len(rr), len(CALIB["errors"]))
        )
    out(
        "  %-24s %-10s %-14s %s"
        % ("ставка", "значение", "откуда", "статус наблюдаемости")
    )
    for sym, (desc, measurable) in MODEL_RATES.items():
        v = rr.get(sym)
        cur = _rate_value(sym)
        if v:
            out(
                "  %-24s %-10s %-14s %s" % (sym, _fmt(cur), "ЗАМЕР", v["observability"])
            )
        else:
            out(
                "  %-24s %-10s %-14s %s"
                % (
                    sym,
                    _fmt(cur),
                    "модель",
                    "измеримо, но НЕ ЗАМЕРЕНО"
                    if measurable
                    else "неизмеримо по отдельности",
                )
            )
    for k in MIO_WF_PER_INST:
        out(
            "  %-24s %-10.3f %-14s без_возмущения"
            % ("MIO_WF_PER_INST." + k, MIO_WF_PER_INST[k], "ЗАМЕР")
        )
    for k, v in (CALIB.get("errors") or {}).items():
        out("  НЕГОДНАЯ ЗАПИСЬ %s: %s" % (k, "; ".join(v)))
    for c in CALIB.get("clash") or []:
        out("  КОНФЛИКТ СТАВОК: %s" % c)


def _fmt(v):
    if v is None:
        return "-"
    return ("%g" % v) if isinstance(v, (int, float)) else str(v)


def _rate_value(sym):
    if sym == "SCHEDULERS":
        return SCHEDULERS
    if sym == "MIO_BYTES_PER_CYCLE":
        return MIO_BYTES_PER_CYCLE
    if sym == "MIO_WAVEFRONT_BYTES":
        return MIO_WAVEFRONT_BYTES
    if sym == "REG_ISA_LIMIT":
        return REG_ISA_LIMIT
    if sym == "REG_OVERHEAD":
        return REG_OVERHEAD
    if sym.startswith("CAP_PER_SCHED."):
        return CAP_PER_SCHED.get(sym.split(".", 1)[1])
    if sym.startswith("LATENCY."):
        return LATENCY.get(sym.split(".", 1)[1])
    if sym == "MIO_CONFLICT":
        return 1.0 if MIO_SAFE else None
    return None


def mio_calibration_for(kernel_name):
    """-> (замеренная цена команды, имя записи) для ядра, если калибровка её знает."""
    for k, v in MIO_WF_PER_INST.items():
        if k and k in (kernel_name or ""):
            return v, k
    return None, None


LATENCY = {
    "HMMA": 8.0,
    "LOP3": 4.0,
    "IADD3": 4.0,
    "IMAD": 6.0,
    "SHF": 4.0,
    "LEA": 4.0,
    "MOV": 4.0,
    "SEL": 4.0,
    "PRMT": 4.0,
    "ISETP": 4.0,
    "PLOP3": 4.0,
    "POPC": 4.0,
    "FFMA": 4.0,
    "FADD": 4.0,
    "FMUL": 4.0,
    "FSETP": 4.0,
    "HFMA2": 4.0,
    "HADD2": 4.0,
    "F2F": 4.0,
    "I2F": 4.0,
    "F2I": 4.0,
    "MUFU": 12.0,
    "LDS": 26.0,
    "LDSM": 26.0,
    "STS": 4.0,
    "SHFL": 26.0,
    "LDG": 400.0,
    "STG": 4.0,
    "LDC": 20.0,
    "S2R": 20.0,
    "CS2R": 6.0,
    "BAR": 8.0,
    "BRA": 6.0,
    "EXIT": 0.0,
    "NOP": 1.0,
}
DEFAULT_LATENCY = 6.0


def footprint(ins, safe=None):
    """Отпечаток команды: {канал: тактов}.  Каждая команда всегда занимает ОДИН слот выдачи.

    safe=True  -- ДЕШЕВЕЙШЕЕ обслуживание (годится как нижняя ГРАНИЦА при любых адресах);
    safe=False -- НОМИНАЛЬНОЕ (ширина*32), точнее там, где адреса полос различны, но границей
                  не является: на многоадресном обращении замер лежит НИЖЕ него.
    """
    if safe is None:
        safe = MIO_SAFE
    op, base, cls = ins.op, ins.base, ins.cls
    fp = {CH_ISSUE: 1.0}
    if cls == IS.TENSOR:
        fp[CH_TENSOR] = 2.0
    elif base in ("LDS", "LDSM", "STS", "SHFL"):
        # MIO: цена в БАЙТАХ на весь SM.  Для ГРАНИЦЫ берётся ДЕШЕВЕЙШЕЕ обслуживание команды
        # (многоадресность даёт не более двух раз, ниже одного такта на обращение не бывает).
        # Ставку "ширина*32" даёт SAFE=False -- она ТОЧНЕЕ там, где адреса различны, но НЕ
        # ЯВЛЯЕТСЯ границей: см. MIO_MULTICAST.
        wb = float(IS.width_of(ins))
        # safe: конфликтность 1 (минимум по всем раскладкам адресов) -- ГРАНИЦА.
        # иначе: конфликтность = ширина/4 (все полосы различны, соседние слова) -- ставка,
        # точная для volta_fwd_block и заниженная там, где есть банковые конфликты.
        wf = mio_wavefronts(wb, 1.0 if safe else max(wb / 4.0, 1.0))
        fp[CH_MIO] = wf * MIO_WAVEFRONT_BYTES
    elif base in (
        "LDG",
        "STG",
        "LD",
        "ST",
        "RED",
        "ATOM",
        "ATOMG",
        "ATOMS",
        "LDL",
        "STL",
    ):
        fp[CH_LSU] = 1.0
    elif base == "MUFU":
        fp[CH_SFU] = 8.0
    elif cls == "плавающие":
        fp[CH_FPU] = 2.0
    elif cls in ("цел.арифм", "цел.лог", "перестановки", "пересылки", "конст/спец"):
        fp[CH_ALU] = 2.0
    elif cls in ("барьеры", "ветвл./пред"):
        # Ветвление и барьеры исполняет ОТДЕЛЬНЫЙ блок, не 16-полосный ALU.  Считать их по
        # ставке ALU = 2 такта -- ошибка, которая ЗАВЫШАЕТ нижнюю границу: на теле c1x16
        # модель давала 76 тактов против ЗАМЕРЕННЫХ 72.1, то есть нарушала собственную
        # границу.  С отдельным каналом BRANCH (1 такт) выходит ровно 72.
        fp[CH_BRANCH] = 1.0
    else:
        fp[CH_ALU] = 2.0
    return fp


def latency(ins):
    return LATENCY.get(ins.base, DEFAULT_LATENCY)


# --------------------------------------------------------------------------------------------
# 1b. ПРЕДИКАТЫ КАК ЗАВИСИМОСТИ.  Вендоренный парсер их не ведёт, а цепь управления цикла идёт
# ИМЕННО через предикат (IADD3 -> ISETP -> @P0 BRA), и без неё RecMII пустого цикла = 0, тогда
# как замер даёт 26 тактов.  Поэтому дополняем dst/srcs предикатными именами.
# --------------------------------------------------------------------------------------------
_PRED_DST = re.compile(r"^(?:ISETP|PSETP|PLOP3|FSETP|HSETP2|VOTE|VOTEU)\S*\s+(P\d+)")


def dst_of(ins):
    m = _PRED_DST.match(ins.text)
    if m:
        return m.group(1)
    return ins.dst


def srcs_of(ins):
    s = set(ins.srcs)
    if ins.pred:
        s.add(ins.pred)
    for p in re.findall(
        r"\bP[0-6]\b", ins.text.split(" ", 1)[1] if " " in ins.text else ""
    ):
        if not _PRED_DST.match(ins.text) or p != _PRED_DST.match(ins.text).group(1):
            s.add(p)
    return s


# --------------------------------------------------------------------------------------------
# 2. ГРАФ ЗАВИСИМОСТЕЙ мейнлупа и RecMII
# --------------------------------------------------------------------------------------------
def dep_graph(ins_list):
    """Дуги (i -> j, задержка, оборотов).  Внутри итерации оборот 0, через обратную дугу -- 1.

    Считаем по регистрам: последняя запись перед чтением -- дуга 0 оборотов; запись ПОСЛЕ
    чтения в порядке тела -- дуга 1 оборота (значение доедет только на следующей итерации).
    Этого достаточно для рекуррентной границы; anti/output-зависимости не нужны, потому что
    период ограничивают только истинные цепи.
    """
    n = len(ins_list)
    lastdef = {}
    edges = []
    for j, it in enumerate(ins_list):
        for r in srcs_of(it):
            if r in lastdef:
                i = lastdef[r]
                edges.append((i, j, latency(ins_list[i]), 0))
        d = dst_of(it)
        if d:
            lastdef[d] = j
    # через итерацию: запись на позиции i "доезжает" до чтения на позиции j <= i
    firstuse = {}
    for j, it in enumerate(ins_list):
        for r in srcs_of(it):
            firstuse.setdefault(r, j)
    for r, i in lastdef.items():
        if r in firstuse and firstuse[r] <= i:
            edges.append((i, firstuse[r], latency(ins_list[i]), 1))
    return edges


REC_LIMIT = 260  # выше этого размера тела Беллман-Форд в бинпоиске неприемлемо долог


def rec_mii(n, edges, limit=None):
    """Обёртка с ограничением размера: у боевых ядер тело в тысячи команд, и точный RecMII
    считать нечем -- он честно помечается как НЕ ПОСЧИТАННЫЙ (0), а не подменяется оценкой."""
    if (limit or REC_LIMIT) and n > (limit or REC_LIMIT):
        return 0.0
    return _rec_mii(n, edges)


def _rec_mii(n, edges):
    """max по циклам (сумма задержек)/(сумма оборотов).  Двоичный поиск + Беллман-Форд:
    T допустимо <=> в графе с весами (lat - T*turns) нет положительного цикла."""
    if not edges:
        return 0.0
    if not any(t > 0 for _, _, _, t in edges):
        return 0.0

    def feasible(T):
        d = [0.0] * n
        for _ in range(n + 1):
            ch = False
            for i, j, l, t in edges:
                w = l - T * t
                if d[i] + w > d[j] + 1e-9:
                    d[j] = d[i] + w
                    ch = True
            if not ch:
                return True
        return False

    lo, hi = 0.0, 1.0
    while not feasible(hi):
        hi *= 2.0
        if hi > 1e6:
            return hi
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return hi


# --------------------------------------------------------------------------------------------
# 3b. ВЕРХНЯЯ ГРАНИЦА: симулятор ВЫДАЧИ ПО ПОРЯДКУ (то, что делает железо, а не оптимум).
#
# Volta выдаёт команды варпа СТРОГО ПО ПОРЯДКУ.  Поэтому варп, у которого готова команда k+1,
# но не готова k, простаивает -- и период определяется не только ёмкостью каналов, но и тем,
# в каком МЕСТЕ цепи стоит команда.  Симулятор гоняет W варпов на планировщике по одному телу
# со сдвигом фаз и возвращает установившийся период.  Это ВЕРХНЯЯ оценка: перестановка команд
# (то, чем занят компилятор) может её только уменьшить, вплоть до max(ResMII, RecMII).
# --------------------------------------------------------------------------------------------
def simulate_inorder(ins_list, warps, iters=40, warmup=12):
    """Такт за тактом: планировщик, wps варпов, выдача СТРОГО ПО ПОРЯДКУ, один слот за такт.

    Каналы моделируются занятостью (команда держит канал c тактов).  MIO берётся долей
    планировщика (нагрузка симметрична по четырём планировщикам SM).  Возвращает
    установившийся период варпа 0.
    """
    wps = max(1, int(round(warps / float(SCHEDULERS))))
    n = len(ins_list)
    if n == 0:
        return float("nan")
    fps = [
        footprint(it, safe=False) for it in ins_list
    ]  # симулятор -- НОМИНАЛЬНАЯ ставка
    lats = [latency(it) for it in ins_list]
    deps = [[] for _ in range(n)]
    lastdef = {}
    for j, it in enumerate(ins_list):
        for r in srcs_of(it):
            if r in lastdef:
                deps[j].append((lastdef[r], 0))
        d = dst_of(it)
        if d:
            lastdef[d] = j
    firstuse = {}
    for j, it in enumerate(ins_list):
        for r in srcs_of(it):
            firstuse.setdefault(r, j)
    for r, i in lastdef.items():
        if r in firstuse and firstuse[r] <= i:
            deps[firstuse[r]].append((i, 1))

    mio_cap = MIO_BYTES_PER_CYCLE / SCHEDULERS
    chan_free = collections.defaultdict(float)
    ready = [dict() for _ in range(wps)]
    pc = [0] * wps
    itn = [0] * wps
    marks = []
    t = 0
    limit = 40 * n * (iters + 4) * 8 + 20000
    while itn[0] < iters and t < limit:
        for off in range(wps):
            w = (t + off) % wps
            if itn[w] >= iters:
                continue
            k = pc[w]
            rdy = 0.0
            for i, turn in deps[k]:
                key = (i, itn[w] - turn)
                if key[1] >= 0:
                    rdy = max(rdy, ready[w].get(key, 0.0))
            if rdy > t:
                continue
            fp = fps[k]
            blocked = False
            for ch in fp:
                if ch != CH_ISSUE and chan_free[ch] > t:
                    blocked = True
                    break
            if blocked:
                continue
            for ch, c in fp.items():
                if ch == CH_ISSUE:
                    continue
                chan_free[ch] = t + (c / mio_cap if ch == CH_MIO else c)
            ready[w][(k, itn[w])] = t + lats[k]
            if len(ready[w]) > 8 * n:
                for key in [q for q in ready[w] if q[1] < itn[w] - 2]:
                    del ready[w][key]
            pc[w] += 1
            if pc[w] >= n:
                pc[w] = 0
                itn[w] += 1
                if w == 0:
                    marks.append(t)
            break
        t += 1
    if len(marks) <= warmup + 2:
        return float("nan")
    return (marks[-1] - marks[warmup]) / float(len(marks) - 1 - warmup)


# --------------------------------------------------------------------------------------------
# 3. ГРАНИЦЫ
# --------------------------------------------------------------------------------------------
def mio_nominal_mean(ins_list):
    """Средняя МОДЕЛЬНАЯ цена команды разделяемой памяти (ставка 'все полосы различны').

    Нужна, чтобы перевести ЗАМЕРЕННУЮ среднюю цену (вайвфронтов на команду, ncu) в множитель к
    отпечатку: замер даёт агрегат, модель раскладывает его по командам пропорционально ширине.
    Это ЧЕСТНО ровно в той мере, в какой конфликты распределены по командам как ширина, -- и это
    допущение названо здесь, а не спрятано."""
    tot, n = 0.0, 0
    for it in ins_list:
        if it.base in ("LDS", "LDSM", "STS", "SHFL"):
            wb = float(IS.width_of(it))
            tot += mio_wavefronts(wb, max(wb / 4.0, 1.0))
            n += 1
    return (tot / n) if n else None


def analyse(ins_list, warps, cal_wf=None):
    """-> dict с ResMII по каналам, RecMII и предсказанием периода.

    cal_wf -- ЗАМЕРЕННАЯ средняя цена команды разделяемой памяти (вайвфронтов). Если задана,
    считается ТРЕТИЙ сценарий res_cal: тот же отпечаток, но с ценой MIO из ЗАМЕРА, а не из ставки.
    """
    wps = warps / float(SCHEDULERS)  # варпов на планировщик

    def _res(safe, mio_scale=1.0):
        load = collections.Counter()
        for it in ins_list:
            for ch, c in footprint(it, safe=safe).items():
                load[ch] += c * (mio_scale if ch == CH_MIO else 1.0)
        r = {}
        for ch, v in load.items():
            if ch == CH_MIO:
                r[ch] = v * warps / MIO_BYTES_PER_CYCLE  # ресурс ВСЕГО SM
            else:
                r[ch] = v * wps / CAP_PER_SCHED.get(ch, 1.0)  # ресурс планировщика
        return r, dict(load)

    res, load = _res(True)  # ГРАНИЦА (годится при любых адресах)
    res_nom, _ = _res(False)  # НОМИНАЛ (адреса полос различны)
    resmii = max(res.values()) if res else 0.0
    resmii_nom = max(res_nom.values()) if res_nom else 0.0
    binding = max(res, key=res.get) if res else None
    edges = dep_graph(ins_list)
    recmii = rec_mii(len(ins_list), edges)
    sim = (
        simulate_inorder(ins_list, warps)
        if len(ins_list) <= REC_LIMIT
        else float("nan")
    )
    # ТРЕТИЙ СЦЕНАРИЙ -- ставка MIO ИЗ ЗАМЕРА. Разница с номиналом и есть невязка модели на этом
    # ядре: то, чего она не знает про раскладку.
    res_cal, resmii_cal, binding_cal, mio_scale = None, None, None, None
    nom_mean = mio_nominal_mean(ins_list)
    if cal_wf and nom_mean:
        mio_scale = cal_wf / nom_mean
        res_cal, _ = _res(False, mio_scale)
        resmii_cal = max(res_cal.values()) if res_cal else 0.0
        binding_cal = max(res_cal, key=res_cal.get) if res_cal else None
    return {
        "res_cal": res_cal,
        "ResMII_cal": resmii_cal,
        "binding_cal": binding_cal,
        "mio_scale": mio_scale,
        "mio_nominal_mean": nom_mean,
        "mio_measured_mean": cal_wf,
        "T_cal": (max(resmii_cal, recmii) if resmii_cal is not None else None),
        "ResMII_nom": resmii_nom,
        "T_nom": max(resmii_nom, recmii),
        "res_nom": res_nom,
        "T_inorder": sim,
        "n": len(ins_list),
        "warps": warps,
        "load": dict(load),
        "res": res,
        "ResMII": resmii,
        "binding": binding,
        "RecMII": recmii,
        "T_pred": max(resmii, recmii),
        "T_high": sim,
    }


# --------------------------------------------------------------------------------------------
# 3d. БЕЗУСЛОВНОЕ ЯДРО ЦИКЛА -- то, без чего ResMII тоже НЕ ЯВЛЯЕТСЯ ГРАНИЦЕЙ.
#
# Сумма отпечатков по ВСЕМ блокам естественного цикла -- это ОЦЕНКА СВЕРХУ: блоки под ветвлением
# исполняются не каждую итерацию, а модель начисляет их каждую.  Нижняя граница из завышенной
# нагрузки -- не граница.  Правильный и ДЕШЁВЫЙ ответ: динамические веса для границы НЕ НУЖНЫ,
# нужен БЕЗУСЛОВНЫЙ ОСТОВ -- блоки, исполняемые НЕ МЕНЬШЕ одного раза за оборот:
#     B доминирует источник обратной дуги  И  B доминируется шапкой.
# Вложенный цикл, если он безусловен, попадает в остов и считается ОДИН раз -- это ЗАНИЖЕНИЕ,
# а занижение границу сохраняет.  Всё, что вне остова, вносит >= 0.  Поэтому
#     ResMII(остов) <= ResMII(истинный динамический) <= T,
# и первая величина -- честная граница, вторая -- предмет пункта 7 (точность, а не законность).
# --------------------------------------------------------------------------------------------
def core_blocks(cfg, header, body, latch):
    """Блоки цикла, исполняемые не меньше раза за оборот."""
    out = set()
    for b in body:
        dom_b = cfg.dom.get(b, set())
        dom_latch = cfg.dom.get(latch, set())
        if header in dom_b and b in dom_latch:
            out.add(b)
    return out or set(body)


def mainloop_ins(cfg, core=False, blocks=None):
    ml = IS.pick_mainloop(cfg, blocks) if blocks is not None else IS.pick_mainloop(cfg)
    if ml is None:
        return None, None
    hdr, body, be = ml
    sel = core_blocks(cfg, hdr, body, be) if core else body
    ins = [it for a in sorted(sel) for it in cfg.blocks[a]]
    return ins, (hdr, be)


def load_kernels(path):
    sass = IS.subprocess.run(
        [IS.find_cuobjdump(), "-sass", path], capture_output=True, text=True
    ).stdout
    return IS.parse_sass(sass)


# --------------------------------------------------------------------------------------------
# 3c. РОЛИ ВАРПОВ.  У боевого ядра варпы разведены на ВЕЗУЩИХ и СЧИТАЮЩИХ: у них РАЗНЫЕ тела
# цикла, но ОДИН период (их связывает рандеву на плитку).  Каналы SM (MIO) складывают нагрузку
# обеих ролей; каналы планировщика -- тоже, потому что варпы обеих ролей висят на одних и тех
# же четырёх планировщиках.  Считать роль отдельно и брать максимум -- НЕВЕРНО, это занижает.
# --------------------------------------------------------------------------------------------
def analyse_roles(cfg, roles, warps_total=None, core=False):
    parts = []
    for rn in roles.names:
        ml = IS.pick_mainloop(cfg, roles.blocks[rn])
        if ml is None:
            continue
        sel = core_blocks(cfg, ml[0], ml[1], ml[2]) if core else ml[1]
        ins = [it for a in sorted(sel) for it in cfg.blocks[a]]
        parts.append((rn, ins, roles.warps[rn] or 0))
    if not parts:
        return None
    load = collections.Counter()
    per_role = {}
    for rn, ins, w in parts:
        l = collections.Counter()
        for it in ins:
            for ch, c in footprint(it).items():
                l[ch] += c
        per_role[rn] = {"n": len(ins), "warps": w, "load": dict(l)}
        for ch, v in l.items():
            load[ch] += v * w  # нагрузка НА ВЕСЬ SM (варпов роли штук)
    res = {}
    for ch, v in load.items():
        if ch == CH_MIO:
            res[ch] = v / MIO_BYTES_PER_CYCLE  # ресурс SM целиком
        else:
            res[ch] = v / float(SCHEDULERS)  # четыре планировщика поровну
    resmii = max(res.values()) if res else 0.0
    binding = max(res, key=res.get) if res else None
    rec = 0.0
    for rn, ins, w in parts:
        rec = max(rec, rec_mii(len(ins), dep_graph(ins)))
    return {
        "res": res,
        "ResMII": resmii,
        "binding": binding,
        "RecMII": rec,
        "T_pred": max(resmii, rec),
        "roles": per_role,
    }


# --------------------------------------------------------------------------------------------
# 4. САМОПРОВЕРКА: модель против замеров стенда
# --------------------------------------------------------------------------------------------
def selftest(binpath, measured_json, verbose=True):
    kern = load_kernels(binpath)
    meas = json.load(open(measured_json))
    byname = {}
    for k, ins in kern.items():
        m = re.search(r"Li(\d+)E" * 9, k)
        if m:
            byname[tuple(int(x) for x in m.groups())] = k
    rows = []
    for key, val in meas.items():
        name, w = key.rsplit("|", 1)
        w = int(w)
        params = meas_params.get(name)
        if params is None or params not in byname:
            continue
        cfg = IS.CFG(kern[byname[params]])
        ins, _ = mainloop_ins(cfg)
        if not ins:
            continue
        a = analyse(ins, w)
        rows.append(
            (
                name,
                w,
                val,
                a["T_pred"],
                a["binding"],
                a["ResMII"],
                a["RecMII"],
                a["T_high"],
                a["T_nom"],
            )
        )
    if verbose:
        print(
            "%-13s %4s %9s %9s %8s %9s %8s %9s %8s %-7s"
            % (
                "вариант",
                "варп",
                "замер",
                "T_low",
                "ошибка",
                "ResMII",
                "RecMII",
                "T_high",
                "в кор.",
                "связ.",
            )
        )
        for name, w, val, pred, b, rs, rc, hi, nom in sorted(
            rows, key=lambda r: (r[0], r[1])
        ):
            err = (pred - val) / val * 100.0
            inside = "да" if (pred - 1e-6 <= val <= hi * 1.02 + 1e-6) else "НЕТ"
            print(
                "%-13s %4d %9.2f %9.2f %7.1f%% %9.1f %8.1f %9.2f %8s %-7s"
                % (name, w, val, pred, err, rs, rc, hi, inside, b)
            )
    errs = [abs(p - v) / v for _, _, v, p, _, _, _, _, _ in rows]
    return rows, errs


meas_params = {}  # заполняется из таблицы вариантов probe.cu


def parse_variants(cu_path):
    """Читает таблицу X(...) из probe.cu -> {имя: (NH,NI,NC,NP,NS,SW,NF)}"""
    txt = open(cu_path).read()
    out = {}
    for m in re.finditer(r'\bX[DS]?\("([^"]+)",\s*([^)]+)\)', txt):
        vals = [v.strip() for v in m.group(2).split(",")]
        try:
            iv = [int(v) for v in vals]
        except ValueError:
            continue
        if len(iv) == 7:
            iv.append(1)  # X(...)  -- DUP по умолчанию 1
        if len(iv) == 8:
            iv.append(0)  # XD(...) -- STRB=0 (шаблонный умолчательный шаг)
        if len(iv) == 9:
            out[m.group(1)] = tuple(iv)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj", nargs="?")
    ap.add_argument("--kernel")
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument(
        "--probe",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "build", "probe"
        ),
    )
    ap.add_argument(
        "--measured",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "probe_measured.json",
        ),
    )
    ap.add_argument(
        "--variants",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe.cu"),
    )
    # ФЛАГ, А НЕ ЗНАЧЕНИЕ: '--calib' с необязательным аргументом СЪЕДАЛ позиционный путь к бинарю
    # (argparse отдал ему 'build/probe'), и модель молча профилировала не тот объект.
    ap.add_argument(
        "--calib",
        action="store_true",
        help="взять ставки из калибровки (data/calib), а не из зашитых констант",
    )
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--no-calib-report", action="store_true")
    args = ap.parse_args()

    if args.calib or args.calib_dir:
        took = load_calibration(args.calib_dir)
        if not args.no_calib_report:
            calibration_report()
            print("ставок подменено замером: %d" % took)
            print()

    if args.selftest:
        global meas_params
        meas_params = parse_variants(args.variants)
        rows, errs = selftest(args.probe, args.measured)
        import statistics

        inside = sum(
            1
            for _, _, v, p, _, _, _, hi, _ in rows
            if p - 1e-6 <= v <= hi * 1.02 + 1e-6
        )
        hi_err = [abs(hi - v) / v for _, _, v, _, _, _, _, hi, _ in rows if hi == hi]
        viol_safe = [
            (n, w, v, p) for n, w, v, p, _, _, _, _, _ in rows if p > v * 1.001
        ]
        viol_nom = [
            (n, w, v, nm) for n, w, v, _, _, _, _, _, nm in rows if nm > v * 1.001
        ]
        print("\nточек: %d" % len(errs))
        print(
            "НИЖНЯЯ граница T_low = max(ResMII,RecMII): медиана |ошибки| %.1f%%, среднее %.1f%%, макс %.1f%%"
            % (
                100 * statistics.median(errs),
                100 * sum(errs) / len(errs),
                100 * max(errs),
            )
        )
        print(
            "ВЕРХНЯЯ граница T_high (выдача по порядку): медиана |ошибки| %.1f%%"
            % (100 * statistics.median(hi_err) if hi_err else float("nan"))
        )
        print(
            "ЗАМЕР ВНУТРИ КОРИДОРА [T_low, T_high]: %d из %d (%.0f%%)"
            % (inside, len(rows), 100.0 * inside / len(rows))
        )
        print(
            "НАРУШЕНИЙ ГРАНИЦЫ (безопасная ставка MIO): %d из %d"
            % (len(viol_safe), len(rows))
        )
        print(
            "НАРУШЕНИЙ ГРАНИЦЫ (номинальная ставка ширина*32): %d из %d"
            % (len(viol_nom), len(rows))
        )
        for n, w, v, p in sorted(viol_nom, key=lambda r: -(r[3] / r[2]))[:8]:
            print(
                "    %-13s варпов %2d  замер %8.2f  номинальная граница %8.2f  превышение %+.0f%%"
                % (n, w, v, p, 100.0 * (p / v - 1.0))
            )
        return

    kern = load_kernels(args.obj)
    names = [k for k in kern if not args.kernel or re.search(args.kernel, k)]
    out = []
    for k in names:
        cfg = IS.CFG(kern[k])
        ins, hdrbe = mainloop_ins(cfg)
        if not ins:
            print("%s: мейнлуп не найден" % k)
            continue
        cal_wf, cal_src = mio_calibration_for(k)
        a = analyse(ins, args.warps, cal_wf=cal_wf)
        a["kernel"] = k
        a["calib_source"] = cal_src
        print("=" * 96)
        print("ЯДРО %s" % k)
        print("  мейнлуп: %d команд, шапка %#06x" % (a["n"], hdrbe[0]))
        print("  ЗАГРУЗКА КАНАЛОВ (на итерацию, при %d варпах):" % args.warps)
        for ch, v in sorted(a["res"].items(), key=lambda kv: -kv[1]):
            mark = " <== СВЯЗЫВАЕТ" if ch == a["binding"] else ""
            print("      %-7s %10.1f тактов%s" % (ch, v, mark))
        print(
            "  ResMII = %.1f   RecMII = %.1f   ->  T >= %.1f тактов"
            % (a["ResMII"], a["RecMII"], a["T_pred"])
        )
        if a["ResMII_cal"] is not None:
            print(
                "  СТАВКА MIO ИЗ ЗАМЕРА (%s: %.3f вайвфронта/команду против модельных %.3f, "
                "x%.3f):"
                % (
                    cal_src,
                    a["mio_measured_mean"],
                    a["mio_nominal_mean"],
                    a["mio_scale"],
                )
            )
            print(
                "      ResMII = %.1f (было %.1f по номиналу), связывает %s  ->  T >= %.1f"
                % (a["ResMII_cal"], a["ResMII_nom"], a["binding_cal"], a["T_cal"])
            )
            if a["binding_cal"] != a["binding"]:
                print(
                    "      ВНИМАНИЕ: замер МЕНЯЕТ связывающий канал (%s -> %s)"
                    % (a["binding"], a["binding_cal"])
                )
        elif CALIB["loaded"]:
            print(
                "  СТАВКА MIO: замера для этого ядра в калибровке НЕТ -- ставка МОДЕЛЬНАЯ"
            )
        out.append(a)
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
