#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""sm_70: СКЕЛЕТЫ -- реализация раздела 7 контракта.

ЧЕСТНАЯ РАМКА (её нельзя размывать в README): продукт ПОРОЖДАЕТ ядро по спецификации, а не
транслирует вход.  Наивный `.cu` служит (а) источником спецификации через блок `TEMPO-OP`,
(б) ЭТАЛОНОМ значений, (в) знаменателем метрики «выход против входа».  Парсера C++ нет и не
будет в v1.

`render` -- ПОДСТАНОВКА по шаблону, без логики железа: вся логика уже в `kernel.cuh` и в
`space.py`.  Это намеренно: чем тупее эмиттер, тем меньше мест, где может завестись
незамеченное расхождение между тем, что отсекатель посчитал, и тем, что собралось.
"""

from __future__ import annotations

import os
from typing import Iterator

from ..base import (
    Atom,
    AtomKind,
    Hyperform,
    Launch,
    OpSpec,
    PluginCapabilityError,
    Rendered,
)
from . import space
from .gemm_bound import Hyperform as GemmHyperform
from .machine import Sm70Machine
from .resources import Sm70Resources

_HERE = os.path.dirname(os.path.abspath(__file__))
_SKEL = os.path.join(_HERE, "skeletons")
_M = Sm70Machine()
_RES = Sm70Resources()

# Байт в одном вайвфронте.  НЕ новое число: это ровно то, чем подписана ставка CAP.MIO
# ("128.0 байт/такт = РОВНО ОДИН ВАЙВФРОНТ ЗА ТАКТ НА SM", channels.json).  Нужен здесь
# потому, что ЁМКОСТЬ канала объявлена в БАЙТАХ ЗА ТАКТ, а значит отпечаток обязан нести
# БАЙТЫ -- см. соглашение о единицах в шапке estimate_atoms.
_BYTES_PER_WAVEFRONT = 128.0

_OPS = {
    "gemm": {
        ("fp16", "fp16"): "gemm_hmma884",
        ("int8", "fp16"): "gemm_w8a16",
    }
}

# ИМЯ И СПИСОК ПАРАМЕТРОВ ЯДРА -- ДАННЫЕ СКЕЛЕТА, а не догадка эмиттера.  Список закрыт:
# скелет, которого здесь нет, ИНСТАНЦИРОВАТЬ НЕЧЕМ, и `render` обязан ОТКАЗАТЬ, а не выдать
# файл без единого ядра (ровно это и происходило: порождённый текст собирался в ПУСТОЙ
# объектный файл, а `entry_probe` называл символ, которого не порождалось никогда).
_ENTRY = {
    "gemm_hmma884": (
        "tempo::gen::k_gemm",
        lambda g: (
            g.BM,
            g.BN,
            g.BK,
            g.WM,
            g.WN,
            g.STAGES,
            g.GSTAGE,
            g.FPREF,
            g.GROUP,
            g.EPI,
            g.SWZ,
            "true" if g.PRED else "false",
            g.MINB,
        ),
        "const __half*, const __half*, __half*, int, int, int",
    ),
}

CAPABILITIES = frozenset(
    {
        "gemm_fp16_ntn",  # C = A * B^T, оба операнда k-мажорные
        "gemm_w8a16",  # байтовый ВЕС + fp16-активации (ради ТРАФИКА, не ФЛОПов)
        "swizzle_phase_injective",
        "l2_block_reorder",
        "split_m_ladder",  # диспетчер по полосе M
        "epilogue_none",  # эпилогов НЕТ: цена слияния silu(gate)*up названа ~6 % одного матмуля
    }
)


class Sm70Skeletons:
    def ops(self):
        return tuple(_OPS)

    def axes(self):
        return space.axes()

    def capabilities(self):
        return CAPABILITIES

    # ФОРМЫ ВЕРСИИ 1 -- ЗАКРЫТЫЙ СПИСОК.  Раскладка объявляется во ВХОДЕ (блок TEMPO-OP), и
    # объявить можно ЛЮБУЮ; скелет же считает РОВНО ОДНУ.  Пока этой проверки не было,
    # объявленная раскладка `b=n` принималась МОЛЧА и порождалось ядро ДРУГОЙ операции
    # (C = A*B^T вместо C = A*B) -- то есть ровно «плохой код вместо отказа», который
    # README и docs/NOT_YET.md обещают не выдавать.  Обещание, опровергаемое поведением,
    # дороже умолчания, поэтому проверка стоит НА ВХОДЕ в скелеты, а не в отчёте.
    _LAYOUT_V1 = ("k", "k", "n")
    _ACC_V1 = "fp32"

    def _require_form_v1(self, op: OpSpec) -> None:
        got = (op.layout_a, op.layout_b, op.layout_c)
        if got != self._LAYOUT_V1:
            raise PluginCapabilityError(
                "скелеты sm_70 считают ТОЛЬКО C = A*B^T с k-мажорными операндами "
                "(раскладки %s), объявлено %s. Это ОТКАЗ, а не приближение: при другой "
                "раскладке то же тело считает ДРУГУЮ операцию и проходит сверку значений "
                "на своей же неверной интерпретации."
                % ("/".join(self._LAYOUT_V1), "/".join(got))
            )
        if op.dtype_acc != self._ACC_V1:
            raise PluginCapabilityError(
                "скелеты sm_70 накапливают только в %s (объявлено %s): накопитель -- это "
                "форма инструкции HMMA.884.F32.F32, а не настройка."
                % (self._ACC_V1, op.dtype_acc)
            )

    def _skeleton_dir(self, op: OpSpec) -> str:
        self._require_form_v1(op)
        table = _OPS.get(op.kind)
        if table is None:
            raise PluginCapabilityError(
                "скелета под операцию %r у sm_70 нет; есть %s"
                % (op.kind, ", ".join(_OPS))
            )
        name = table.get((op.dtype_a, op.dtype_b))
        if name is None:
            raise PluginCapabilityError(
                "скелета под операнды (%s, %s) нет; есть %s. "
                "W8A8 (байтовые ОБА операнда) не построен: по замеру счётного выигрыша на sm_70 "
                "он не даёт (IMMA нет), а байтовая сторона A требует симметричной ветки "
                "загрузчика и ДВУСТОРОННЕЙ свёртки смещения."
                % (op.dtype_a, op.dtype_b, ", ".join("%s+%s" % k for k in table))
            )
        return os.path.join(_SKEL, name)

    def variants(self, op: OpSpec) -> Iterator[Hyperform]:
        # Отказ обязан прийти ДО перечисления: перечислить 984 гиперформы под операцию,
        # которую скелет не считает, значит выдать вердикт о том, чего не было.
        self._require_form_v1(op)
        return space.variants(op)

    # -- ОТСЕЧЕНИЕ БЕЗ СБОРКИ ----------------------------------------------------------------
    def passes(self, op: OpSpec, g: GemmHyperform) -> int:
        """Сколько раз ОДИН процессор проигрывает шаг мейнлупа НА ЭТОЙ ФОРМЕ.

        БЕЗ ЭТОГО ЧИСЛА ГРАНИЦА НЕ ЗАВИСИТ ОТ ФОРМЫ ВООБЩЕ, и ранжировать формы моделью
        НЕВОЗМОЖНО В ПРИНЦИПЕ (замерено: до этой правки модель ставила первым один и тот же
        вариант на всех шести боевых формах, а замеренный победитель стоял 136-280-м из 408).

            проходов = (K / BK)                     шагов мейнлупа
                     * ceil(блоков / (SM * блоков_на_SM))   ВОЛН

        Волна -- квант машины: хвост волны оплачивается целиком, и именно он делает
        «лишние» строки плитки дорогими при малом M.  Округление ВВЕРХ обязательно: 81 блок
        на 80 процессорах стоит двух волн, а не 1.0125.
        """
        M, N, K = int(op.shapes["M"]), int(op.shapes["N"]), int(op.shapes["K"])
        ksteps = max(1, K // g.BK)
        ctas = -(-M // g.BM) * (N // g.BN)
        occ = _RES.occupancy(
            min(int(_M.rate("GEOM.REG_ISA_LIMIT").value), g.regs_for_verdict()),
            g.smem,
            g.threads,
        )
        per_sm = max(1, occ.ctas_per_sm)
        sms = int(_M.rate("GEOM.SMS").value)
        waves = -(-ctas // (sms * per_sm))
        return int(ksteps * max(1, waves))

    def estimate_atoms(self, op: OpSpec, h: Hyperform) -> list:
        """Набор атомов ВСЕЙ операции, как её видит ОДИН процессор.  Ни одной сборки.

        Атом -- не команда, а ФАЗА: множество операций, занимающее канал на промежутке.
        Поэтому отпечаток одного шага мейнлупа домножается на число проходов (`passes`), и
        получается занятость канала за всю операцию.  Считаем на ВАРП за один шаг по K
        (kстеп = 4 у m8n8k4):
          * тензорные квадропары: MB*NB*(BK/4) штук;
          * чтения фрагментов из разделяемой: (MB+NB)*(BK/4) команд LDS.128;
          * запись плитки в разделяемую: (BM+BN)*BK/8 порций на блок -> на варп делим.

        СОГЛАШЕНИЕ О ЕДИНИЦАХ (на его смешении ломаются настоящие плагины -- предупреждение
        стоит в теле фальсификатора `nullarch`, и этот плагин на нём и сломался):
          * канал ПЛАНИРОВЩИКА (scope="sched") объявляет ёмкость в ТАКТАХ НА КОМАНДУ ->
            отпечаток несёт ЧИСЛО КОМАНД.  Прежняя редакция клала сюда «команды x такты» и
            считала TENSOR ВДВОЕ (на ранжирование это не влияло: связывал не он);
          * канал ПРОЦЕССОРА (scope="sm") объявляет ёмкость в ЕДИНИЦАХ ЗА ТАКТ -> отпечаток
            несёт ЧИСЛО ЕДИНИЦ.  У CAP.MIO единица -- БАЙТ, а прежняя редакция клала
            ВАЙВФРОНТЫ, то есть занижала канал в 128 раз.

        ЧЕГО ЗДЕСЬ НЕТ И ЭТО НАЗВАНО: канала внекристальной подачи.  При M <= 128 замерено,
        что связывает ЧТЕНИЕ ВЕСОВ (78-90 % полосы при 12-47 % счёта), и такого канала в
        таблице ставок НЕТ -- см. docs/NOT_YET.md.  Ранжирование в этой полосе моделью НЕ
        ОБЕСПЕЧЕНО, и правило приёмки (tests/test_prune_acceptance.py) печатает это числом.
        """
        g = space.to_gemm(h)
        ksteps = g.BK // 4
        warps = g.WM * g.WN
        reps = float(self.passes(op, g))
        out = []
        uid = 0

        def add(kind, op_id, fp, lat, region="body", deps=()):
            nonlocal uid
            out.append(
                Atom(
                    uid=uid,
                    kind=kind,
                    op_id=op_id,
                    footprint={k: v * reps for k, v in fp.items()},
                    latency=lat,
                    region=region,
                    deps=deps,
                )
            )
            uid += 1

        qp = g.MB * g.NB * ksteps
        add(
            AtomKind.COMPUTE,
            "HMMA.884.F32.F32 x%d" % qp,
            {"TENSOR": qp * 1.0, "ISSUE": qp * 1.0},
            _M.rate("LATENCY.HMMA").value,
        )

        lds = (g.MB + g.NB) * ksteps
        wf = 2.0  # LDS.128: пол «ширина/8 Б» = 2 вайвфронта ДАЖЕ при полной рассылке
        # Конфликтность -- ФУНКЦИЯ РАСКЛАДКИ, и она объявлена законом плитки (conflict_factor).
        # Без неё ось свиззла не двигала границу ВООБЩЕ, то есть модель её не различала.
        add(
            AtomKind.MEM_READ,
            "LDS.128 x%d" % lds,
            {
                "MIO": lds * wf * g.conflict_factor() * _BYTES_PER_WAVEFRONT,
                "ISSUE": lds * 1.0,
            },
            _M.rate("LATENCY.LDS").value,
        )

        sts = ((g.BM + g.BN) * g.BK // 8) // max(1, warps)
        add(
            AtomKind.XFER_ISSUE,
            "LDG.128 x%d" % sts,
            {"LSU": sts * 1.0, "ISSUE": sts * 1.0},
            _M.rate("LATENCY.LDG").value,
            region="prologue",
        )
        add(
            AtomKind.MEM_WRITE,
            "STS.128 x%d" % sts,
            {"MIO": sts * 2.0 * _BYTES_PER_WAVEFRONT, "ISSUE": sts * 1.0},
            _M.rate("LATENCY.STS").value,
            region="prologue",
        )
        add(AtomKind.BARRIER, "BAR.SYNC", {"ISSUE": 1.0}, 0.0)
        return out

    def resources_of(self, op: OpSpec, h: Hyperform):
        """(regs, max_live, smem) БЕЗ СБОРКИ.  Оценка регистров -- по накопителям + подаче.

        Отдаётся ОПТИМИСТИЧНЫЙ КРАЙ полосы (`regs_for_verdict`), а не точка: этим числом
        конвейер ВЫБРАСЫВАЕТ вариант без сборки, а выбрасывать по завышенной оценке значит
        терять победителя молча.  Обоснование и замер -- в gemm_bound.regs_for_verdict.
        """
        g = space.to_gemm(h)
        regs = g.regs_for_verdict()
        return int(regs), int(regs - 7), int(g.smem)

    def launch_of(self, op: OpSpec, h: Hyperform) -> Launch:
        g = space.to_gemm(h)
        M, N = op.shapes["M"], op.shapes["N"]
        nm = -(-M // g.BM)
        nn = N // g.BN
        return Launch(
            grid_ctas=nm * nn,
            threads=g.threads,
            smem_bytes=g.smem,
            entry=self.entry_probe(h),
        )

    # -- ЭМИССИЯ -----------------------------------------------------------------------------
    def render(self, op: OpSpec, h: Hyperform) -> Rendered:
        skel = self._skeleton_dir(op)
        g = space.to_gemm(h)
        with open(os.path.join(skel, "kernel.cuh"), encoding="utf-8") as f:
            body = f.read()
        header = (
            "// SPDX-License-Identifier: LicenseRef-TRL-1.0\n"
            "// Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>\n"
            "// Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.\n"
            "// ПОРОЖДЕНО tempo: плагин sm_70, гиперформа %s, операция %s %sx%sx%s.\n"
            % (
                g.tag(),
                op.kind,
                op.shapes.get("M"),
                op.shapes.get("N"),
                op.shapes.get("K"),
            )
        )
        # ЯВНАЯ ИНСТАНЦИАЦИЯ -- ЕДИНСТВЕННОЕ, ЧТО ПОДСТАВЛЯЕТ ЭМИТТЕР, И ОНА ОБЯЗАНА БЫТЬ.
        # Прежняя редакция подставляла блок `#define TEMPO_*`, которого в скелете НЕТ НИ
        # ОДНОГО ВХОЖДЕНИЯ: тело выходило ОДИНАКОВЫМ для всех 880 гиперформ, а порождённый
        # файл не содержал НИ ОДНОГО ядра -- собери его, и получишь ПУСТОЙ объектный файл.
        # Инстанциация решает это и одновременно делает `entry_probe` ПРАВДОЙ: имя, которое
        # он называет, теперь действительно есть в машинном коде.
        name = os.path.basename(skel)
        if name not in _ENTRY:
            raise PluginCapabilityError(
                "у скелета %r не объявлено имя ядра и список его параметров, поэтому "
                "инстанцировать нечего. Порождать файл БЕЗ ядра запрещено: он собирается в "
                "пустой объектный файл, а доказательство маршрута ищет несуществующий символ."
                % name
            )
        sym, params, args = _ENTRY[name]
        inst = (
            "\n// ---- ИНСТАНЦИАЦИЯ (единственное, что подставляет эмиттер) ----\n"
            "// Параметры пришли ИЗ ГИПЕРФОРМЫ, а не из умолчаний скелета. Без этой строки\n"
            "// шаблон не порождает НИ ОДНОГО ядра, и объектный файл выходит пустым.\n"
            "template __global__ void %s<%s>(%s);\n"
            % (sym, ", ".join(str(x) for x in params(g)), args)
        )
        return Rendered(
            source=header + body + inst,
            launch=self.launch_of(op, h),
            includes=(),
            notes="эмиттер только инстанцирует; вся логика -- в скелете и в space.py",
            filename="kernel_%s.cu" % g.tag(),
        )

    def entry_probe(self, h: Hyperform) -> str:
        """Чем ДОКАЗАТЬ, что зашли в НАШЕ ядро.

        Правило проекта, оплаченное днём работы: A/B, сдвинувший сквозную метрику на 0.03 %,
        означает, что туда НЕ ЗАХОДИЛИ (и что 42 из 48 слоёв Gemma-4 шли мимо наших ядер).

        Возвращается ИМЯ, КОТОРОЕ ДЕЙСТВИТЕЛЬНО ЕСТЬ В МАШИННОМ КОДЕ -- шаблонный идентификатор
        ровно в том виде, в каком его печатает профилировщик.  Прежняя редакция называла
        `tempo_gemm_<тег>`, а такого символа не порождалось НИ ОДНОГО: доказательство маршрута
        искало то, чего нет, и всегда «не находило» -- то есть не доказывало ничего.
        """
        sym, params, _ = _ENTRY["gemm_hmma884"]
        return "%s<%s>" % (sym, ", ".join(str(x) for x in params(space.to_gemm(h))))
