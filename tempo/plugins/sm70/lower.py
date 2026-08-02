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

from ..base import Atom, AtomKind, Hyperform, Launch, OpSpec, PluginCapabilityError, Rendered
from . import space
from .gemm_bound import Hyperform as GemmHyperform
from .machine import Sm70Machine

_HERE = os.path.dirname(os.path.abspath(__file__))
_SKEL = os.path.join(_HERE, "skeletons")
_M = Sm70Machine()

_OPS = {
    "gemm": {
        ("fp16", "fp16"): "gemm_hmma884",
        ("int8", "fp16"): "gemm_w8a16",
    }
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

    def _skeleton_dir(self, op: OpSpec) -> str:
        table = _OPS.get(op.kind)
        if table is None:
            raise PluginCapabilityError(
                "скелета под операцию %r у sm_70 нет; есть %s" % (op.kind, ", ".join(_OPS))
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
        return space.variants(op)

    # -- ОТСЕЧЕНИЕ БЕЗ СБОРКИ ----------------------------------------------------------------
    def estimate_atoms(self, op: OpSpec, h: Hyperform) -> list:
        """Статический набор атомов ОДНОГО шага мейнлупа.  Ни одной сборки.

        Считаем на ВАРП за один шаг по K (kстеп = 4 у m8n8k4):
          * тензорные квадропары: MB*NB*(BK/4) штук;
          * чтения фрагментов из разделяемой: (MB+NB)*(BK/4) команд LDS.128;
          * запись плитки в разделяемую: (BM+BN)*BK/8 порций на блок -> на варп делим.
        """
        g = space.to_gemm(h)
        ksteps = g.BK // 4
        warps = g.WM * g.WN
        out = []
        uid = 0

        def add(kind, op_id, fp, lat, region="body", deps=()):
            nonlocal uid
            out.append(
                Atom(uid=uid, kind=kind, op_id=op_id, footprint=fp, latency=lat,
                     region=region, deps=deps)
            )
            uid += 1

        qp = g.MB * g.NB * ksteps
        add(AtomKind.COMPUTE, "HMMA.884.F32.F32 x%d" % qp,
            {"TENSOR": qp * _M.rate("CAP.TENSOR").value, "ISSUE": qp * 1.0},
            _M.rate("LATENCY.HMMA").value)

        lds = (g.MB + g.NB) * ksteps
        wf = 2.0  # LDS.128: пол «ширина/8 Б» = 2 вайвфронта ДАЖЕ при полной рассылке
        add(AtomKind.MEM_READ, "LDS.128 x%d" % lds,
            {"MIO": lds * wf, "ISSUE": lds * 1.0}, _M.rate("LATENCY.LDS").value)

        sts = ((g.BM + g.BN) * g.BK // 8) // max(1, warps)
        add(AtomKind.XFER_ISSUE, "LDG.128 x%d" % sts,
            {"LSU": sts * 1.0, "ISSUE": sts * 1.0}, _M.rate("LATENCY.LDG").value,
            region="prologue")
        add(AtomKind.MEM_WRITE, "STS.128 x%d" % sts,
            {"MIO": sts * 2.0, "ISSUE": sts * 1.0}, _M.rate("LATENCY.STS").value,
            region="prologue")
        add(AtomKind.BARRIER, "BAR.SYNC", {"ISSUE": 1.0}, 0.0)
        return out

    def resources_of(self, op: OpSpec, h: Hyperform):
        """(regs, max_live, smem) БЕЗ СБОРКИ.  Оценка регистров -- по накопителям + подаче."""
        g = space.to_gemm(h)
        regs = g.regs_estimate()
        return int(regs), int(regs - 7), int(g.smem)

    def launch_of(self, op: OpSpec, h: Hyperform) -> Launch:
        g = space.to_gemm(h)
        M, N = op.shapes["M"], op.shapes["N"]
        nm = -(-M // g.BM)
        nn = N // g.BN
        return Launch(
            grid_ctas=nm * nn, threads=g.threads, smem_bytes=g.smem, entry="tempo_gemm_%s" % g.tag()
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
            % (g.tag(), op.kind, op.shapes.get("M"), op.shapes.get("N"), op.shapes.get("K"))
        )
        inst = (
            "\n// ---- ИНСТАНЦИАЦИЯ (единственное, что подставляет эмиттер) ----\n"
            "#define TEMPO_BM %d\n#define TEMPO_BN %d\n#define TEMPO_BK %d\n"
            "#define TEMPO_WM %d\n#define TEMPO_WN %d\n#define TEMPO_STAGES %d\n"
            "#define TEMPO_GSTAGE %d\n#define TEMPO_FPREF %d\n#define TEMPO_GROUP %d\n"
            "#define TEMPO_EPI %d\n#define TEMPO_SWZ %d\n#define TEMPO_PRED %d\n"
            "#define TEMPO_MINB %d\n"
            % (g.BM, g.BN, g.BK, g.WM, g.WN, g.STAGES, g.GSTAGE, g.FPREF, g.GROUP,
               g.EPI, g.SWZ, int(g.PRED), g.MINB)
        )
        return Rendered(
            source=header + inst + body,
            launch=self.launch_of(op, h),
            includes=(os.path.join(skel, "prims.cuh"),),
            notes="эмиттер только подставляет; вся логика -- в скелете и в space.py",
        )

    def entry_probe(self, h: Hyperform) -> str:
        """Чем ДОКАЗАТЬ, что зашли в НАШЕ ядро.

        Правило проекта, оплаченное днём работы: A/B, сдвинувший сквозную метрику на 0.03 %,
        означает, что туда НЕ ЗАХОДИЛИ (и что 42 из 48 слоёв Gemma-4 шли мимо наших ядер).
        """
        return "tempo_gemm_%s" % space.to_gemm(h).tag()
