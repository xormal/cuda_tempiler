#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""Порождает `dispatch.inc` из `select.py`.  ОДИН источник таблицы, два потребителя.

Диспетчер существует в двух местах по необходимости: python-предикат зовётся ОДИН раз при
загрузке (там же, где решается, брать ли наше ядро вообще), а таблица инстанциаций нужна
компилятору.  Чтобы эти двое не разъехались, второе ПОРОЖДАЕТСЯ из первого.

    python3 gen_dispatch.py > dispatch.inc
"""

from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _select():
    spec = importlib.util.spec_from_file_location(
        "tempo_select", os.path.join(HERE, "select.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


HEAD = """// SPDX-License-Identifier: LicenseRef-TRL-1.0
// Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
// Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
//
// ПОРОЖДЕНО из select.py -- РУКАМИ НЕ ПРАВИТЬ. Правится select.py, потом перегенерация:
//   python3 kernels/shipped/gemm_fp16/sm_70/gen_dispatch.py > dispatch.inc
//
// ЛЕСТНИЦА ПО M -- НЕ УДОБСТВО, А ЗАМЕР: лучшая ЕДИНАЯ гиперформа на все 35 боевых точек
// даёт геосреднее 0.746 к сопернику, выбор по полосе M -- 0.885. Разница 0.139 больше
// всего, что удалось выиграть ВНУТРИ одной гиперформы.
//
// TEMPO_FALLBACK: любая форма, не покрытая строками ниже, обязана уйти на штатный путь
// вызывающего. Возврат cudaErrorInvalidConfiguration -- ЗАКОННЫЙ ответ, а не сбой.
"""


def main():
    m = _select()
    C = m.COMMON
    out = [HEAD]

    def line(cond, tag, BM, BN, BK, WM, WN, MINB, FPREF):
        return (
            "  if (%s)\n    return launch_one<%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d>(p, s);  // %s"
            % (
                cond,
                BM,
                BN,
                BK,
                WM,
                WN,
                C["STAGES"],
                C["GSTAGE"],
                FPREF,
                C["GROUP"],
                C["EPI"],
                C["SWZ"],
                "true" if C["PRED"] else "false",
                MINB,
                tag,
            )
        )

    for m_max, tag, BM, BN, BK, WM, WN, MINB, FPREF in m.LADDER:
        out.append(
            line(
                "p.M <= %d && (p.N %% %d) == 0 && (p.K %% %d) == 0" % (m_max, BN, BK),
                tag,
                BM,
                BN,
                BK,
                WM,
                WN,
                MINB,
                FPREF,
            )
        )
    tag, BM, BN, BK, WM, WN, MINB, FPREF = m.WIDE
    out.append(
        line(
            "(p.N %% %d) == 0 && (p.K %% %d) == 0" % (BN, BK),
            tag + "  (M > 256, N кратно 256)",
            BM,
            BN,
            BK,
            WM,
            WN,
            MINB,
            FPREF,
        )
    )
    tag, BM, BN, BK, WM, WN, MINB, FPREF = m.NARROW
    out.append(
        line(
            "(p.N %% %d) == 0 && (p.K %% %d) == 0" % (BN, BK),
            tag + "  (M > 256, узкое N)",
            BM,
            BN,
            BK,
            WM,
            WN,
            MINB,
            FPREF,
        )
    )
    out.append("")
    out.append("  return cudaErrorInvalidConfiguration;  // TEMPO_FALLBACK")
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
