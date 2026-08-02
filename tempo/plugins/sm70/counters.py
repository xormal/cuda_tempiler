#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""sm_70: ИМЕНА СЧЁТЧИКОВ.  Они архитектурные, поэтому живут здесь, а не в core.

ЧТО СЧЁТЧИК ГОВОРИТ И ЧЕГО НЕ ГОВОРИТ (правило перевода, оплаченное сутками):
    ncu меряет РЕСУРС.  СВЯЗЫВАНИЕ доказывает только СДВИГ ВРЕМЕНИ от снятия ресурса.
    Курс между ними = сколько раз отрабатывает место.
Замерено: снятие 18.4 % трафика smem в ЭПИЛОГЕ дало 0.2 % времени, а 14.5 % в МЕЙНЛУПЕ --
7.5 % (6/6 форм).  Один и тот же процент ресурса стоит в 37 раз разного времени.

ЛОВУШКА ЛОКАЛИ: под ru-локалью float("5 238") падает и МОЛЧА теряет все значения >= 1000.
Разбор обязателен с LC_ALL=C (это делает tools/ncu.py, поэтому здесь только имена).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from ..base import PluginCapabilityError

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)

COUNTERS = {
    "smem_wavefronts": [
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    ],
    "smem_requests": [
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld_cmd_read.sum",
    ],
    "conflict_share": [
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    ],
    "tensor_inst": ["sm__inst_executed_pipe_tensor.sum"],
    "inst_total": ["sm__inst_executed.sum"],
    "dram_bytes": ["dram__bytes_read.sum", "dram__bytes_write.sum"],
    "l2_hit": ["lts__t_sector_hit_rate.pct"],
    "occupancy": ["sm__warps_active.avg.pct_of_peak_sustained_active"],
    "stalls": [
        "smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio",
        "smsp__average_warp_latency_issue_stalled_short_scoreboard.ratio",
        "smsp__average_warp_latency_issue_stalled_mio_throttle.ratio",
        "smsp__average_warp_latency_issue_stalled_barrier.ratio",
        "smsp__average_warp_latency_issue_stalled_lg_throttle.ratio",
    ],
}

# Стойки простоев -> что они на самом деле означают на sm_70 (для отчёта, core не ветвится).
STALL_MEANING = {
    "long_scoreboard": "задержка ГЛОБАЛЬНОЙ загрузки не покрыта; подача, а не счёт",
    "short_scoreboard": "задержка РАЗДЕЛЯЕМОЙ (LDS = 26 тактов) не покрыта",
    "mio_throttle": "очередь разделяемой забита -- смотреть ВАЙВФРОНТЫ, а не байты",
    "barrier": "рандеву; при ОДНОМ резидентном блоке останавливает SM целиком",
    "lg_throttle": "очередь глобальных: обычно РАЗБРОС секторов, а не объём",
}


def _load_ncu():
    p = os.path.join(_ROOT, "tools", "ncu.py")
    spec = importlib.util.spec_from_file_location("tempo_tool_ncu", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Sm70Meters:
    def counters(self, kind: str):
        try:
            return list(COUNTERS[kind])
        except KeyError:
            raise PluginCapabilityError(
                "у sm_70 нет группы счётчиков %r; есть: %s"
                % (kind, ", ".join(sorted(COUNTERS)))
            ) from None

    def profile(self, binary: Path, kernel: str, counters: list):
        """Замер через проверенный tools/ncu.py (два независимых маршрута со сверкой)."""
        return _load_ncu().metrics(str(binary), kernel, list(counters))

    def clock_lock(self, card: int, mhz: int):
        """Фиксация частот с ГАРАНТИРОВАННЫМ снятием (try/finally + atexit + сигналы).

        Реализация -- в tools/timeit.py: там она проверена самопроверкой 24/24, и второй её
        копии в дереве быть не должно.
        """
        p = os.path.join(_ROOT, "tools", "timeit.py")
        spec = importlib.util.spec_from_file_location("tempo_tool_timeit", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        for name in ("clock_lock", "ClockLock", "lock_clocks"):
            if hasattr(m, name):
                return getattr(m, name)(card, mhz)
        raise PluginCapabilityError(
            "tools/timeit.py не отдаёт фиксатор частот под известным именем; "
            "мерить без фиксации ЗАПРЕЩЕНО -- дрейф даёт до 60 % разброса"
        )
