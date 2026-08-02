#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""sm_70: ТУЛЧЕЙН.  Обёртка над проверенным `tools/cc_ab.py`, а не вторая его реализация.

ПОЧЕМУ ОБЁРТКА, А НЕ ПЕРЕПИСЬ.  `cc_ab.py` (1893 строки) держит три якоря: Q(W) 4/4, боевой
bwd 255 рег / 41232 Б / 128 нитей БЕЗ ЕДИНОГО ТАКТА GPU, и «kESK ровно +15 регистров».
Вторая реализация того же -- это второе место, которое разъедется с первым.  Правило проекта
(«одна реализация закона на два инструмента») здесь действует буквально.

ТРИ ВЕЩИ, КОТОРЫЕ ЭТА ОБЁРТКА ОБЯЗАНА ВЫТАСКИВАТЬ ОТДЕЛЬНО, потому что они -- РАЗНЫЕ БОЛЕЗНИ:
    регистры / разлив по отчёту / КАДР СТЕКА.
Ненулевой кадр при НУЛЕ разливов по отчёту замерен и означает не «почти влезло», а другое:
компилятор завёл локальный массив.  Плюс отдельно -- LDL/STL В ЦИКЛАХ (по обратным BRA):
разлив в прологе стоит один раз, разлив в теле -- каждый оборот.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

from ..base import BuildResult, EnvReq, PluginCapabilityError

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
_TOOLS = os.path.join(_ROOT, "tools")


def _load(name):
    """Инструмент грузится ПО ПУТИ: он остаётся исполняемым файлом, а не становится модулем пакета."""
    p = os.path.join(_TOOLS, name + ".py")
    spec = importlib.util.spec_from_file_location("tempo_tool_" + name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _env():
    p = os.path.join(_ROOT, "tempo", "cli", "env.py")
    spec = importlib.util.spec_from_file_location("tempo_env", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Sm70Toolchain:
    ARCH = "sm_70"

    def arch_flags(self):
        return ["-arch=sm_70", "-std=c++17", "-O3", "--expt-relaxed-constexpr"]

    def requirements(self):
        e = _env()
        return [
            EnvReq("nvcc", e.nvcc(), "компилятор; отвергнет хост-gcc старше 14"),
            EnvReq("ptxas", e.ptxas(), "регистры / разлив / КАДР СТЕКА (три РАЗНЫХ числа)"),
            EnvReq("cuobjdump", e.cuobjdump(), "SASS для разборщика и проверки LDL/STL в циклах"),
            EnvReq("cuda_include", e.cuda_include(), "заголовки для линтера разделяемой"),
        ]

    def compile(self, source: Path, out: Path, mode="cubin", extra=None, build_dir=None):
        """Сборка через cc_ab.compile_cubin, если он доступен; иначе прямой nvcc.

        КАТАЛОГ СБОРКИ ЗАДАЁТСЯ ЯВНО И НИКОГДА НЕ НАСЛЕДУЕТСЯ. Замеренная цена нарушения:
        один прогон под sudo делает _build/ собственностью root, и потом ВСЁ падает
        PermissionError, а харнесс рапортует 22 FAIL, не имеющих отношения к правке.
        """
        e = _env()
        nvcc = e.nvcc()
        if nvcc is None:
            raise PluginCapabilityError(
                "nvcc не найден. Лечение: TEMPO_ROOTS=/путь/к/conda или CUDA_HOME=... "
                "(см. python3 tempo/cli/env.py)"
            )
        extra = list(extra or [])
        bd = str(build_dir) if build_dir else os.path.join(_ROOT, "build", "tempo")
        os.makedirs(bd, exist_ok=True)
        flags = self.arch_flags() + extra
        if mode == "cubin":
            flags += ["-cubin"]
        elif mode == "shared":
            flags += ["-shared", "-Xcompiler", "-fPIC"]
        cmd = [nvcc] + flags + ["-Xptxas", "-v", str(source), "-o", str(out)]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=bd)
        log = (p.stdout or "") + (p.stderr or "")
        return BuildResult(
            ok=(p.returncode == 0),
            binary=Path(out) if p.returncode == 0 else None,
            regs=_grab(log, r"Used (\d+) registers"),
            spill_st=_grab(log, r"(\d+) bytes spill stores"),
            spill_ld=_grab(log, r"(\d+) bytes spill loads"),
            stack_frame=_grab(log, r"(\d+) bytes stack frame"),
            smem_static=_grab(log, r"(\d+) bytes smem"),
            smem_dynamic=0,
            ldl_stl_in_loops=None,  # считается по SASS отдельно (см. cc_ab / isa_sass)
            log=log,
        )

    def disasm(self, binary: Path) -> str:
        e = _env()
        cu = e.cuobjdump()
        if cu is None:
            raise PluginCapabilityError("cuobjdump не найден (см. tempo/cli/env.py)")
        p = subprocess.run([cu, "-sass", str(binary)], capture_output=True, text=True)
        return p.stdout

    # -- мост к проверенному инструменту -----------------------------------------------------
    def cc_ab(self):
        """Полный `tools/cc_ab.py` как модуль -- его якоря и его selftest остаются его."""
        return _load("cc_ab")


def _grab(text, pattern):
    import re

    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0
