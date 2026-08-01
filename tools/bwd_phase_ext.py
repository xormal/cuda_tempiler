# -*- coding: utf-8 -*-
"""СБОРКА БОЕВОГО РАСШИРЕНИЯ НА ДВОЙНИКЕ (для ЗАМЕРА ВРЕМЕНИ, вне боевого дерева).

МИНА, РАДИ КОТОРОЙ ЭТОТ ФАЙЛ СУЩЕСТВУЕТ. Очевидный путь -- прокинуть каталог двойника через
FA2SM70_EXTRA_NVCC="-I<двойник>" -- НЕ РАБОТАЕТ И МОЛЧА ЛЖЁТ. torch.utils.cpp_extension кладёт
-I из extra_include_paths в common_cflags, а extra_cuda_cflags дописывает ПОСЛЕ (cpp_extension.py:
2185 и 2203+), поэтому боевой каталог fmha_kernel в списке -I стоит РАНЬШЕ, и <kernel_backward.h>
резолвится в БОЕВОЙ файл. Маска тогда не действует НИ НА ЧТО, все варианты собираются в одно и то
же ядро, а разложение выходит "все фазы бесплатны".
ЗАМЕРЕНО: та же плитка 128,64,128, маска 1 -- через приписанный -I даёт 255 регистров (боевое
ядро), через двойник первым в списке -I даёт 249. Разница видна только так; по времени она была бы
неотличима от честного нуля.
Поэтому здесь расширение собирается СВОИМ вызовом load() с двойником ПЕРВЫМ в extra_include_paths.
Боевое дерево только читается (исходник .cu и заголовки), сборка идёт в ..

ЗАПУСКА ЯДЕР ЗДЕСЬ НЕТ: файл только собирает модуль и печатает отпечаток (регистры/разлив и счёт
команд). Замер времени -- отдельным прогоном владельца на свободной карте.
"""

import os
import subprocess
import sys

# КАПКАН: этот каталог автоматически становится sys.path[0], а в нём лежит наш tools/timeit.py --
# он ЗАТЕНЯЕТ стандартный timeit, и `import torch` падает изнутри (_strobelight). Убираем.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]

REPO = "../VLLM_fa2/solutions/fa2_sm70_cutlass_grade"
TWIN = "./profiled/bwd/inc"
BUILD = "./profiled/bwd/ext"
CUDA_HOME = "/opt/conda/miniconda3/envs/cuda128"


def build(mask, seal=1, extra=()):
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    os.environ.setdefault("CUDA_HOME", CUDA_HOME)
    os.environ.setdefault("CC", "/usr/bin/gcc")
    os.environ.setdefault("CXX", "/usr/bin/g++")
    from torch.utils.cpp_extension import load

    name = "fa2_bwd_phase_m%d" % mask
    bdir = os.path.join(BUILD, name)
    os.makedirs(bdir, exist_ok=True)
    flags = [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "-gencode=arch=compute_70,code=sm_70",
        "-DHAS_PYTORCH",
        "-DFMHA_BWD_R7",
        "-DFMHA_BWD_COAL_FO",
        "-DFMHA_STRIP_MASK=%d" % mask,
        "-DFMHA_BWD_PHASE_SEAL=%d" % seal,
        "-Xptxas",
        "-v",
    ] + list(extra)
    inc = [
        TWIN,  # <<< ДВОЙНИК ПЕРВЫМ -- это и есть весь фокус
        os.path.join(REPO, "fa2_src/cutlass/include"),
        os.path.join(REPO, "fa2_src/cutlass/tools/util/include"),
        os.path.join(REPO, "fa2_src/fmha_kernel"),
    ]
    m = load(
        name=name,
        sources=[os.path.join(REPO, "fa2_sm70/csrc/attn_bwd_cutlass.cu")],
        build_directory=bdir,
        extra_include_paths=inc,
        extra_cuda_cflags=flags,
        verbose=True,
    )
    # ВОРОТА ПОРЯДКА -I: если боевой каталог оказался раньше двойника -- собрано НЕ ТО.
    nj = open(os.path.join(bdir, "build.ninja"), encoding="utf-8").read()
    i_twin, i_prod = (
        nj.find("-I" + TWIN),
        nj.find("-I" + os.path.join(REPO, "fa2_src/fmha_kernel")),
    )
    if i_twin < 0 or (0 <= i_prod < i_twin):
        sys.exit(
            "[ПОРЯДОК -I] боевой fmha_kernel стоит РАНЬШЕ двойника -- собрано боевое ядро, "
            "маска не подействовала. Замер такой сборкой недействителен."
        )
    print("[ОК] двойник стоит раньше боевого fmha_kernel в -I")
    so = os.path.join(bdir, name + ".so")
    subprocess.run(
        "%s/bin/cuobjdump -sass %s > %s/k.sass" % (CUDA_HOME, so, bdir), shell=True
    )
    return m, bdir


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
