# -*- coding: utf-8 -*-
"""СВЕРКА С БОЕВЫМ ДЕРЕВОМ СЕЙЧАС: инертен ли чужой правк, приехавший ПОСРЕДИ перебора.

ПРОИСШЕСТВИЕ. 01.08.2026 в 22:07 сосед по машине отгрузил в боевое дерево перестановку плитки K
амплитудой 64 Б: правка в `volta_fwd_ws.cuh` (сторона ЗАПИСИ, `kcd`) и парная ей в
`gemm/volta_warp_mma.h` (сторона ЧТЕНИЯ, `swj`). Перебор дополнений в этот момент шёл: ранние
варианты собраны против прежнего текста, поздние -- против нового.

РАССУЖДЕНИЕ ГОВОРИТ, что правка инертна: обе стороны стоят под шаблонным SWZ, а отгруженный путь
(`nosc = 8`) раскрывается с SWZ = 0, то есть перестановка компилируется в ничто. НО РАССУЖДЕНИЕ
ЗДЕСЬ -- РОВНО ТОТ ИНСТРУМЕНТ, КОТОРЫЙ СЕГОДНЯ ПРОМАХНУЛСЯ ТРИЖДЫ. Поэтому проверяем замером:
собираем боевое дерево КАК ОНО ЕСТЬ СЕЙЧАС, гоняем то же тело и сверяем ДВЕ величины --
побитовый ответ и счётчик конфликтов -- с базой перебора (закреплённой на тексте 19:07).

Совпали обе -> дрейф инертен для измеряемого ядра, кривая перебора действительна.
Разошлись     -> перебор недействителен целиком, и это надо сказать, а не сгладить.
"""

import hashlib
import json
import os
import subprocess
import sys

# --- ПУТИ ОКРУЖЕНИЯ: единственное место -- tempo/cli/env.py (правило Р8 спецификации) ---
def _tempo_env_load():
    import importlib.util as _u, os as _o

    _p = _o.path.join(
        _o.path.dirname(_o.path.abspath(__file__)), "..", "tempo", "cli", "env.py"
    )
    try:
        _s = _u.spec_from_file_location("tempo_env", _p)
        _m = _u.module_from_spec(_s)
        _s.loader.exec_module(_m)
        return _m
    except Exception:  # инструмент, вынесенный из дерева, обязан остаться запускаемым

        class _Stub:
            def __getattr__(self, _n):
                return lambda *a, **k: None

        return _Stub()


_ENV = _tempo_env_load()


os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
os.environ.setdefault("CUDA_HOME", _ENV.cuda_home() or "")
os.environ.setdefault("CC", "/usr/bin/gcc")
os.environ.setdefault("CXX", "/usr/bin/g++")
os.environ.setdefault("CUDAHOSTCXX", "/usr/bin/g++")

REPO = "../VLLM_fa2/solutions/fa2_sm70_cutlass_grade"
BDIR = "./pad/fwd_ws/red/prod_now/ext"
FLAGS = [
    "-O3",
    "-std=c++17",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-DHAS_PYTORCH",
    "-ccbin",
    "/usr/bin/g++",
    "-Wno-deprecated-gpu-targets",
    "-gencode=arch=compute_70,code=sm_70",
    "-Xptxas",
    "-v",
]
INC = [
    os.path.join(REPO, p)
    for p in (
        "fa2_sm70/csrc",
        "fa2_src/cutlass/include",
        "fa2_src/cutlass/tools/util/include",
        "fa2_src/fmha_kernel",
    )
]


def main():
    import torch

    sys.path.insert(0, REPO)
    import fa2_sm70 as F
    from torch.utils.cpp_extension import load



    os.makedirs(BDIR, exist_ok=True)
    m = load(
        name="pads_fwd_ws_red_prodnow",
        sources=[os.path.join(REPO, "fa2_sm70/csrc/attn_fwd_cutlass.cu")],
        build_directory=BDIR,
        extra_include_paths=INC,
        extra_cuda_cflags=FLAGS,
        verbose=False,
    )
    torch.manual_seed(0)
    B, H, Hkv, Sq, Sk, D = 1, 4, 1, 512, 512, 256
    q = torch.randn(B, Sq, H, D, dtype=torch.float16).cuda()
    k = torch.randn(B, Hkv, Sk, D, dtype=torch.float16).cuda()
    v = torch.randn(B, Hkv, Sk, D, dtype=torch.float16).cuda()
    c = F.quantize_kv_prefill(k, v, fmt=8)
    o, lse = m.attn_fwd_volta_i8(
        q.contiguous(),
        c["Kb"],
        c["scales"],
        c["Vb"],
        c["scales"],
        float(1.0 / D**0.5),
        True,
        8,
    )
    torch.cuda.synchronize()
    print(
        "ОТВЕТ md5 O=%s" % hashlib.md5(o.detach().cpu().numpy().tobytes()).hexdigest()
    )
    # какие файлы боевого дерева прочитал компилятор ИМЕННО СЕЙЧАС
    dep = os.path.join(BDIR, "attn_fwd_cutlass.cuda.o.d")
    if os.path.exists(dep):
        txt = open(dep).read().replace("\\\n", " ").split(":", 1)[1]
        files = sorted({t for t in txt.split() if t.startswith(REPO)})
        h = hashlib.md5()
        for f in files:
            h.update(
                (
                    f + " " + hashlib.md5(open(f, "rb").read()).hexdigest() + "\n"
                ).encode()
            )
        print(
            "ОТПЕЧАТОК боевых зависимостей (%d файлов): %s"
            % (len(files), h.hexdigest())
        )
        json.dump(
            {f: hashlib.md5(open(f, "rb").read()).hexdigest() for f in files},
            open("./pad/fwd_ws/red/prod_now/deps.json", "w"),
            indent=1,
        )


if __name__ == "__main__":
    main()
