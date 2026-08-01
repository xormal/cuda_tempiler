# -*- coding: utf-8 -*-
"""Собирает САМОПРОВЕРОЧНОЕ наложение для tools/twin.py (пример формата + сборочный образец)."""

import hashlib, json, os

PROD = "../VLLM_fa2/solutions/fa2_sm70_cutlass_grade"
REL = "fa2_sm70/csrc/volta_fwd_ws.cuh"
OUT = "./profiled/overlays/selftest_ws.json"

src = open(os.path.join(PROD, REL), encoding="utf-8").read()
md5 = hashlib.md5(open(os.path.join(PROD, REL), "rb").read()).hexdigest()

HDR = r"""
// ================================================================================================
// [РАЗМЕТКА ФАЗ -- НАЛОЖЕНИЕ tools/twin.py, БОЕВОГО ФАЙЛА НЕТ НИ ОДНОГО БАЙТА]
// ================================================================================================
// Этот блок ПОРОЖДЁН, а не написан в боевом дереве. Он даёт МАСКУ фаз: снять любую комбинацию,
// чтобы невязка 1 = SUM(s_i) + (s_all - SUM(s_i)) + (1 - s_all) разделилась на ПЕРЕКРЫТИЕ и
// НЕНАЗВАННОЕ. Одним числом (прежний DIAG) снимается ровно одна фаза -- перекрытие тогда не
// отделить от неназванного.
//   бит 0  gemm1  первое умножение Q*K^T   (прежний DIAG=5)
//   бит 1  gemm2  второе умножение P*V     (прежний DIAG=3)
// Маска приходит ДВУМЯ путями и они сведены в одно kStrip:
//   -DFMHA_STRIP_MASK=N при сборке (это подставляет twin.py build --mask N), и
//   кодами 64..95 в СУЩЕСТВУЮЩЕМ параметре DIAG (64 + mask) -- новый параметр в конец списка
//   менял бы УКРАШЕННОЕ ИМЯ всех прежних инстанцирований, и побайтового равенства с боевой
//   сборкой было бы уже не доказать.
#ifndef FMHA_STRIP_MASK
#define FMHA_STRIP_MASK 0u
#endif
// [ПОЧЕМУ push_macro, А НЕ #ifndef] Те же имена определяет fa2_src/fmha_kernel/fwd_phase.h
// (разметка ДРУГОГО форварда), и общий .cu включает оба заголовка. При голом #ifndef смысл
// макроса зависел бы от ПОРЯДКА ВКЛЮЧЕНИЯ: переставили строки -- и маска молча превратилась в
// `if constexpr (true)`, то есть фальсификатор стал бы боевым путём без единого признака.
#pragma push_macro("FMHA_PHASE")
#pragma push_macro("FMHA_PHASE_ELSE")
#pragma push_macro("FMHA_PHASE_ON")
#pragma push_macro("FMHA_PHASE_OFF")
#undef FMHA_PHASE
#undef FMHA_PHASE_ELSE
#undef FMHA_PHASE_ON
#undef FMHA_PHASE_OFF
#define FMHA_PHASE(name, id)      if constexpr (!((kStrip) & (1u << (id))))
#define FMHA_PHASE_ELSE(name, id) else
#define FMHA_PHASE_ON(id)         (!((kStrip) & (1u << (id))))
#define FMHA_PHASE_OFF(id)        (((kStrip) & (1u << (id))) != 0u)
"""

DIAG_OLD = "          int DIAG = 0,"
DIAG_NEW = r"""          // [МАСКА ФАЗ -- ТОТ ЖЕ ПАРАМЕТР, КОДЫ 64..95] DIAG = 64 + mask снимает ЛЮБУЮ
          // комбинацию фаз. Умолчание берётся из макроса FMHA_STRIP_MASK (0 в боевой сборке),
          // поэтому `-DFMHA_STRIP_MASK=N` действует без правки диспетчера, а прежние одиночные
          // коды DIAG=0..5 остаются побайтово прежними.
          int DIAG = ((FMHA_STRIP_MASK) != 0u) ? (64 + (int)(unsigned)(FMHA_STRIP_MASK)) : 0,"""

KNO = "  constexpr bool kNoScale = (KVFMT == 4 || KVFMT == 5 || KVFMT == 7);"
KSTRIP = r"""
  // [СВЕДЕНИЕ ДВУХ ПУТЕЙ В ОДНУ МАСКУ] Прежний одиночный DIAG переводится в СВОЙ бит, маска
  // добавляется поверх. Отсюда: DIAG == 0 -> kStrip == 0 -> ни одного `if constexpr` с истинным
  // условием -> боевой путь побайтово прежний (это и проверяет twin.py build --mask 0).
  constexpr unsigned kStripLegacy = (DIAG == 5) ? (1u << 0)      // первое умножение
                                  : (DIAG == 3) ? (1u << 1)      // второе умножение
                                                : 0u;
  constexpr unsigned kStrip = kStripLegacy
                            | ((DIAG >= 64 && DIAG < 96) ? (unsigned)(DIAG - 64) : 0u);"""

G1_OLD = (
    "    if constexpr (DIAG != 5) s.template accumulate_bi8<D, KVFMT, SWZ, KSP>"
    "(sQ, LDQ, wm * SM, sK, LDK8, wn * SN, lane);"
)
G1_NEW = r"""    // ФАЗА 0 (первое умножение): HMMA + чтение фрагментов Q/K. Затравка аккумулятора
    // (seed_rank1) ОСТАЁТСЯ снаружи -- она читает sQsum из разделяемой, поэтому s.acc не
    // сворачивается в константу и весь софтмакс ниже сохраняет зависимость от накопителя.
    FMHA_PHASE(gemm1, 0) {
      s.template accumulate_bi8<D, KVFMT, SWZ, KSP>(sQ, LDQ, wm * SM, sK, LDK8, wn * SN, lane);
    }"""

G2_OLD = (
    "    if constexpr (DIAG != 3) o.template accumulate_bi8<BK, KVFMT>"
    "(sP, LDP, wm * OM, sV, LDV, wn * ON, lane);"
)
G2_NEW = r"""    // ФАЗА 1 (второе умножение): HMMA + чтение фрагментов P/V. rescale_fold остаётся
    // СНАРУЖИ -- он единственный потребитель vb_run, и, унеся его вместе с фазой, мы унесли бы
    // ВЕСЬ второй канал редукции sRedS: замер приписал бы этой фазе чужую работу.
    FMHA_PHASE(gemm2, 1) {
      o.template accumulate_bi8<BK, KVFMT>(sP, LDP, wm * OM, sV, LDV, wn * ON, lane);
    }"""

TAIL = "}  // namespace fwd_ws\n}  // namespace fa2_sm70"
TAIL_ADD = r"""

// Возвращаем имена разметки владельцу (см. пояснение у push_macro выше): за пределами этого
// заголовка FMHA_PHASE снова означает то, что означал до него.
#pragma pop_macro("FMHA_PHASE_OFF")
#pragma pop_macro("FMHA_PHASE_ON")
#pragma pop_macro("FMHA_PHASE_ELSE")
#pragma pop_macro("FMHA_PHASE")"""

HARNESS = """// ПОРОЖДЁН twin.py из наложения selftest_ws.json. Только инстанцирование: ядро НЕ ЗАПУСКАЕТСЯ,
// сборка идёт на процессоре (nvcc -cubin + cuobjdump -sass).
#include <cuda_fp16.h>
#include "volta_fwd_ws.cuh"
template __global__ void fa2_sm70::fwd_ws::volta_fwd_ws<32,64,256,2,4,true,64,1,1,256>(
  const __half*,const int8_t*,const float*,const int8_t*,const float*,__half*,float*,
  int,int,float,int,int,
  long,long,long,long,long,long,long,long,long,long,long,long,long,long long*,int*);
"""

edits = [
    dict(
        id="mask-header",
        why="определить маску FMHA_STRIP_MASK и макросы фаз; push_macro -- "
        "чтобы смысл не зависел от порядка включения заголовков",
        mode="insert_after",
        anchor="#define PH_OUT(dst)\n#endif",
        replace=HDR,
    ),
    dict(
        id="diag-default",
        why="умолчание DIAG из макроса: -DFMHA_STRIP_MASK=N действует без "
        "правки диспетчера и не трогает список параметров шаблона",
        mode="replace",
        anchor=DIAG_OLD,
        replace=DIAG_NEW,
    ),
    dict(
        id="kstrip",
        why="свести прежний одиночный DIAG и маску в одно kStrip",
        mode="insert_after",
        anchor=KNO,
        replace=KSTRIP,
    ),
    dict(
        id="phase0-gemm1",
        why="фаза 0: первое умножение; затравка аккумулятора оставлена "
        "снаружи, иначе S свернётся в константу и утащит софтмакс",
        mode="replace",
        anchor=G1_OLD,
        replace=G1_NEW,
    ),
    dict(
        id="phase1-gemm2",
        why="фаза 1: второе умножение; rescale_fold оставлен снаружи, иначе "
        "снятие фазы унесёт второй канал редукции sRedS",
        mode="replace",
        anchor=G2_OLD,
        replace=G2_NEW,
    ),
    dict(
        id="mask-footer",
        why="вернуть имена макросов владельцу",
        mode="insert_after",
        anchor=TAIL,
        replace=TAIL_ADD,
    ),
]

for e in edits:
    n = src.count(e["anchor"])
    assert n == 1, (e["id"], n)

ov = {
    "name": "selftest_ws",
    "version": "1",
    "comment": (
        "САМОПРОВЕРКА МЕХАНИЗМА twin.py на боевом заголовке volta_fwd_ws.cuh. "
        "Две фазы -- ровно столько, сколько нужно, чтобы показать пару и её невязку. "
        "Полная разметка форварда/бэкварда живёт в своих наложениях."
    ),
    "prod_root": PROD,
    "prod_include_dirs": [
        "fa2_sm70/csrc",
        "fa2_src/fmha_kernel",
        "fa2_src/cutlass/include",
    ],
    "files": [{"file": REL, "md5": md5, "edits": edits}],
    "write": [{"path": "inst_ws.cu", "text": HARNESS}],
    "build": {
        "harness": "inst_ws.cu",
        "steps": [
            {
                "name": "nvcc -cubin",
                "cmd": "{nvcc} -arch=sm_70 -O3 -std=c++17 -cubin -Wno-deprecated-gpu-targets "
                "-ccbin /usr/bin/g++ {inc} -DFMHA_STRIP_MASK={mask}u {extra} "
                "-o {outdir}/{name}.cubin {harness}",
            },
            {
                "name": "cuobjdump -sass",
                "cmd": "{cuobjdump} -sass {outdir}/{name}.cubin > {outdir}/{name}.sass",
            },
        ],
    },
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write(
    json.dumps(ov, ensure_ascii=False, indent=2) + "\n"
)
print("написано", OUT, "правок", len(edits))
