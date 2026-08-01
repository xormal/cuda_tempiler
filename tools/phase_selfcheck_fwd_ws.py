# -*- coding: utf-8 -*-
"""САМОПРОВЕРКА ФАЗОВОГО ПРОФИЛИРОВЩИКА НА ЯКОРЕ: byte-forward volta_fwd_ws (d=256, KVFMT=8).

ЧТО ЭТО. Инструмент без воспроизведения известного числа -- не инструмент, а новый источник
уверенных неверных ответов. Якорь для phaseprof.py -- таблица §3b docs/VOLTA_SM70.md:

    первое умножение (Q*K^T) 35.1 %   второе (P*V) 19.7 %   софтмакс 15.2 %
    подача 5.7 %              рандеву 2.0 %                 СУММА 77.7 %

Доли получены РУЧНОЙ сборкой ядер со снятой фазой. К счастью, фальсификаторы остались в
ОТГРУЖЕННОМ коде как коды `nosc` (attn_fwd_cutlass.cu:785-789), поэтому воспроизведение не требует
пересборки -- те же самые ядра вызываются из питона:

    nosc = 8      база (боевой байтовый форвард: смещённый байт + чередующаяся таблица)
    nosc = 1508   DIAG=5  снято ПЕРВОЕ умножение
    nosc = 1308   DIAG=3  снято ВТОРОЕ умножение
    nosc = 1108   DIAG=1  снят СОФТМАКС
    nosc = 1408   DIAG=4  снята ПОДАЧА (везущие не везут, рандеву осталось)
    nosc = 1208   DIAG=2  сняты ДВА внутренних рандеву считающих

Ответ при DIAG != 0 НЕВЕРЕН -- читается только время. Поэтому секундомер запускается с
Harness.NO_CHECK: гейт корректности здесь не отключён по небрежности, он ЛОГИЧЕСКИ НЕПРИМЕНИМ,
и харнесс сам напишет это в "НЕ РАЗОБРАНО".

ЗАПУСК (только на СВОБОДНОЙ карте 0 или 1; карты 2-3 -- боевой сервер, не трогать):

    FA2_SUDO_PASS=... CARD=1 /opt/conda/miniconda3/envs/vllm/bin/python phase_selfcheck_fwd_ws.py

FA2_SUDO_PASS нужен харнессу для ФИКСАЦИИ ЧАСТОТ. Без фиксации замер отменяется: на этой машине
разброс частоты даёт до 60 %. Осознанно без неё -- LOCK=0 (результат помечается недействительным).
"""

import importlib.util
import json
import math
import os
import sys

# --- ПОРЯДОК ИМПОРТА. В tools/ лежит timeit.py (чужой харнесс), и каталог скрипта СТОИТ ПЕРВЫМ
# в sys.path автоматически -- он ПЕРЕКРЫВАЕТ стандартный модуль timeit, из-за чего не
# импортируется даже torch. Поэтому: сначала убираем каталог из пути, импортируем стандартное и
# torch, а свои модули грузим ЯВНО ПО ФАЙЛУ.
_TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _TOOLS]

FA2 = "../VLLM_fa2/solutions/fa2_sm70_cutlass_grade"
sys.path.append(FA2)

import torch  # noqa: E402
import fa2_sm70  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


phaseprof = _load("fa2_phaseprof", os.path.join(_TOOLS, "phaseprof.py"))
_TIMEIT = os.path.join(_TOOLS, "timeit.py")
harness_mod = _load("fa2_timeit_harness", _TIMEIT) if os.path.exists(_TIMEIT) else None

D = 256
BASE_CODE = 8
# имя фазы -> (код nosc, доля из §3b)
PHASES = [
    ("gemm1_QK", 1508, 35.1),
    ("gemm2_PV", 1308, 19.7),
    ("softmax", 1108, 15.2),
    ("feed", 1408, 5.7),
    ("rendez", 1208, 2.0),
]


def quant_pertoken(x):
    a = x.float().abs().amax(-1, keepdim=True).clamp_min(6.1e-5) / 127.0
    q = torch.clamp(torch.round(x.float() / a), -127, 127).to(torch.int8).contiguous()
    return q, a.squeeze(-1).contiguous().float()


def to_biased(q):
    return (q.to(torch.int16) + 128).to(torch.uint8).view(torch.int8).contiguous()


def main():
    card = int(os.environ.get("CARD", "1"))
    if card not in (0, 1):
        print("ОТКАЗ: карты 2 и 3 -- боевой сервер. Разрешены только 0 и 1.")
        return 2
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(card))
    Sq = int(os.environ.get("SQ", 512))
    H, Hkv = int(os.environ.get("H", 16)), 2
    Sk = int(os.environ.get("SK", 32768))
    rounds = int(os.environ.get("ROUNDS", 21))
    lock_mhz = None if os.environ.get("LOCK", "1") == "0" else 1530

    if harness_mod is None:
        print(
            "ОТКАЗ: не найден tools/timeit.py -- харнесс замера обязателен "
            "(гейт соседа, фиксация частот, вердикт действительности)."
        )
        return 2

    torch.manual_seed(0)
    pext = fa2_sm70._ext.prefill_ext()
    sc = 1.0 / math.sqrt(D)

    q = (
        torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3
    ).contiguous()
    kf = (
        torch.randn(1, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.4
    ).contiguous()
    vf = (
        torch.randn(1, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.4
    ).contiguous()
    kq, ks = quant_pertoken(kf)
    vq, vs = quant_pertoken(vf)
    kqb, vqb = to_biased(kq), to_biased(vq)
    ksv = torch.stack([ks, vs], dim=-1).contiguous()

    def mk(code):
        return lambda: pext.attn_fwd_volta_i8(q, kqb, ksv, vqb, ksv, sc, True, code)

    variants = {"base": mk(BASE_CODE)}
    for name, code, _ in PHASES:
        variants[name] = mk(code)

    H_ = harness_mod.Harness(
        card=card,
        rounds=rounds,
        warmup=8,
        lock_mhz=lock_mhz,
        allow_no_lock=(lock_mhz is None),
    )
    res = H_.compare(
        variants,
        "base",
        check=harness_mod.Harness.NO_CHECK,
        label="фазы volta_fwd_ws d=256 KVFMT=8 (DIAG отгруженного ядра), Sk=%d" % Sk,
    )
    print(res.report())
    print()

    if not res.summary:
        print("СЕКУНДОМЕР НЕ ЗАПУСКАЛСЯ -- разлагать нечего.")
        return 1

    t0 = res.summary["base"]["median_ms"]
    # t_i восстанавливаем из ПАРНОГО отношения (медиана пораундовых отношений), а не из медианы
    # времени: дрейф частот между вариантами тогда сокращается.
    times = {
        name: t0 / res.summary[name]["ratio_median"]
        for name, _, _ in PHASES
        if name in res.summary
    }

    unparsed = list(res.verdict.unparsed) + [
        "вариант 'снять ВСЕ фазы' в ОТГРУЖЕННОМ ядре ОТСУТСТВУЕТ: DIAG -- одно число, а не маска "
        "битов. Поэтому невязка здесь НЕ РАЗДЕЛЕНА на перекрытие и ненайденное. Разделение даёт "
        "разметка макросом FMHA_PHASE (см. fmha_phase.h) -- одна лишняя сборка.",
        "ПАР по той же причине нет -- перекрытие поточечно не мерено.",
        "фальсификаторы DIAG написаны БЕЗ ПЛОМБ: сбитый хвост НЕ проверялся, доли могут быть "
        "завышены на величину чужой работы, выброшенной компилятором. Проверка -- "
        "`phaseprof.py static` на размеченном ядре.",
        "эпилог (запись O и LSE), пролог и пересчёт накопителя онлайн-софтмакса НЕ НАЗВАНЫ ни "
        "одной из пяти фаз -- они целиком внутри невязки.",
    ]
    if not res.verdict.valid:
        unparsed.insert(
            0,
            "ЗАМЕР ПРИЗНАН НЕДЕЙСТВИТЕЛЬНЫМ харнессом (см. ОТМЕНА выше): доли "
            "ниже читать НЕЛЬЗЯ, они приведены только для отладки.",
        )

    anchor = {name: pct for name, _, pct in PHASES}
    phaseprof.report_shares(
        t0,
        times,
        None,
        {},
        "ms",
        note="volta_fwd_ws d=256 KVFMT=8, фальсификаторы DIAG "
        "(%s)" % ("ДЕЙСТВИТЕЛЬНО" if res.verdict.valid else "*** НЕДЕЙСТВИТЕЛЬНО ***"),
        anchor=anchor,
    )
    phaseprof.print_unparsed(unparsed)

    dst = os.environ.get("OUT", "./data/phase_fwd_ws_measured.json")
    json.dump(
        {
            "base": t0,
            "phases": times,
            "all": None,
            "pairs": {},
            "units": "ms",
            "note": "volta_fwd_ws d=256 KVFMT=8 (DIAG отгруженного ядра), карта %d, Sk=%d, "
            "вердикт=%s" % (card, Sk, "действителен" if res.verdict.valid else "НЕТ"),
            "anchor": anchor,
            "unparsed": unparsed,
            "harness": res.to_dict(),
        },
        open(dst, "w"),
        ensure_ascii=False,
        indent=1,
    )
    print("\n  записано: %s" % dst)
    print("  разбор повторно:  python3 phaseprof.py replay --times %s" % dst)
    return 0 if res.verdict.valid else 1


if __name__ == "__main__":
    sys.exit(main())
