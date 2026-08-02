#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЯКОРЬ ХАРНЕССА: воспроизведение ОПУБЛИКОВАННОГО отношения forward / SDPA-eff.

Проверяет не ядро, а САМ ИНСТРУМЕНТ (`tools/timeit.py`): попадает ли он в уже напечатанные в
`solutions/fa2_sm70_cutlass_grade/README.md` (раздел "Prefill", causal) числа `ours/eff`, и
накрывает ли бутстрап-интервал опубликованное значение.

    | B, H, Sk, d      | ours/eff |
    | 1, 32, 1024, 128 |   1.07x  |
    | 1, 32, 2048, 128 |   1.09x  |
    | 1, 16, 2048,  64 |   1.10x  |
    | 1,  1, 2048, 128 |   2.00x  |   <- включается причинный split-K

Опубликованные числа сняты как СРЕДНЕЕ по 100 итерациям при зафиксированных 1530 МГц (протокол
`benchmarks/_common.timed`).  Здесь -- другая дисциплина (парные отношения, чередование внутри
раунда, медиана отношений), поэтому совпадение ожидается в пределах бутстрап-интервала плюс
разница протоколов; расхождение НЕ подкручивается, а печатается.

ЗАПУСК (только на ПУСТОЙ карте -- иначе инструмент сам откажется):
    FA2SM70_BUILD_DIR=./build/ext FA2_SUDO_PASS=... \
    <TEMPO_PY> tools/anchor_fwd_sdpa.py --card 1     # <TEMPO_PY> см. tempo/cli/env.py

Порядок жёсткий: гейт среды (карта пуста?) -> сборка/сверка -> фиксация частот -> прогрев ->
раунды.  До прохождения первого шага torch даже не импортируется: отказ обязан быть бесплатным
для чужого замера на соседней карте.
"""

import argparse
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOLDIR = "../VLLM_fa2/solutions/fa2_sm70_cutlass_grade"

# КАТАЛОГ СКРИПТА УХОДИТ ИЗ sys.path ПЕРВЫМ ДЕЙСТВИЕМ -- ЭТО ОБЯЗАТЕЛЬНО, А НЕ ГИГИЕНА.
# Прежней редакции не хватало: она грузила tools/timeit.py по пути (ниже), и это верно, но
# каталог скрипта ВСЁ РАВНО остаётся sys.path[0], а torch внутри себя делает
# `from timeit import default_timer` -- и попадает в НАШ timeit.py.
# ВОСПРОИЗВЕДЕНО 2026-08-01: ImportError: cannot import name 'default_timer' from 'timeit'
# (./tools/timeit.py).  То есть `import torch` в этом самом файле (строка ниже)
# падал ВСЕГДА, на любой карте: хардварный якорь был недостижим и по этой причине тоже.
sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") != HERE]

# tools/timeit.py грузится ПО ПУТИ, а не через sys.path: имя `timeit` заслонило бы стандартный
# модуль, который импортирует torch.  Это не педантизм -- это ровно тот класс аварий, который
# инструмент и должен исключать.
_spec = importlib.util.spec_from_file_location(
    "fa2_timeit", os.path.join(HERE, "timeit.py")
)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

# (B, H, Sk, d, causal, опубликованное ours/eff)
SHAPES = [
    (1, 32, 1024, 128, True, 1.07),
    (1, 32, 2048, 128, True, 1.09),
    (1, 16, 2048, 64, True, 1.10),
    (1, 1, 2048, 128, True, 2.00),
]


def kernel_name(fn):
    """ТОЧКА ВХОДА: имя самого тяжёлого CUDA-ядра одного вызова.  Не установилось -> None."""
    try:
        import torch
        from torch.profiler import profile, ProfilerActivity

        with profile(activities=[ProfilerActivity.CUDA]) as p:
            fn()
            torch.cuda.synchronize()
        best, bt = None, -1.0
        for e in p.key_averages():
            t = getattr(e, "self_device_time_total", None)
            if t is None:
                t = getattr(e, "self_cuda_time_total", 0.0)
            if t and t > bt:
                best, bt = e.key, t
        return best
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, default=1, help="ФИЗИЧЕСКИЙ индекс карты")
    ap.add_argument("--rounds", type=int, default=21)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--mhz", type=int, default=1530)
    ap.add_argument(
        "--tol", type=float, default=0.03, help="допуск на разницу протоколов"
    )
    ap.add_argument(
        "--out", default=os.path.join(HERE, "..", "data", "anchor_fwd_sdpa.json")
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="мерить вопреки гейту; результат ВСЁ РАВНО остаётся недействительным",
    )
    a = ap.parse_args()

    # --- ШАГ 0: гейт среды ДО импорта torch ---------------------------------------------------
    h0 = T.Harness(card=a.card, lock_mhz=a.mhz)
    ok, v, facts = h0.precheck()
    print(
        f"=== ГЕЙТ СРЕДЫ, карта {a.card}: "
        f"{'МОЖНО МЕРИТЬ' if ok else '*** МЕРИТЬ НЕЛЬЗЯ ***'} ==="
    )
    for k, val in facts.items():
        print(f"  {k}: {val}")
    for f in v.fatal:
        print(f"ОТМЕНА: {f}")
    print(f"--- НЕ РАЗОБРАНО / НЕ УСТАНОВЛЕНО ({len(v.unparsed)}) ---")
    for u in v.unparsed:
        print(f"  ? {u}")
    if not ok and not a.force:
        print(
            "\nЯКОРЬ НЕ СНЯТ: карта занята.  torch не импортировался, память не занята, "
            "чужой замер не испорчен."
        )
        return 2

    # --- ШАГ 1: железо ------------------------------------------------------------------------
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.card))
    sys.path.insert(0, SOLDIR)
    sys.path.insert(0, os.path.join(SOLDIR, "benchmarks"))
    import torch
    import torch.nn.functional as F
    from _common import exact_ref, relL2, expand_kv, efficient_backend
    import fa2_sm70

    rows = []
    for B, H, Sk, d, causal, published in SHAPES:
        torch.manual_seed(0)
        scale = 1.0 / math.sqrt(d)
        Q = torch.randn(B, H, Sk, d, device="cuda", dtype=torch.float16) * 0.5
        K = torch.randn(B, H, Sk, d, device="cuda", dtype=torch.float16) * 0.5
        V = torch.randn(B, H, Sk, d, device="cuda", dtype=torch.float16) * 0.5
        Ks, Vs = expand_kv(K, V, H)

        # Бэкенд SDPA переключается ОДИН раз на всю форму, а не внутри замеряемого вызова:
        # вход/выход из контекст-менеджера на каждой итерации мерился бы вместе с ядром.
        ctx = efficient_backend()
        ctx.__enter__()

        def ours():
            return fa2_sm70.attention(Q, K, V, causal=causal, scale=scale)[0]

        def sdpa_eff():
            return F.scaled_dot_product_attention(
                Q, Ks, Vs, is_causal=causal, scale=scale
            )

        # ГЕЙТ КОРРЕКТНОСТИ ПЕРЕД ВРЕМЕНЕМ: оба варианта против точного fp32-эталона.
        def check(_Q=Q, _K=K, _V=V, _sc=scale, _c=causal):
            ref = exact_ref(_Q, _K, _V, _sc, _c)
            eo, ee = relL2(ours(), ref), relL2(sdpa_eff(), ref)
            if not (eo < 1e-3 and ee < 1e-3):
                return (False, f"relL2 ours={eo:.2e} eff={ee:.2e} (порог 1e-3)")
            return (True, f"relL2 ours={eo:.2e} eff={ee:.2e}")

        h = T.Harness(
            card=a.card,
            rounds=a.rounds,
            warmup=a.warmup,
            iters=a.iters,
            lock_mhz=a.mhz,
            force=a.force,
        )
        try:
            res = h.compare(
                {"sdpa_eff": sdpa_eff, "ours": ours},
                base="sdpa_eff",
                check=check,
                label=f"forward B{B} H{H} Sk{Sk} d{d} causal={causal}",
                entry_probe={
                    "sdpa_eff": lambda: kernel_name(sdpa_eff),
                    "ours": lambda: kernel_name(ours),
                },
            )
        finally:
            ctx.__exit__(None, None, None)
        print()
        print(res.report())
        s = res.summary.get("ours")
        if s:
            inside = s["ci_lo"] <= published <= s["ci_hi"]
            near = abs(s["ratio_median"] - published) <= a.tol * published
            print(
                f"ЯКОРЬ {B},{H},{Sk},{d}: опубликовано {published:.2f}x, замерено "
                f"{s['ratio_median']:.4f}x ДИ [{s['ci_lo']:.4f},{s['ci_hi']:.4f}] -> "
                + (
                    "СОШЛОСЬ (в ДИ)"
                    if inside
                    else (
                        f"в ДИ НЕ попало, но в допуске {100 * a.tol:.0f} %"
                        if near
                        else f"РАСХОЖДЕНИЕ {100 * (s['ratio_median'] / published - 1):+.1f} % -- "
                        "разобрать, а не сгладить"
                    )
                )
            )
            rows.append(
                {
                    "shape": [B, H, Sk, d],
                    "published": published,
                    "measured": s["ratio_median"],
                    "ci": [s["ci_lo"], s["ci_hi"]],
                    "inside_ci": bool(inside),
                    "within_tol": bool(near),
                    "valid": res.verdict.valid,
                    "result": res.to_dict(),
                }
            )
        else:
            rows.append(
                {
                    "shape": [B, H, Sk, d],
                    "published": published,
                    "measured": None,
                    "valid": False,
                    "result": res.to_dict(),
                }
            )
        del Q, K, V, Ks, Vs
        torch.cuda.empty_cache()

    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    good = [
        r
        for r in rows
        if r.get("valid") and (r.get("inside_ci") or r.get("within_tol"))
    ]
    print(
        f"\nИТОГ ЯКОРЯ: {len(good)}/{len(rows)} форм сошлись с публикацией; JSON -> {a.out}"
    )
    return 0 if len(good) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
