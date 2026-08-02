#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ФАЛЬСИФИКАТОР ПОСЫЛКИ «int8 на Volta даёт x1.85»: замер, а не цитата.

ЗАЧЕМ. В постановке наряда стоит «int8 на HMMA Volta замерен: 96.5 ТОПС против 52.2 ТФЛОП/с fp16
= x1.85». Число 52.2 -- это колонка DP4A в нашей же таблице (COMPILER_VS_HARDWARE.md §9.1), а не
fp16. То есть x1.85 -- это int8-через-HMMA против int8-через-DP4A: обе величины ЦЕЛОЧИСЛЕННЫЕ, и к
fp16 отношение не имеет. Ошибка прошла через памятку, сводку, постановку и отчёт, потому что все
цитировали ОДИН источник и ни один не открыл заголовок колонки.

Здесь она проверяется прямо, на БОЕВЫХ формах Gemma-4-12B, тремя плечами на одной форме:
  fp16  -- torch F.linear (cuBLAS HMMA), боевой путь;
  int8-как-fp16 -- те же байты, уложенные в мантиссу fp16 (0x6400|q = РОВНО 1024+q), и поданные
                   в тот же F.linear. Инструкция ТА ЖЕ (HMMA.884), поэтому ожидание -- ровно 1.00;
  int8 через DP4A -- torch._int_mm, единственный целочисленный тракт, который у sm_70 есть.

ЧЕГО ЭТОТ ЗАМЕР НЕ ГОВОРИТ: он не про точность и не про полосу. Узкий операнд на Volta покупает
ПОЛОСУ и ЁМКОСТЬ (и это отдельно замерено в W8A16), но не ФЛОПы.

Запуск: /home/alex/miniconda3/envs/vllm/bin/python bench/int8_roof.py --dev 0
"""

import argparse
import json
import os
import statistics
import time

PROJ = [
    ("q", 3840, 4096),
    ("k,v", 3840, 2048),
    ("o", 4096, 3840),
    ("gate,up", 3840, 15360),
    ("down", 15360, 3840),
]
MS = [
    512,
    2048,
    8192,
]  # где счёт связывает; при M<=64 связывает чтение, и формат не при чём


def bench(fn, target_ms=120.0):
    import torch

    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(3):
        fn()
    e1.record()
    torch.cuda.synchronize()
    t = e0.elapsed_time(e1) / 3
    it = max(3, min(300, int(target_ms / max(t, 1e-4))))
    e0.record()
    for _ in range(it):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default="/mnt/d1/alex/tempo/data")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.dev))
    import torch

    dev = "cuda:0"
    rows = []
    for role, K, N in PROJ:
        for M in MS:
            flop = 2.0 * M * N * K
            x16 = torch.randn(M, K, device=dev, dtype=torch.float16) * 0.05
            w16 = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.05
            # int8, уложенный в мантиссу fp16: 0x6400 = 1024.0, мантисса 10 бит -> 0x6400|q ТОЧНО
            # равно 1024+q для q в 0..255. Ни одной инструкции конверсии.
            xq = torch.randint(0, 256, (M, K), device=dev, dtype=torch.int32)
            wq = torch.randint(0, 256, (N, K), device=dev, dtype=torch.int32)
            x8f = (0x6400 | xq).to(torch.int16).view(torch.float16)
            w8f = (0x6400 | wq).to(torch.int16).view(torch.float16)
            x8 = (xq - 128).to(torch.int8)
            w8 = (wq - 128).to(torch.int8)
            # ГЕЙТ: укладка обязана быть ТОЧНОЙ, иначе меряем не то
            exact = bool(((x8f.float() - 1024.0).to(torch.int32) == xq).all().item())

            r_fp16, r_i8f, r_dp4a = [], [], []
            err = None
            for _ in range(args.rounds):
                a = bench(lambda: torch.nn.functional.linear(x16, w16))
                b = bench(lambda: torch.nn.functional.linear(x8f, w8f))
                r_fp16.append(a)
                r_i8f.append(b)
                try:
                    c = bench(lambda: torch._int_mm(x8, w8.t()))
                    r_dp4a.append(c)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
            med = statistics.median
            rec = {
                "role": role,
                "K": K,
                "N": N,
                "M": M,
                "exact_pack": exact,
                "fp16_ms": med(r_fp16),
                "fp16_tflops": flop / (med(r_fp16) * 1e-3) / 1e12,
                "int8_as_fp16_ms": med(r_i8f),
                "int8_as_fp16_tflops": flop / (med(r_i8f) * 1e-3) / 1e12,
                "ratio_int8fp16_over_fp16": med(r_fp16) / med(r_i8f),
            }
            if r_dp4a:
                rec["dp4a_ms"] = med(r_dp4a)
                rec["dp4a_tops"] = flop / (med(r_dp4a) * 1e-3) / 1e12
                rec["ratio_dp4a_over_fp16"] = med(r_fp16) / med(r_dp4a)
            else:
                rec["dp4a_err"] = err
            rows.append(rec)
            print(
                f"  {role:9s} K{K:6d} N{N:6d} M{M:5d}  fp16 {rec['fp16_tflops']:6.2f} | "
                f"int8-как-fp16 {rec['int8_as_fp16_tflops']:6.2f} "
                f"(x{rec['ratio_int8fp16_over_fp16']:.3f}) | "
                + (
                    f"DP4A {rec['dp4a_tops']:6.2f} (x{rec['ratio_dp4a_over_fp16']:.3f})"
                    if r_dp4a
                    else f"DP4A -- {err}"
                )
                + ("  укладка ТОЧНА" if exact else "  !!! УКЛАДКА НЕТОЧНА")
            )
            del x16, w16, xq, wq, x8f, w8f, x8, w8
            torch.cuda.empty_cache()
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "int8_roof.json")
    with open(p, "w") as f:
        json.dump(
            {
                "gpu": torch.cuda.get_device_name(0),
                "rows": rows,
                "ts": time.strftime("%Y-%m-%d %H:%M"),
            },
            f,
            indent=1,
            ensure_ascii=False,
        )
    print("# записано:", p)


if __name__ == "__main__":
    main()
