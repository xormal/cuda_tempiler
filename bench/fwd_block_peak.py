# -*- coding: utf-8 -*-
"""ЗАМЕР ДОЛИ ТЕНЗОРНОГО ПИКА у боевого volta_fwd_block (d=256), карта 1.
Нужен, потому что число «95.2 %» в docs/VOLTA_SM70.md на этом ядре НЕ МЕРИЛОСЬ (см. README §3).
Ничего в чужих деревьях не меняется: .so загружается как есть."""

import os, sys, time, importlib.util, subprocess
import torch

SO = "../VLLM_fa2/work_e4m3/build/fa2sm70_prefill/fa2sm70_prefill.so"
spec = importlib.util.spec_from_file_location("fa2sm70_prefill", SO)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def clocks():
    q = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "1",
            "--query-gpu=clocks.sm,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return q


def bench(B, H, Hkv, Sq, Sk, causal, reps=30):
    dev = "cuda"
    Q = torch.randn(B, Sq, H, 256, device=dev, dtype=torch.float16)
    K = torch.randn(B, Hkv, Sk, 256, device=dev, dtype=torch.float16)
    V = torch.randn(B, Hkv, Sk, 256, device=dev, dtype=torch.float16)
    sc = 256**-0.5
    for _ in range(5):
        m.attn_fwd_volta(Q, K, V, sc, causal)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m.attn_fwd_volta(Q, K, V, sc, causal)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    t = ts[len(ts) // 2]
    # FLOP: две матрицы 2*Sq*Sk*D каждая; causal -> примерно половина (bottom-right)
    frac = 0.5 * (1.0 + (Sk - Sq) / float(Sk)) if causal else 1.0
    flop = 2.0 * 2.0 * B * H * Sq * Sk * 256 * frac
    return t, flop / t / 1e12


print("# ЯДРО volta_fwd_block d=256, BQ=64/BK=64/2x4 варпа (отгруженная конфигурация)")
print("# пик HMMA.884 на V100 при 1530 МГц = 125.3 ТФЛОП/с")
print("%-28s %10s %10s %8s  %s" % ("форма", "мс", "ТФЛОП/с", "%пика", "частота/мощн"))
for B, H, Hkv, Sq, Sk, causal in [
    (1, 8, 8, 4096, 4096, False),
    (1, 8, 8, 4096, 4096, True),
    (1, 16, 16, 2048, 2048, False),
    (2, 8, 8, 8192, 8192, False),
    (1, 8, 8, 8192, 8192, False),
]:
    try:
        t, tf = bench(B, H, Hkv, Sq, Sk, causal)
        print(
            "%-28s %10.3f %10.2f %7.1f%%  %s"
            % (
                "B%dH%dSq%dSk%d%s" % (B, H, Sq, Sk, "c" if causal else ""),
                t * 1e3,
                tf,
                100 * tf / 125.3,
                clocks(),
            )
        )
    except RuntimeError as e:
        print("ОШИБКА", e)
        break
