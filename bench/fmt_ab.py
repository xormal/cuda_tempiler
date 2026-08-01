# -*- coding: utf-8 -*-
"""A/B ФОРМАТОВ 0/6/7/8 боевого volta_fwd_ws: ПАРНЫЕ ОТНОШЕНИЯ, чередование ВНУТРИ раунда.
Нужен потому, что порядок 6/7/8 -- приёмочный тест модели, а числа в docs получены на другом
дереве. Меряется ТОЛЬКО время (байты K/V не переупакованы под смещённый байт, ответ у 6/7/8
поэтому неверен -- ровно методика фальсификаторов из docs)."""

import importlib.util, statistics, subprocess, sys, time
import torch

SO = "../VLLM_fa2/work_e4m3/build/fa2sm70_prefill/fa2sm70_prefill.so"
spec = importlib.util.spec_from_file_location("fa2sm70_prefill", SO)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def env():
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
    n = subprocess.run(
        ["nvidia-smi", "-i", "1", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return q, len([x for x in n.split("\n") if x.strip()])


B, H, Hkv, Sq, D = 1, 16, 2, 512, 256


def run(Sk, rounds=41):
    dev = "cuda"
    Q = torch.randn(B, Sq, H, D, device=dev, dtype=torch.float16)
    Kq = torch.randint(-100, 100, (B, Hkv, Sk, D), device=dev, dtype=torch.int8)
    Vq = torch.randint(-100, 100, (B, Hkv, Sk, D), device=dev, dtype=torch.int8)
    Ks = torch.rand(B, Hkv, Sk, device=dev, dtype=torch.float32) * 0.01 + 0.01
    Vs = torch.rand(B, Hkv, Sk, device=dev, dtype=torch.float32) * 0.01 + 0.01
    Ks2 = (
        torch.stack([Ks, Vs], dim=-1).reshape(B, Hkv, 2 * Sk).contiguous()
    )  # ЧЕРЕДУЮЩАЯСЯ таблица (fmt 8)
    none = torch.empty(0, device=dev, dtype=torch.float32)
    sc = D**-0.5
    calls = {
        0: lambda: m.attn_fwd_volta_i8(Q, Kq, Ks, Vq, Vs, sc, True, 0),
        6: lambda: m.attn_fwd_volta_i8(Q, Kq, Ks, Vq, Vs, sc, True, 6),
        7: lambda: m.attn_fwd_volta_i8(Q, Kq, none, Vq, none, sc, True, 7),
        8: lambda: m.attn_fwd_volta_i8(Q, Kq, Ks2, Vq, none, sc, True, 8),
    }
    for f in calls:
        for _ in range(3):
            calls[f]()
    torch.cuda.synchronize()
    per = {f: [] for f in calls}
    for r in range(rounds):
        for f in sorted(calls):  # ЧЕРЕДОВАНИЕ ВНУТРИ РАУНДА
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                calls[f]()
            torch.cuda.synchronize()
            per[f].append((time.perf_counter() - t0) / 3)
    ratios = {
        f: statistics.median([per[0][i] / per[f][i] for i in range(rounds)])
        for f in calls
    }
    return ratios, {f: statistics.median(per[f]) * 1e3 for f in calls}


print(
    "# volta_fwd_ws, B=1 H=16 Hkv=2 Sq=512 causal, ПАРНЫЕ отношения к формату 0 (>1 = быстрее)"
)
for Sk in [8192, 32768, 131072]:
    e0 = env()
    rr, tt = run(Sk)
    e1 = env()
    print(
        "Sk=%-7d частота %s -> %s, чужих процессов %d/%d"
        % (Sk, e0[0], e1[0], e0[1], e1[1])
    )
    for f in sorted(rr):
        print("    формат %-2d  отношение к 0 = %.4f   %.3f мс" % (f, rr[f], tt[f]))
