# -*- coding: utf-8 -*-
"""Один запуск боевого ядра под ncu. Формат/ядро выбирается аргументом."""

import importlib.util, sys, torch

SO = "../VLLM_fa2/work_e4m3/build/fa2sm70_prefill/fa2sm70_prefill.so"
spec = importlib.util.spec_from_file_location("fa2sm70_prefill", SO)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
which = sys.argv[1] if len(sys.argv) > 1 else "ws0"
dev = "cuda"
D = 256
if which.startswith("ws"):
    f = int(which[2:])
    B, H, Hkv, Sq, Sk = 1, 16, 2, 512, 2048
    Q = torch.randn(B, Sq, H, D, device=dev, dtype=torch.float16)
    Kq = torch.randint(-100, 100, (B, Hkv, Sk, D), device=dev, dtype=torch.int8)
    Vq = torch.randint(-100, 100, (B, Hkv, Sk, D), device=dev, dtype=torch.int8)
    Ks = torch.rand(B, Hkv, Sk, device=dev, dtype=torch.float32) * 0.01 + 0.01
    Vs = torch.rand(B, Hkv, Sk, device=dev, dtype=torch.float32) * 0.01 + 0.01
    Ks2 = torch.stack([Ks, Vs], dim=-1).reshape(B, Hkv, 2 * Sk).contiguous()
    none = torch.empty(0, device=dev, dtype=torch.float32)
    sc = D**-0.5
    if f == 7:
        m.attn_fwd_volta_i8(Q, Kq, none, Vq, none, sc, True, 7)
    elif f == 8:
        m.attn_fwd_volta_i8(Q, Kq, Ks2, Vq, none, sc, True, 8)
    else:
        m.attn_fwd_volta_i8(Q, Kq, Ks, Vq, Vs, sc, True, f)
else:
    B, H, Hkv, Sq, Sk = 1, 8, 8, 2048, 2048
    Q = torch.randn(B, Sq, H, D, device=dev, dtype=torch.float16)
    K = torch.randn(B, Hkv, Sk, D, device=dev, dtype=torch.float16)
    V = torch.randn(B, Hkv, Sk, D, device=dev, dtype=torch.float16)
    m.attn_fwd_volta(Q, K, V, D**-0.5, False)
torch.cuda.synchronize()
