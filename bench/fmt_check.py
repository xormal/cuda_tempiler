# -*- coding: utf-8 -*-
"""ПРОВЕРКА ТОЧКИ ВХОДА: действительно ли вызовы 0/6/7/8 попадают в РАЗНЫЕ ядра.
6 и 8 читают ОДНИ И ТЕ ЖЕ масштабы (8 -- та же таблица, только чередующаяся) -> обязаны совпасть
ПОБИТОВО; 7 масштабов не читает -> обязан отличаться. Если это не так, замер мерил не то."""

import importlib.util, os, subprocess, torch

SO = "../VLLM_fa2/work_e4m3/build/fa2sm70_prefill/fa2sm70_prefill.so"
spec = importlib.util.spec_from_file_location("fa2sm70_prefill", SO)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
B, H, Hkv, Sq, Sk, D = 1, 16, 2, 512, 2048, 256
dev = "cuda"
torch.manual_seed(0)
Q = torch.randn(B, Sq, H, D, device=dev, dtype=torch.float16)
Kq = torch.randint(-100, 100, (B, Hkv, Sk, D), device=dev, dtype=torch.int8)
Vq = torch.randint(-100, 100, (B, Hkv, Sk, D), device=dev, dtype=torch.int8)
Ks = torch.rand(B, Hkv, Sk, device=dev, dtype=torch.float32) * 0.01 + 0.01
Vs = torch.rand(B, Hkv, Sk, device=dev, dtype=torch.float32) * 0.01 + 0.01
Ks2 = torch.stack([Ks, Vs], dim=-1).reshape(B, Hkv, 2 * Sk).contiguous()
none = torch.empty(0, device=dev, dtype=torch.float32)
sc = D**-0.5
O = {}
O[0] = m.attn_fwd_volta_i8(Q, Kq, Ks, Vq, Vs, sc, True, 0)[0]
O[6] = m.attn_fwd_volta_i8(Q, Kq, Ks, Vq, Vs, sc, True, 6)[0]
O[7] = m.attn_fwd_volta_i8(Q, Kq, none, Vq, none, sc, True, 7)[0]
O[8] = m.attn_fwd_volta_i8(Q, Kq, Ks2, Vq, none, sc, True, 8)[0]
print(
    "6 == 8 побитово:", torch.equal(O[6], O[8]), " (обязано быть True -- та же таблица)"
)
print(
    "6 == 7 побитово:",
    torch.equal(O[6], O[7]),
    " (обязано быть False -- 7 без таблицы)",
)
print(
    "0 == 6 побитово:",
    torch.equal(O[0], O[6]),
    " (False: 0 -- доп.код, 6 -- смещённый байт)",
)
print("конечность:", {f: bool(torch.isfinite(O[f]).all()) for f in O})
pid = os.getpid()
apps = subprocess.run(
    ["nvidia-smi", "-i", "1", "--query-compute-apps=pid", "--format=csv,noheader"],
    capture_output=True,
    text=True,
).stdout.split()
print(
    "процессы на карте 1:",
    apps,
    " свой pid:",
    pid,
    " ЧУЖИХ:",
    len([a for a in apps if a.strip() and int(a) != pid]),
)
