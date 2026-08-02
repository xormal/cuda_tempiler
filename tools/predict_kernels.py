#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M4: предсказание для ядер, которых модель НЕ ВИДЕЛА.  Выдаёт долю тензорного пика,
которую разрешает временной отпечаток, и связывающий канал.  Сверять с секундомером."""

import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Разборщик SASS переехал из бывшего vendor/ на сторону ПЛАГИНА (он архитектурный).
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "tempo", "plugins", "sm70"
    ),
)
import tempo as T
import isa_sass as IS

so, pat = sys.argv[1], sys.argv[2]
CORE = (
    "--all-blocks" not in sys.argv
)  # по умолчанию -- БЕЗУСЛОВНЫЙ ОСТОВ (законная форма)
if "--nominal" in sys.argv:
    T.MIO_SAFE = False  # ставка "ширина*32": ТОЧНЕЕ, но НЕ граница
kern = T.load_kernels(so)
print(
    "%-46s %5s %6s %8s %8s %8s %-8s %7s"
    % ("ядро", "варп", "команд", "TENSOR", "ResMII", "T_pred", "связывает", "%пика")
)
seen = set()
for k in sorted(kern):
    if not re.search(pat, k):
        continue
    cfg = IS.CFG(kern[k])
    thr = IS.total_threads(cfg.body)
    warps = (thr // 32) if thr else 8
    roles = IS.detect_roles(cfg)
    if roles and all(roles.warps[n] for n in roles.names):
        a = T.analyse_roles(cfg, roles, core=CORE)
        n = sum(r["n"] * r["warps"] for r in a["roles"].values())
    else:
        ins, _ = T.mainloop_ins(cfg, core=CORE)
        if not ins:
            continue
        a = T.analyse(ins, warps)
        n = a["n"]
    key = "res_nom" if ("--nominal" in sys.argv and "res_nom" in a) else "res"
    res = a.get(key, a["res"])
    rm = max(res.values()) if res else 0.0
    tens = res.get("TENSOR", 0.0)
    if tens <= 0:
        continue
    short = k[:96]
    if short in seen:
        continue
    seen.add(short)
    tpred = max(rm, a["RecMII"])
    print(
        "%-46s %5d %6d %8.0f %8.0f %8.0f %-8s %6.1f%%"
        % (
            short,
            warps,
            n,
            tens,
            rm,
            tpred,
            max(res, key=res.get),
            100.0 * tens / tpred,
        )
    )
