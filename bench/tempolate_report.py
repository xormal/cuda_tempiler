#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""Сводка хода 1: выход темполятора против ДВУХ планок сразу.

Правило проекта: метрика "выход против входа" печатается ТОЛЬКО в паре с абсолютной планкой
(cuBLAS-fp16). Первая без второй -- самообман: наивный вход обогнать тривиально.
"""
import json, sys, collections

ROLE = {(3840, 4096): "q", (3840, 2048): "k,v", (4096, 3840): "o",
        (3840, 15360): "gate,up", (15360, 3840): "down"}
ORDER = [(3840, 4096), (3840, 2048), (4096, 3840), (3840, 15360), (15360, 3840)]
PEAK = 125.3


def load_grid(p):
    base, cand = {}, collections.defaultdict(list)
    build = {}
    for l in open(p):
        if l.startswith("BUILD "):
            d = json.loads(l[6:]); build[d["tag"]] = d
        elif l.startswith("BASE "):
            d = json.loads(l[5:]); base[(d["K"], d["N"], d["M"])] = d
        elif l.startswith("CAND "):
            d = json.loads(l[5:]); cand[(d["K"], d["N"], d["M"])].append(d)
    return build, base, cand


def load_old(p):
    """Прежний замер того же дерева: rows_cxx (наш мейнлуп + наивный вход) и rows_torch (W8A16)."""
    old = {}
    try:
        j = json.load(open(p))
    except Exception:
        return old
    for r in j.get("rows_cxx", []):
        old[(r["K"], r["N"], r["M"])] = dict(r)
    for r in j.get("rows_torch", []):
        d = old.setdefault((r["K"], r["N"], r["M"]), {})
        d["w8gemv"] = r.get("w8a16_gemv_tflops", 0)
        d["w8hmma"] = r.get("w8a16_hmma_tflops", 0)
    return old


def main():
    grid = sys.argv[1]
    oldp = sys.argv[2] if len(sys.argv) > 2 else None
    build, base, cand = load_grid(grid)
    old = load_old(oldp) if oldp else {}
    print("%-8s %6s %6s %6s | %9s %7s | %-22s %8s %7s %8s | %8s %8s %8s" % (
        "роль", "K", "N", "M", "cuBLAS", "%пика", "гиперформа темполятора", "ТФЛОП/с", "%пика",
        "xcuBLAS", "было", "xвход", "W8A16"))
    print("-" * 150)
    agg = []
    for kn in ORDER:
        for M in (1, 8, 32, 128, 512, 2048, 8192):
            key = (kn[0], kn[1], M)
            if key not in base:
                continue
            b = base[key]
            rows = sorted(cand[key], key=lambda x: -x["ratio_med"])
            if not rows:
                continue
            w = rows[0]
            ref = [x for x in rows if x["tag"] == "REF_ship128x128"]
            o = old.get(key, {})
            naive = o.get("naive_tflops", 0) or 0
            w8 = max(o.get("w8gemv", 0) or 0, o.get("w8hmma", 0) or 0)
            print("%-8s %6d %6d %6d | %9.2f %6.1f%% | %-22s %8.2f %6.1f%% %8.4f | %8.4f %8.0f %8.3f" % (
                ROLE[kn], kn[0], kn[1], M, b["cublas_tflops"], 100 * b["cublas_tflops"] / PEAK,
                w["tag"], w["tflops"], 100 * w["tflops"] / PEAK, w["ratio_med"],
                (o.get("volta_ratio_med", -1) if o else -1),
                (w["tflops"] / naive) if naive > 0 else -1, w8))
            agg.append((ROLE[kn], M, w["ratio_med"], (o.get("volta_ratio_med", -1) if o else -1)))
    print()
    pre = [a for a in agg if a[1] >= 128]
    dec = [a for a in agg if a[1] < 128]
    for name, s in (("ПРЕФИЛЛ M>=128", pre), ("ДЕКОД M<=32", dec)):
        if not s:
            continue
        r = sorted(x[2] for x in s)
        g = 1.0
        for x in s:
            g *= x[2]
        g **= 1.0 / len(s)
        won = [x for x in s if x[2] > 1.0]
        par = [x for x in s if 0.95 <= x[2] <= 1.0]
        print("%-16s точек %2d: %.3f..%.3f, медиана %.3f, геосреднее %.3f | >1.00: %d | 0.95-1.00: %d"
              % (name, len(s), r[0], r[-1], r[len(r) // 2], g, len(won), len(par)))
        if won:
            print("     ВЫИГРЫШ: " + ", ".join("%s/M%d=%.3f" % (x[0], x[1], x[2]) for x in won))
    imp = [(x[2] / x[3]) for x in agg if x[3] > 0]
    if imp:
        imp.sort()
        print("\nПРИРОСТ К ПРЕЖНЕМУ ЗАМЕРУ (то же дерево, тот же стенд): %.3f..%.3f, медиана %.3f"
              % (imp[0], imp[-1], imp[len(imp) // 2]))


if __name__ == "__main__":
    main()
