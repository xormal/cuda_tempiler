#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""РАЗВЁРТКА СТЕНДА С ДИСЦИПЛИНОЙ ЗАМЕРА.

Один ЗАПУСК probe = один РАУНД: варианты идут внутри раунда по таблице, то есть чередуются.
Раундов много, по каждому варианту берётся МЕДИАНА по раундам.  Вокруг КАЖДОГО раунда
записывается состояние карты 1: частота SM, мощность, число ЧУЖИХ процессов.  Точка, снятая
при чужом процессе, помечается -- кривая с изломом это ровно тот случай, где посторонняя
нагрузка подделывает признак.

ЗАПУСК
    python3 tools/sweep.py --bin build/reg/probe_r64 --warps 8 --rounds 7 --out data/x.json
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

CARD = "1"  # ТОЛЬКО карта 1: 0 занята соседом, 2 и 3 -- боевой сервер


def gpu_state():
    q = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            CARD,
            "--query-gpu=clocks.sm,power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        clk, pw, mem = [x.strip() for x in q.split(",")]
    except ValueError:
        clk, pw, mem = "?", "?", "?"
    apps = subprocess.run(
        ["nvidia-smi", "-i", CARD, "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    nproc = len([x for x in apps.split("\n") if x.strip()])
    return {"clk": clk, "power": pw, "mem": mem, "nproc": nproc}


def run_round(binpath, warps, iters, blocks, only, exact):
    cmd = [
        binpath,
        "--warps",
        str(warps),
        "--iters",
        str(iters),
        "--reps",
        "1",
        "--blocks",
        str(blocks),
    ]
    if only:
        cmd += ["--only", only]
    if exact:
        cmd += ["--exact"]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = CARD
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = {}
    for ln in p.stdout.split("\n"):
        if "|" not in ln or ln.startswith("#") or ln.startswith("вариант"):
            continue
        left, right = ln.split("|")
        f = left.split()
        try:
            out[f[0]] = float(right.split()[0])
        except (IndexError, ValueError):
            continue
    return out, p.returncode, p.stderr.strip()[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--blocks", type=int, default=80)
    ap.add_argument("--only")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    per = {}
    states = []
    for r in range(a.rounds):
        s0 = gpu_state()
        vals, rc, err = run_round(a.bin, a.warps, a.iters, a.blocks, a.only, a.exact)
        s1 = gpu_state()
        states.append({"before": s0, "after": s1, "rc": rc, "err": err})
        if rc != 0 and not vals:
            if not a.quiet:
                print(
                    "РАУНД %d: ОТКАЗ ЗАПУСКА rc=%d %s" % (r, rc, err), file=sys.stderr
                )
            continue
        for k, v in vals.items():
            per.setdefault(k, []).append(v)
    res = {}
    for k, v in per.items():
        res[k] = {
            "median": statistics.median(v),
            "min": min(v),
            "max": max(v),
            "n": len(v),
        }
    nproc_max = (
        max(
            [s["before"]["nproc"] for s in states]
            + [s["after"]["nproc"] for s in states]
        )
        if states
        else -1
    )
    meta = {
        "bin": a.bin,
        "warps": a.warps,
        "iters": a.iters,
        "blocks": a.blocks,
        "rounds": a.rounds,
        "states": states,
        "nproc_max": nproc_max,
        "clk": [s["before"]["clk"] for s in states],
        "power": [s["before"]["power"] for s in states],
    }
    if a.out:
        json.dump(
            {"meta": meta, "data": res}, open(a.out, "w"), indent=1, ensure_ascii=False
        )
    if not a.quiet:
        print(
            "# %s варпов=%d раундов=%d чужих процессов max=%d частоты=%s"
            % (
                os.path.basename(a.bin),
                a.warps,
                a.rounds,
                nproc_max,
                ",".join(meta["clk"][:3]),
            )
        )
        for k in sorted(res):
            d = res[k]
            print(
                "%-16s %10.3f  (разброс %.2f%%, n=%d)"
                % (
                    k,
                    d["median"],
                    100.0 * (d["max"] - d["min"]) / max(d["median"], 1e-9),
                    d["n"],
                )
            )


if __name__ == "__main__":
    main()
