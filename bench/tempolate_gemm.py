#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ДРАЙВЕР ТЕМПОЛЯТОРА для плотного fp16-умножения: перечислить -> отсечь БЕЗ СБОРКИ ->
собрать -> ГЕЙТ КОРРЕКТНОСТИ -> замерить парными отношениями -> выбрать.

Вход конвейера -- inputs/naive_gemm_fp16.cu (он же знаменатель метрики "выход против входа").
Выход -- гиперформа + её исходник + паспорт (регистры, кадр стека, smem, обе планки).

  python3 bench/tempolate_gemm.py --shape 3840:15360 --m 2048 --top 48
"""

import argparse, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
# Закон отсекателя АРХИТЕКТУРНЫЙ (карта фрагмента m8n8k4) -> он на стороне ПЛАГИНА.
from tempo.plugins.sm70.gemm_bound import Hyperform, bound_tflops, occupancy  # noqa: E402

SKEL = os.path.join(ROOT, "tempo", "plugins", "sm70", "skeletons", "gemm_hmma884")
CUDA_HOME = os.environ.get("CUDA_HOME", "/home/alex/miniconda3/envs/cuda128")
NVCC = os.path.join(CUDA_HOME, "bin", "nvcc")


# ------------------------------------------------------------------ P3: перечисление
def enumerate_space(bn_ok, wide=False):
    BMs = [16, 32, 64, 96, 128, 160, 192, 256]
    BNs = [32, 64, 96, 128, 160, 192, 256]
    BKs = [16, 32, 64]
    out = []
    for BM in BMs:
        for BN in BNs:
            if not bn_ok(BN):
                continue
            for BK in BKs:
                for WM in (1, 2, 4, 8):
                    for WN in (1, 2, 4):
                        if WM * WN > 16 or WM * WN < 1:
                            continue
                        for ST in (2, 3, 4):
                            for GS in (1, 2):
                                for FP in (1, 2) if wide else (1, 2):
                                    for GR in (1, 4, 8, 16) if wide else (8,):
                                        for EP in (0, 1):
                                            nw = WM * WN
                                            minb = 2 if nw <= 4 else 1
                                            h = Hyperform(
                                                BM,
                                                BN,
                                                BK,
                                                WM,
                                                WN,
                                                ST,
                                                GS,
                                                FP,
                                                GR,
                                                EP,
                                                True,
                                                minb,
                                            )
                                            if h.legal() is None:
                                                out.append(h)
    return out


# ------------------------------------------------------------------ P4: отсечение
def prune(cands, M, N, K, top, per_geom=3):
    """Отсечение БЕЗ СБОРКИ + РАЗНООБРАЗИЕ ПО ГЕОМЕТРИИ.

    Ранжирование одной лишь границей вырождается: десятки вариантов конвейера при одной геометрии
    делят первое место, и вся квота уходит на них. Поэтому геометрии ранжируются границей, а внутри
    геометрии берётся не более per_geom вариантов конвейера (по возрастанию оценки регистров).
    Это ровно то место, где отсекатель обязан МОЛЧАТЬ: канала задержки в модели нет, глубину
    пересылки и предвыборку она не различает -- их различает секундомер.
    """
    geoms = {}
    for h in cands:
        if K % h.BK or N % h.BN:
            continue
        if K // h.BK < h.STAGES + h.GSTAGE:
            continue
        warps, ctas = occupancy(min(255, h.regs_estimate()), h.smem, h.threads)
        if ctas < 1:
            continue
        g = (h.BM, h.BN, h.BK, h.WM, h.WN)
        geoms.setdefault(g, []).append((bound_tflops(h, M, N, K), h, warps, ctas))
    ranked = []
    for g, lst in geoms.items():
        lst.sort(key=lambda x: (x[1].regs_estimate(), -x[3], x[1].smem))
        ranked.append((max(v[0] for v in lst), lst))
    ranked.sort(key=lambda x: -x[0])
    out, r = [], 0
    while len(out) < top and any(len(l) > 0 for _, l in ranked):
        progressed = False
        for _, lst in ranked:
            if r < len(lst) and len(out) < top:
                out.append(lst[r])
                progressed = True
        if not progressed:
            break
        r += 1
        if r >= per_geom:
            break
    return out[:top]


# ------------------------------------------------------------------ P5/P6: эмиссия и сборка
def build(cands, out_bin, extra=""):
    inc = os.path.join(SKEL, "configs.inc")
    with open(inc, "w") as f:
        f.write("// ПОРОЖДЕНО tempolate_gemm.py -- не править руками\n")
        for _, h, _, _ in cands:
            f.write(h.cfg_line() + "\n")
    cmd = "%s -O3 -std=c++17 -arch=sm_70 -lcublas -Xptxas -v -I%s -o %s %s %s" % (
        NVCC,
        SKEL,
        out_bin,
        os.path.join(SKEL, "harness.cu"),
        extra,
    )
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-4000:])
        print(p.stderr[-8000:])
        raise SystemExit("СБОРКА НЕ ПРОШЛА")
    return time.time() - t0, p.stderr


# ------------------------------------------------------------------ P7..P9
def run(out_bin, shape, ms, rounds, dev):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        os.path.join(CUDA_HOME, "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["CUDA_VISIBLE_DEVICES"] = str(dev)
    cmd = [
        out_bin,
        "--rounds",
        str(rounds),
        "--shape",
        "%d:%d" % shape,
        "--m",
        ",".join(map(str, ms)),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode:
        print(p.stdout[-4000:])
        print(p.stderr[-4000:])
        raise SystemExit("ПРОГОН НЕ ПРОШЁЛ")
    return p.stdout


def parse(txt):
    build_, base, cand, fails = {}, [], [], []
    for line in txt.splitlines():
        if line.startswith("BUILD "):
            d = json.loads(line[6:])
            build_[d["tag"]] = d
        elif line.startswith("BASE "):
            base.append(json.loads(line[5:]))
        elif line.startswith("CAND "):
            cand.append(json.loads(line[5:]))
        elif line.startswith("FAILGATE "):
            fails.append(json.loads(line[9:]))
    return build_, base, cand, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="3840:15360")
    ap.add_argument("--m", default="2048")
    ap.add_argument("--top", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--out", default="/tmp/tempo_gemm")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    K, N = map(int, a.shape.split(":"))
    ms = [int(x) for x in a.m.split(",")]
    os.makedirs(a.out, exist_ok=True)

    space = enumerate_space(lambda bn: N % bn == 0, wide=a.wide)
    top = prune(space, ms[0], N, K, a.top)
    print("пространство: %d законных, отобрано %d" % (len(space), len(top)))
    for b, h, w, c in top[:12]:
        print(
            "   %-38s nu_mio=%.3f  граница=%6.1f ТФЛОП/с  рег~%3d  варпов/SM=%2d  блоков/SM=%d  smem=%d"
            % (h.tag(), h.nu_mio(), b, h.regs_estimate(), w, c, h.smem)
        )

    binp = os.path.join(a.out, "harness")
    dt, log = build(top, binp)
    print("сборка: %.1f с" % dt)
    txt = run(binp, (K, N), ms, a.rounds, a.dev)
    bmap, base, cand, fails = parse(txt)
    for f in fails:
        print("ГЕЙТ НЕ ПРОЙДЕН:", f)
    for b in base:
        print(
            "\ncuBLAS  K=%d N=%d M=%d: %.4f мс = %.2f ТФЛОП/с (сверка эталона третьим путём rel=%.1e)"
            % (
                b["K"],
                b["N"],
                b["M"],
                b["cublas_ms"],
                b["cublas_tflops"],
                b["cublas_spot_rel"],
            )
        )
        rows = [
            c
            for c in cand
            if c["M"] == b["M"] and c["N"] == b["N"] and c["K"] == b["K"]
        ]
        rows.sort(key=lambda c: -c["ratio_med"])
        print(
            "  %-38s %8s %8s %8s %7s %6s %5s %s"
            % ("гиперформа", "мс", "ТФЛОП/с", "xcuBLAS", "рег", "кадр", "smem", "relL2")
        )
        for c in rows[:24]:
            bi = bmap.get(c["tag"], {})
            print(
                "  %-38s %8.4f %8.2f %8.4f %7d %6d %5d %.1e"
                % (
                    c["tag"],
                    c["ms"],
                    c["tflops"],
                    c["ratio_med"],
                    bi.get("regs", -1),
                    bi.get("frame", -1),
                    bi.get("smem", -1),
                    c["rel"],
                )
            )
    if a.json:
        with open(a.json, "w") as f:
            json.dump(
                {
                    "build": bmap,
                    "base": base,
                    "cand": cand,
                    "fail": fails,
                    "space": len(space),
                    "picked": len(top),
                },
                f,
                indent=1,
            )
        print("\nсохранено:", a.json)


if __name__ == "__main__":
    main()
