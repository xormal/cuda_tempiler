#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЧЕСТНЫЙ ЭТАЛОН линейной части Gemma-4-12B: ЧЕТЫРЕ отсчёта, которые нельзя путать между собой.

  (1) cuBLAS-fp16 через torch F.linear -- ЖИВОЙ СОПЕРНИК и ровно тот путь, каким идёт боевая
      линейная часть. Абсолютная планка продукта: её и обязан перевести через 1.0 темполятор.
  (2) НАИВНЫЙ вход (../inputs/naive_gemm_fp16.cu, nvcc -O3) -- нижняя отметка, от которой считается
      собственная метрика темполятора «во сколько раз выход обгоняет вход».
  (3) наш нынешний W8A16 (боевое дерево: w8a16_gemv.cu / w8a16_hmma.cuh) -- планка для int8-пути.
      cuBLAS для int8 планкой НЕ является: пути int8 на тензорных ядрах sm_70 у него нет вовсе
      (IMMA появилась с Turing), поэтому «обогнали cuBLAS в int8» ничего не доказывает.
  (4) рукописный мейнлуп Volta -- проверка заявки «наш плотный fp16-GEMM = 0.67-0.82x cuBLAS»
      НА БОЕВЫХ ФОРМАХ (заявка получена на квадратах 4096^3 и на боевые формы не переносится
      автоматически). Считает C++-часть, gemm_baseline.cu.

ДИСЦИПЛИНА ЗАМЕРА, соблюдённая буквально:
  * гейт КОРРЕКТНОСТИ раньше секундомера -- relL2 против fp32-эталона, а для W8A16 против эталона,
    построенного ДРУГИМ маршрутом (распаковка в плотный тензор), чтобы общее неверное чтение
    раскладки не прошло обе проверки;
  * парные отношения, чередование плеч ВНУТРИ раунда, медиана ОТНОШЕНИЙ (не отношение медиан);
  * частота снимается ВО ВРЕМЯ работающего цикла через NVML (nvidia-smi между замерами показывает
    ПРОСТОЙ и занижает частоту втрое); доли от пика считаются от РЕАЛЬНОЙ частоты;
  * чужие процессы на карте считаются до и после, СВОЙ pid исключён.

Запуск:
  CUDA_HOME=/home/alex/miniconda3/envs/cuda128 \
  python3 bench/gemm_baseline.py --dev 0
"""

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FA2 = os.environ.get("TEMPO_HOST_TREE", "")
GEMM_INC = os.path.join(FA2, "fa2_src/fmha_kernel/gemm")

# Боевые формы Gemma-4-12B: H=3840, I=15360, 16 голов, 8 KV, d=256, 48 слоёв.
# Роль -> (K = вход, N = выход). k и v тождественны по форме, gate и up тождественны -- меряется по
# одной ФОРМЕ, а не по одному имени, иначе одно и то же число попадает в таблицу дважды.
PROJ = [
    ("q", 3840, 4096),
    ("k,v", 3840, 2048),
    ("o", 4096, 3840),
    ("gate,up", 3840, 15360),
    ("down", 15360, 3840),
]
MS = [1, 8, 32, 128, 512, 2048, 8192]

# Паспортный пик тензорных ядер V100-SXM2: 640 ТЯ * 2 * 64 ФЛОП/такт * f.
# При 1530 МГц = 125.3 ТФЛОП/с. ВНИМАНИЕ: «пик fp16 = 52.2 ТФЛОП/с», встречающийся в наших же
# документах, -- это колонка DP4A (целочисленный тракт), а не fp16. Опровергнуто замером
# (reports/E_VERDICT_int8_floor.md): int8, поданный как fp16, идёт ТОЙ ЖЕ инструкцией HMMA.884 и
# даёт ровно ту же скорость, отношение 1.00 на всех K.
PEAK_PER_MHZ = 640 * 2 * 64 * 1e6 / 1e12  # ТФЛОП/с на МГц


# --------------------------------------------------------------------------- NVML: частота ПОД НАГРУЗКОЙ
class Nvml:
    def __init__(self, dev):
        self.ok = False
        try:
            self.lib = ctypes.CDLL("libnvidia-ml.so.1")
            if self.lib.nvmlInit_v2() != 0:
                return
            self.h = ctypes.c_void_p()
            if self.lib.nvmlDeviceGetHandleByIndex_v2(dev, ctypes.byref(self.h)) != 0:
                return
            self.c = ctypes.c_uint()
            self.ok = True
        except Exception:
            pass

    def sm_mhz(self):
        if not self.ok:
            return None
        if self.lib.nvmlDeviceGetClockInfo(self.h, 1, ctypes.byref(self.c)) != 0:
            return None
        return float(self.c.value)


NV = None
CLOCKS = []  # (время, МГц) -- поток из ОТДЕЛЬНОЙ нити, ТОЛЬКО пока карта считает
BUSY = threading.Event()  # поднят на время работающего секундомера


class ClockSampler(threading.Thread):
    """Частота опрашивается ОТДЕЛЬНОЙ нитью, непрерывно, всё время прогона.

    ДВА СПОСОБА ПОЛУЧИТЬ ЗДЕСЬ 1530 ВМЕСТО УСТАНОВИВШИХСЯ 1300, оба испробованы и оба неверны:
      * опрос ИЗ ОСНОВНОЙ нити сразу после постановки цикла в очередь -- просадка по мощности
        наступает не мгновенно, и отсчёт попадает в ещё не просевшее окно;
      * опрос ВСЁ ВРЕМЯ ПРОГОНА -- медиану разбавляют ПАУЗЫ между плечами, где карта простаивает
        и возвращается на 1530 (в прогоне 1.66 млн отсчётов медиана вышла 1530 при минимуме 390).
    Поэтому пишется только промежуток, когда секундомер В ПОЛЁТЕ (флаг BUSY). Опорный замер --
    bench/data_power.cu: 1297-1387 МГц при 291 Вт на тяжёлых формах против 1530 на константных
    данных. Разница идёт прямо в знаменатель доли от пика."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop_ev = threading.Event()

    def run(self):
        while not self.stop_ev.is_set():
            if not BUSY.wait(0.05):
                continue  # карта простаивает между плечами -- эти отсчёты НЕ наши
            c = NV.sm_mhz() if NV else None
            if c:
                CLOCKS.append((time.time(), c))

    def stop(self):
        self.stop_ev.set()
        self.join(timeout=3)


def clocks_between(t0, t1):
    return [c for t, c in CLOCKS if t0 <= t <= t1]


def smi_procs(dev):
    """Чужие процессы на карте. СВОЙ pid исключается -- иначе харнесс всегда рапортует сам себя."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(dev),
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    mine = str(os.getpid())
    lines = [
        l for l in out.splitlines() if l.strip() and l.split(",")[0].strip() != mine
    ]
    return len(lines), "; ".join(lines)


def gpu_state(dev):
    sm, pw, ut = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(dev),
                "--query-gpu=clocks.sm,power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .split(", ")
    )
    n, procs = smi_procs(dev)
    return {
        "sm_mhz": float(sm),
        "power_w": float(pw),
        "util": float(ut),
        "n_foreign_procs": n,
        "foreign_procs": procs,
    }


# --------------------------------------------------------------------------- секундомер
def bench(fn, target_ms=400.0, min_iter=3, max_iter=2000):
    """Один замер. Окно 400 мс, а не 120: просадка по мощности наступает НЕ мгновенно, и на коротком
    окне карта успевает отработать на ещё не просевшей частоте. Боевой префилл держит линейную часть
    занятой секундами, то есть УСТАНОВИВШИЙСЯ режим -- он и должен мериться."""
    import torch

    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(min_iter):
        fn()
    e1.record()
    torch.cuda.synchronize()
    t = e0.elapsed_time(e1) / min_iter
    it = max(min_iter, min(max_iter, int(target_ms / max(t, 1e-4))))
    for _ in range(
        max(1, it // 3)
    ):  # прогрев ДО стабилизации мощности, вне секундомера
        fn()
    torch.cuda.synchronize()
    e0.record()
    for _ in range(it):
        fn()
    e1.record()
    BUSY.set()  # частота пишется ТОЛЬКО пока цикл в полёте:
    torch.cuda.synchronize()  # иначе медиану разбавляют паузы между плечами
    BUSY.clear()  # и она выходит 1530 при установившихся 1300
    return e0.elapsed_time(e1) / it


def median(v):
    return statistics.median(v) if v else None


def stat(v):
    if not v:
        return None
    return {"n": len(v), "min": min(v), "med": statistics.median(v), "max": max(v)}


# --------------------------------------------------------------------------- W8A16
def make_gptq_int8(K, N, G, dev, seed):
    """Веса GPTQ bits=8: qweight [K/4,N] int32, qzeros [K/G,N/4] int32, scales [K/G,N] fp16.
    Возвращает ещё и ПЛОТНЫЙ fp16-эквивалент -- эталон значений, построенный ДРУГИМ маршрутом."""
    import torch

    torch.manual_seed(seed)
    qw = torch.randint(-(2**31), 2**31 - 1, (K // 4, N), device=dev, dtype=torch.int32)
    qz = torch.randint(
        -(2**31), 2**31 - 1, (K // G, N // 4), device=dev, dtype=torch.int32
    )
    sc = torch.rand(K // G, N, device=dev, dtype=torch.float16) * 0.02 + 0.001
    sh = torch.arange(0, 32, 8, device=dev)
    q = ((qw.unsqueeze(1) >> sh.view(1, 4, 1)) & 0xFF).reshape(K, N).float()
    z = ((qz.unsqueeze(2) >> sh.view(1, 1, 4)) & 0xFF).reshape(K // G, N).float()
    gidx = torch.arange(K, device=dev) // G
    W = ((q - z[gidx]) * sc.float()[gidx]).half()  # [K,N] -- у GPTQ вес k-мажорный
    del q, z, gidx
    torch.cuda.empty_cache()
    return qw, qz, sc, W


# --------------------------------------------------------------------------- C++ плечи
def build_cxx(out_bin):
    cuda_home = os.environ.get("CUDA_HOME", "/opt/conda")
    cmd = [
        os.path.join(cuda_home, "bin/nvcc"),
        "-O3",
        "-std=c++17",
        "-arch=sm_70",
        "-Wno-deprecated-gpu-targets",
        "-lcublas",
        "-I",
        GEMM_INC,
        "-o",
        out_bin,
        os.path.join(HERE, "gemm_baseline.cu"),
    ]
    subprocess.check_call(cmd)
    return out_bin


def run_cxx(binpath, dev, rounds, ms_list, quiet=False):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        os.environ.get("CUDA_HOME", "/opt/conda")
        + "/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    cmd = [
        binpath,
        "--dev",
        str(dev),
        "--rounds",
        str(rounds),
        "--m",
        ",".join(str(m) for m in ms_list),
        "--naive-budget",
        "6000",
    ]
    for _role, K, N in PROJ:
        cmd += ["--shape", f"{K}:{N}"]
    # частота снимается фоном: C++-часть занимает карту почти непрерывно
    clocks, stop = [], threading.Event()

    def watch():
        while not stop.is_set():
            c = NV.sm_mhz() if NV else None
            if c:
                clocks.append(c)
            stop.wait(0.25)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    stop.set()
    th.join(timeout=3)
    rows, skips = [], []
    for line in p.stdout.splitlines():
        if line.startswith("JSON "):
            rows.append(json.loads(line[5:]))
        elif line.startswith("SKIP"):
            skips.append(line)
        elif not quiet:
            print("   [cxx]", line)
    if p.returncode != 0:
        print("   [cxx] RC", p.returncode, p.stderr[-2000:])
    return rows, skips, [c for c in clocks if c > 200]


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-w8a16", action="store_true")
    ap.add_argument("--skip-cxx", action="store_true")
    ap.add_argument("--ms", default="")
    args = ap.parse_args()
    ms_list = [int(x) for x in args.ms.split(",")] if args.ms else MS

    global NV
    NV = Nvml(args.dev)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.dev))
    import torch

    dev = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = False
    name = torch.cuda.get_device_name(0)

    st_before = gpu_state(args.dev)
    print(f"# карта {args.dev} = {name}; NVML {'есть' if NV.ok else 'НЕТ'}")
    print(
        f"# ДО:  {st_before['sm_mhz']:.0f} МГц, {st_before['power_w']:.0f} Вт, "
        f"чужих процессов {st_before['n_foreign_procs']} {st_before['foreign_procs']}"
    )
    t_start = time.time()
    sampler = ClockSampler()
    sampler.start()

    # ---------- C++ плечи: cuBLAS(C), наивный вход, рукописный мейнлуп -----------------------
    cxx_rows, cxx_skips, cxx_clocks = [], [], []
    if not args.skip_cxx:
        binp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "tempo_gemm_baseline")
        print("# сборка C++-части ...")
        build_cxx(binp)
        print("# C++ плечи (cuBLAS / наивный вход / рукописный мейнлуп) ...")
        cxx_rows, cxx_skips, cxx_clocks = run_cxx(binp, args.dev, args.rounds, ms_list)
        for r in cxx_rows:
            print(
                f"  cxx K{r['K']:6d} N{r['N']:6d} M{r['M']:5d}  cuBLAS {r['cublas_tflops']:7.2f} "
                f"| мейнлуп {r['volta_tflops']:7.2f} x{r['volta_ratio_med']:.3f} ({r['volta_cfg']}) "
                f"| наивный {r['naive_tflops']:6.3f} x{r['naive_ratio_med']:.4f}"
            )

    # ---------- torch-плечи: F.linear (боевой путь) и W8A16 ----------------------------------
    rows = []
    for role, K, N in PROJ:
        w8 = None
        if not args.skip_w8a16:
            try:
                if FA2 not in sys.path:
                    sys.path.insert(0, FA2)
                import fa2_sm70  # noqa: F401

                ext = fa2_sm70._ext.w8a16_ext()
                qw, qz, sc, W = make_gptq_int8(K, N, 128, dev, K * 131 + N)
                w8 = (ext, qw, qz, sc, W)
            except Exception as e:
                print(f"# W8A16 недоступен: {type(e).__name__}: {e}")
                args.skip_w8a16 = True

        for M in ms_list:
            x = torch.randn(M, K, device=dev, dtype=torch.float16) * 0.05
            Wl = (
                torch.randn(N, K, device=dev, dtype=torch.float16) * 0.05
            )  # раскладка F.linear
            flop = 2.0 * M * N * K

            # --- гейт корректности РАНЬШЕ секундомера --------------------------------------
            y_cb = torch.nn.functional.linear(x, Wl)
            ref = x.float() @ Wl.float().t()
            rel_cb = ((y_cb.float() - ref).norm() / ref.norm()).item()
            del ref, y_cb
            torch.cuda.empty_cache()

            rec = {
                "role": role,
                "K": K,
                "N": N,
                "M": M,
                "flop": flop,
                "cublas_rel_vs_fp32": rel_cb,
            }

            w8_arms = {}
            if w8 is not None:
                ext, qw, qz, sc, W = w8
                ref8 = x.float() @ W.float()
                for tag, fn in (
                    ("w8a16_gemv", lambda: ext.w8a16_gemv(x, qw, qz, sc, 128)),
                    ("w8a16_hmma", lambda: ext.w8a16_gemm_hmma(x, qw, qz, sc, 128)),
                ):
                    try:
                        y = fn()
                        rel = ((y.float() - ref8).norm() / ref8.norm()).item()
                        del y
                        if rel < 2e-2:
                            w8_arms[tag] = (fn, rel)
                        else:
                            rec[tag + "_rel_FAIL"] = rel
                    except Exception as e:
                        rec[tag + "_err"] = f"{type(e).__name__}: {e}"
                del ref8
                torch.cuda.empty_cache()

            # --- парные раунды: плечи ЧЕРЕДУЮТСЯ ВНУТРИ раунда ----------------------------
            cb_ms, ratios = [], {k: [] for k in w8_arms}
            t_mark = time.time()  # частота ЭТОЙ формы, а не средняя по всему прогону:
            #                             медиана по всем отсчётам врёт, потому что львиная доля
            #                             отсчётов приходит с лёгких форм, где просадки нет вовсе
            for _ in range(args.rounds):
                c = bench(lambda: torch.nn.functional.linear(x, Wl))
                cb_ms.append(c)
                for tag, (fn, _rel) in w8_arms.items():
                    t = bench(fn)
                    ratios[tag].append(c / t)  # >1 = наше плечо БЫСТРЕЕ cuBLAS
            cb = median(cb_ms)
            rec["cublas_ms"] = cb
            rec["cublas_tflops"] = flop / (cb * 1e-3) / 1e12
            rec["clock"] = stat(clocks_between(t_mark, time.time()))
            if rec["clock"]:
                rec["peak_at_own_clock"] = PEAK_PER_MHZ * rec["clock"]["med"]
                rec["pct_of_peak"] = (
                    100 * rec["cublas_tflops"] / rec["peak_at_own_clock"]
                )
            for tag, (fn, rel) in w8_arms.items():
                r = median(ratios[tag])
                rec[tag + "_rel"] = rel
                rec[tag + "_ms"] = cb / r
                rec[tag + "_tflops"] = flop / ((cb / r) * 1e-3) / 1e12
                rec[tag + "_ratio_vs_cublas"] = r
            rows.append(rec)
            print(
                f"  {role:9s} K{K:6d} N{N:6d} M{M:5d}  cuBLAS {cb:9.4f} мс "
                f"{rec['cublas_tflops']:7.2f} ТФЛОП/с  relFP32 {rel_cb:.1e}  "
                + "  ".join(f"{t}:x{rec[t + '_ratio_vs_cublas']:.3f}" for t in w8_arms)
            )
            del x, Wl
            torch.cuda.empty_cache()
        if w8 is not None:
            del w8
            torch.cuda.empty_cache()

    sampler.stop()
    st_after = gpu_state(args.dev)
    clk_torch, clk_cxx = stat([c for _t, c in CLOCKS]), stat(cxx_clocks)
    all_clk = stat([c for _t, c in CLOCKS] + cxx_clocks)
    print(
        f"# ПОСЛЕ: {st_after['sm_mhz']:.0f} МГц, {st_after['power_w']:.0f} Вт, "
        f"чужих процессов {st_after['n_foreign_procs']} {st_after['foreign_procs']}"
    )
    if all_clk:
        print(
            f"# частота ПОД НАГРУЗКОЙ: мин {all_clk['min']:.0f} / медиана {all_clk['med']:.0f} / "
            f"макс {all_clk['max']:.0f} МГц ({all_clk['n']} отсчётов)"
        )
    fmed = all_clk["med"] if all_clk else 1530.0
    out = {
        "gpu": name,
        "dev": args.dev,
        "rounds": args.rounds,
        "seconds": round(time.time() - t_start, 1),
        "state_before": st_before,
        "state_after": st_after,
        "clock_under_load": all_clk,
        "clock_torch_arms": clk_torch,
        "clock_cxx_arms": clk_cxx,
        "peak_tflops_1530": PEAK_PER_MHZ * 1530,
        "peak_tflops_at_measured_clock": PEAK_PER_MHZ * fmed,
        "valid_no_neighbour": st_before["n_foreign_procs"] == 0
        and st_after["n_foreign_procs"] == 0,
        "rows_torch": rows,
        "rows_cxx": cxx_rows,
        "cxx_skips": cxx_skips,
    }
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, f"gemm_baseline{args.tag}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"# записано: {p}   ({out['seconds']:.0f} с)")


if __name__ == "__main__":
    main()
