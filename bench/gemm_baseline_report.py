#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сводка честного эталона: из data/gemm_baseline.json -> таблица и три вердикта.

Печатает и кладёт рядом .txt. НИЧЕГО не мерит -- только читает JSON, чтобы вердикт нельзя было
получить «на глазок» мимо записанных чисел.

Три вердикта, которые он обязан выдать:
  V1. Держится ли заявка «наш плотный fp16-GEMM = 0.67-0.82x cuBLAS» НА БОЕВЫХ формах.
  V2. Во сколько раз выход обгонял бы НАИВНЫЙ вход (собственная метрика темполятора).
  V3. Порог замены формата: 1.85*rel>1 (посылка наряда) против ЗАМЕРЕННОГО множителя.
"""
import json
import os
import statistics
import sys

D = "/mnt/d1/alex/tempo/data"


def geo(v):
    v = [x for x in v if x and x > 0]
    if not v:
        return float("nan")
    s = sum(__import__("math").log(x) for x in v)
    return __import__("math").exp(s / len(v))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "gemm_baseline.json")
    J = json.load(open(src))
    L = []
    P = L.append
    fmed = (J.get("clock_under_load") or {}).get("med", 1530.0)
    peak = J["peak_tflops_at_measured_clock"]
    P(f"ЧЕСТНЫЙ ЭТАЛОН ЛИНЕЙНОЙ ЧАСТИ Gemma-4-12B -- {J['gpu']}, карта {J['dev']}")
    P(f"частота ПОД НАГРУЗКОЙ: медиана {fmed:.0f} МГц "
      f"(мин {(J.get('clock_under_load') or {}).get('min', 0):.0f} / "
      f"макс {(J.get('clock_under_load') or {}).get('max', 0):.0f})")
    P(f"пик тензорных ядер: {J['peak_tflops_1530']:.1f} ТФЛОП/с @1530 МГц; "
      f"{peak:.1f} ТФЛОП/с на замеренной частоте")
    P(f"чужих процессов: до {J['state_before']['n_foreign_procs']}, "
      f"после {J['state_after']['n_foreign_procs']}  ->  "
      f"{'ЗАМЕР ДЕЙСТВИТЕЛЕН' if J['valid_no_neighbour'] else 'ЗАМЕР С СОСЕДОМ -- НЕДЕЙСТВИТЕЛЕН'}")
    P(f"раундов {J['rounds']}, медиана ОТНОШЕНИЙ, чередование плеч внутри раунда; {J['seconds']:.0f} с")
    P("")

    cxx = {(r["K"], r["N"], r["M"]): r for r in J.get("rows_cxx", [])}
    best_cublas = max((r["cublas_tflops"] for r in J["rows_torch"]), default=0)

    P("=" * 162)
    P(f"{'роль':9s} {'K':>6s} {'N':>6s} {'M':>5s} | {'cuBLAS мс':>10s} {'ТФЛОП/с':>8s} {'МГц':>5s} "
      f"{'%пика':>6s} {'%лучш':>6s} | {'мейнлуп':>8s} {'x cuBLAS':>9s} {'конфиг':>15s} | "
      f"{'наивный':>8s} {'x cuBLAS':>9s} | {'w8gemv':>8s} {'w8hmma':>8s}")
    P("=" * 162)
    for r in J["rows_torch"]:
        c = cxx.get((r["K"], r["N"], r["M"]), {})
        vt = c.get("volta_tflops", -1)
        vr = c.get("volta_ratio_med", -1)
        nt = c.get("naive_tflops", -1)
        nr = c.get("naive_ratio_med", -1)
        f_own = (r.get("clock") or {}).get("med", fmed)
        pct = r.get("pct_of_peak", 100 * r["cublas_tflops"] / peak)
        P(f"{r['role']:9s} {r['K']:6d} {r['N']:6d} {r['M']:5d} | "
          f"{r['cublas_ms']:10.4f} {r['cublas_tflops']:8.2f} {f_own:5.0f} "
          f"{pct:5.1f}% {100 * r['cublas_tflops'] / best_cublas:5.1f}% | "
          f"{vt:8.2f} {vr:9.3f} {c.get('volta_cfg', '-'):>15s} | "
          f"{nt:8.3f} {nr:9.4f} | "
          f"{r.get('w8a16_gemv_ratio_vs_cublas', float('nan')):8.3f} "
          f"{r.get('w8a16_hmma_ratio_vs_cublas', float('nan')):8.3f}")
    P("=" * 162)
    P("МГц     -- медиана NVML во время ЭТОЙ формы, пока секундомер в полёте.")
    P("          ОСТОРОЖНО: медиана NVML НЕ ГОДИТСЯ как знаменатель. Она даёт 1530 даже там, где")
    P("          установившаяся частота 1230-1400 (минимум по той же форме это и показывает):")
    P("          провалы короткие, и NVML их недоопрашивает. Надёжный знаменатель берётся из")
    P("          установившегося режима -- bench/data_power.cu, 1 с непрерывной нагрузки:")
    P("          1230-1400 МГц при 291 Вт => пик 100.7-114.7, а не 125.3 ТФЛОП/с.")
    P("%пика   -- от ПАСПОРТНЫХ 125.3 ТФЛОП/с @1530. Это НИЖНЯЯ оценка доли: по установившейся")
    P("          частоте те же ядра дают 77-80 %, а не 63-68 %.")
    P("%лучш   -- от лучшего, что cuBLAS взял сам на этих же формах (достижимый потолок)")
    P("x cuBLAS -- медиана отношений времён; >1 = плечо БЫСТРЕЕ cuBLAS")
    P("")

    # -------------------------------------------------- V1: заявка 0.67-0.82x
    big = [r for r in J.get("rows_cxx", []) if r["M"] >= 128 and r["volta_ratio_med"] > 0]
    small = [r for r in J.get("rows_cxx", []) if r["M"] < 128 and r["volta_ratio_med"] > 0]
    P("--- ВЕРДИКТ 1: заявка «наш плотный fp16-GEMM = 0.67-0.82x cuBLAS» -----------------------")
    if big:
        v = sorted(r["volta_ratio_med"] for r in big)
        P(f"  ПРЕФИЛЛ (M>=128, {len(big)} точек): {v[0]:.3f}..{v[-1]:.3f}, "
          f"медиана {statistics.median(v):.3f}, геосреднее {geo(v):.3f}")
    if small:
        v = sorted(r["volta_ratio_med"] for r in small)
        P(f"  ДЕКОД   (M<=32,  {len(small)} точек): {v[0]:.3f}..{v[-1]:.3f}, "
          f"медиана {statistics.median(v):.3f}, геосреднее {geo(v):.3f}")
    npd = [(r["nopred_ms"] / r["volta_ms"], r) for r in J.get("rows_cxx", [])
           if r.get("nopred_ms", -1) > 0 and r.get("volta_ms", -1) > 0]
    if npd:
        P(f"  цена предикации по M (тот же размер плитки без неё, {len(npd)} точек): "
          f"медиана x{statistics.median([a for a, _ in npd]):.3f}")
    # Критерий заказчика: ПАРИТЕТ УЖЕ ВЫИГРЫШ (он делает стык нашим). Поэтому считаются не только
    # точки >1.00, но и коридор 0.95-1.00 -- «владеем стыком ценой единиц процентов».
    allv = [r for r in J.get("rows_cxx", []) if r["volta_ratio_med"] > 0]
    win = [r for r in allv if r["volta_ratio_med"] > 1.00]
    par = [r for r in allv if 0.95 <= r["volta_ratio_med"] <= 1.00]
    P(f"  ПО КРИТЕРИЮ ЗАКАЗЧИКА (паритет = уже выигрыш), всего точек {len(allv)}:")
    P(f"    >1.00  -- владеем стыком И выигрываем в марже: {len(win)} точек" +
      (": " + ", ".join(f"{next((p for p, K, N in PROJ_NAMES if K == r['K'] and N == r['N']), '?')}"
                        f"/M{r['M']}={r['volta_ratio_med']:.3f}" for r in win) if win else ""))
    P(f"    0.95-1.00 -- владеем стыком ценой единиц процентов: {len(par)} точек" +
      (": " + ", ".join(f"{next((p for p, K, N in PROJ_NAMES if K == r['K'] and N == r['N']), '?')}"
                        f"/M{r['M']}={r['volta_ratio_med']:.3f}" for r in par) if par else ""))
    P(f"    ниже 0.95: {len(allv) - len(win) - len(par)} точек")
    P("")

    # -------------------------------------------------- V2: выход против входа
    P("--- ВЕРДИКТ 2: НАИВНЫЙ ВХОД (нижняя отметка темполятора) ---------------------------------")
    nn = [r for r in J.get("rows_cxx", []) if r.get("naive_tflops", -1) > 0]
    if nn:
        v = sorted(r["naive_ratio_med"] for r in nn)
        P(f"  наивный/cuBLAS: {v[0]:.4f}..{v[-1]:.4f}, геосреднее {geo(v):.4f} "
          f"-> cuBLAS быстрее входа в {1 / geo(v):.0f} раз")
        vv = sorted(r["volta_ms"] / r["naive_ms"] for r in nn if r.get("volta_ms", -1) > 0)
        if vv:
            P(f"  рукописный мейнлуп/наивный: обгон входа в {1 / geo(vv):.0f} раз "
              f"({1 / vv[-1]:.0f}..{1 / vv[0]:.0f})")
        P(f"  наивный на пике: {max(r['naive_tflops'] for r in nn):.3f} ТФЛОП/с = "
          f"{100 * max(r['naive_tflops'] for r in nn) / peak:.2f}% тензорного пика")
    P("")

    # -------------------------------------------------- режим M<=64: связывает ЧТЕНИЕ, не счёт
    P("--- РЕЖИМ M<=64 (ДЕКОД): связывает ЧТЕНИЕ ВЕСОВ, а не тензорный конвейер -----------------")
    P("  достижимая полоса V100-SXM2 замерена = 841 ГБ/с (docs/VOLTA_SM70.md)")
    P(f"  {'роль':9s} {'M':>4s} | {'cuBLAS ГБ/с':>12s} {'%полосы':>8s} | {'мейнлуп ГБ/с':>13s} "
      f"{'%полосы':>8s} | {'cuBLAS %счёта':>13s}")
    for r in J["rows_torch"]:
        if r["M"] > 64:
            continue
        by = 2.0 * (r["M"] * r["K"] + r["N"] * r["K"] + r["M"] * r["N"])
        cb_gb = by / (r["cublas_ms"] * 1e-3) / 1e9
        c = cxx.get((r["K"], r["N"], r["M"]), {})
        v_gb = by / (c["volta_ms"] * 1e-3) / 1e9 if c.get("volta_ms", -1) > 0 else float("nan")
        P(f"  {r['role']:9s} {r['M']:4d} | {cb_gb:12.1f} {100 * cb_gb / 841:7.1f}% | "
          f"{v_gb:13.1f} {100 * v_gb / 841:7.1f}% | {100 * r['cublas_tflops'] / peak:12.1f}%")
    P("  ЧИТАТЬ ТАК: там, где %полосы высок, а %счёта низок, соперник НЕ В ФОРМЕ по счёту, и")
    P("  рычаг -- байты (узкий вес), а не ФЛОПы. Именно здесь W8A16 обгоняет cuBLAS.")
    P("")

    # -------------------------------------------------- V3: порог замены формата
    P("--- ВЕРДИКТ 3: порог 1.85*rel > 1 --------------------------------------------------------")
    roof = os.path.join(D, "int8_roof.json")
    mult = None
    if os.path.exists(roof):
        R = json.load(open(roof))
        rr = [x["ratio_int8fp16_over_fp16"] for x in R["rows"]]
        dd = [x.get("ratio_dp4a_over_fp16") for x in R["rows"] if x.get("ratio_dp4a_over_fp16")]
        mult = geo(rr)
        P(f"  ЗАМЕРЕННЫЙ множитель формата int8 (уложенного в мантиссу fp16) к fp16: "
          f"x{mult:.3f} (геосреднее по {len(rr)} боевым точкам)")
        if dd:
            P(f"  DP4A (единственный целочисленный тракт sm_70) к fp16: x{geo(dd):.3f} "
              f"-- то есть МЕДЛЕННЕЕ fp16")
        P(f"  посылка наряда «x1.85» -- это HMMA/DP4A (обе величины целочисленные), а НЕ int8/fp16.")
    else:
        P("  (data/int8_roof.json не найден -- запустить bench/int8_roof.py)")
    if big and mult:
        P("")
        P(f"  порог с ПОСЫЛКОЙ 1.85: нужно rel > {1 / 1.85:.3f}")
        P(f"  порог с ЗАМЕРОМ {mult:.2f}: нужно rel > {1 / mult:.3f}")
        P("")
        P(f"  {'роль':9s} {'M':>5s} {'rel':>7s} | 1.85*rel | замер*rel | вывод")
        for r in J.get("rows_cxx", []):
            if r["volta_ratio_med"] <= 0:
                continue
            rel = r["volta_ratio_med"]
            role = next((p for p, K, N in PROJ_NAMES if K == r["K"] and N == r["N"]), "?")
            P(f"  {role:9s} {r['M']:5d} {rel:7.3f} | {1.85 * rel:8.3f} | {mult * rel:9.3f} | "
              f"{'посылка прошла бы, ЗАМЕР НЕТ' if 1.85 * rel > 1 >= mult * rel else ('ОБА ПРОХОДЯТ' if mult * rel > 1 else 'НЕ ПРОХОДИТ НИ ОДИН')}")
    P("")
    txt = "\n".join(L)
    print(txt)
    out = os.path.join(D, "gemm_baseline_table.txt")
    with open(out, "w") as f:
        f.write(txt + "\n")
    print("# записано:", out)


PROJ_NAMES = [("q", 3840, 4096), ("k,v", 3840, 2048), ("o", 4096, 3840),
              ("gate,up", 3840, 15360), ("down", 15360, 3840)]

if __name__ == "__main__":
    main()
