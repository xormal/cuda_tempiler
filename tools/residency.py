# -*- coding: utf-8 -*-
"""КАЛЬКУЛЯТОР РЕЗИДЕНТНОСТИ: превращает ТРАФИК ПЛИТОК во ВРЕМЯ (или отказывается превращать).

ЗАЧЕМ. Сокращение трафика НЕ переходит во время автоматически. Замерено в этом проекте:
группировка соседних блоков запросов даёт 13.4x меньше чтений плиток KV, а префилл при d<=128
сидит в L2 на 97.7 % попаданий (HBM 3.26 % от пика) -- экономить нечего. Там же замерена лестница
плитки запросов при d=128, Sq=Sk=4096: BQ 32/64/128 -> 353.1 / 376.1 / 495.1 мкс, то есть трафик
падает вчетверо, а время РАСТЁТ в 1.41 раза. Аргумент "меньше трафика -> быстрее" законен только
там, где рабочее множество НЕ ВЛЕЗАЕТ в L2.

ЧТО СЧИТАЕТ. По (B, H, Hkv, Sq, Sk, d, BQ, BK, размер элемента, число SM):
  1. ВОЛНУ: сколько CTA резидентны одновременно, на сколько разных потоков KV они распадаются,
     сколько CTA приходится на поток (это и есть множитель переиспользования плитки);
  2. ОКНО -- рабочее множество KV, живое в один момент, и сравнение с L2 (6 МБ), с разделяемой
     (96 КБ) и с L1 (32 КБ при 96 КБ разделяемой);
  3. ТРАФИК на двух уровнях (обращения к L2 и промахи в HBM) и предсказанную долю попаданий в L2;
  4. ВРЕМЯ по каналам (HBM / L2 / тензорный конвейер) и ВЕРДИКТ: переходит ли сокращение трафика
     во время и во сколько раз максимум;
  5. ГРАНИЦУ по форме: при каком Sk (для каждого d) окно перестаёт влезать в L2.

МОДЕЛЬ ОКНА -- главное, и она НЕ "весь KV сразу". Резидентные CTA одного потока идут по ключам
СИНХРОННО (это замерено: "плотное внимание держит 97.7 % попаданий в L2 только потому, что 80
блоков идут по ключам синхронно"; при том же числе плиток согласованные маски дают x1.135, а
независимые -- x0.967, то есть УБЫТОК). Расхождение позиций внутри потока ограничено числом
резидентных CTA этого потока: когда блок запросов доигрывает, на его место встаёт следующий,
начинающий с нуля. Отсюда

    ОКНО = СУММА по потокам  min(резидентных CTA потока, число плиток ключей) * (2 * BK * d * esz)

(множитель 2 -- K и V: за один шаг по ключам трогаются обе плитки). Верхняя оценка окна --
весь KV резидентных потоков; нижняя (идеальная синхронность) -- по одной плитке на поток.
Печатаются все три.

ЧЕГО МОДЕЛЬ НЕ ЗНАЕТ -- раздел "НЕ РАЗОБРАНО" печатается ВСЕГДА. Неполный инструмент даёт не
"меньше данных", а ДРУГОЙ ОТВЕТ, с той же уверенностью.

ЗАПУСК:
    python3 tools/residency.py --selftest                 # три якоря + лестница BQ + декод
    python3 tools/residency.py --d 128 --Sq 4096 --Sk 4096 --B 1 --H 32 --Hkv 8 --causal
    python3 tools/residency.py --preset gemma4-global --Sk 16384
    python3 tools/residency.py --scan                      # граница по форме
"""

import argparse
import math
import sys
from collections import Counter

# ---------------------------------------------------------------------------------------------
# МАШИНА. Числа из solutions/fa2_sm70_cutlass_grade/docs/VOLTA_SM70.md.
# [замерено] -- на этой машине; [спец] -- из документации; [ОЦЕНКА] -- НЕ замерено, помечено везде.
# ---------------------------------------------------------------------------------------------
MACHINE = {
    "sm": 80,  # [спец] V100
    "l2": 6 * 1024**2,  # [спец] 6 МБ
    "smem_per_sm": 96 * 1024,  # [спец] из 128 КБ единого L1+smem
    "l1_per_sm": 32 * 1024,  # [спец] остаток единого кэша при 96 КБ разделяемой
    "regfile": 65536,  # [спец] 32-битных регистров на SM
    "warps_per_sm": 64,  # [спец]
    "hbm_read": 841e9,  # [замерено] sum() 256 МБ fp32 (не 900!)
    "hbm_rw": 819e9,  # [замерено] copy_ 256 МБ fp32
    "l2_bw": 2155e9,  # [ОЦЕНКА] пропускная L2 НЕ ЗАМЕРЕНА на этой машине
    "tensor_peak": 125.3e12,  # [спец] HMMA fp16/fp32-накопление
    "tensor_ach": 120.96e12,  # [замерено] рукописный мейнлуп = 96.8 % пика
    "clock": 1.53e9,  # [замерено] фиксированные частоты
}


# Отгруженная политика плиток -- прочитана из fa2_sm70/csrc/attn_fwd_cutlass.cu (run_fmha).
# Возвращает (BQ, BK).
def shipped_tile(d, Sq, Sk):
    if d <= 64:
        return 64, 64
    if d <= 128:
        return (64, 128) if Sk >= 16384 else (32, 128)
    if d <= 256:
        if Sk >= 8192 and Sq >= 128:
            return 128, 128
        if Sk >= 8192 and Sq >= 64:
            return 64, 128
        return 32, 128
    return 32, 128  # d<=512 и выше: <32,128,512>


# ЗАМЕРЕННЫЕ конфигурации (ncu, docs/SM70_KERNEL_PLAYBOOK.md §25.2): (BQ,BK,d) -> (smem Б, регистров).
# Число варпов в блоке = BQ/8 (замерено: BQ 32/64/128 -> блок 128/256/512 нитей).
MEASURED_CFG = {
    (32, 128, 128): (25 * 1024, 168),
    (64, 128, 128): (34 * 1024, 174),
    (128, 128, 128): (52 * 1024, 128),
}


def threads_of(BQ):
    """Замерено: BQ 32/64/128 -> блок 128/256/512 нитей (4/8/16 варпов)."""
    return max(128, BQ * 4)


def occupancy(BQ, BK, d, smem_override=None, regs_override=None):
    """CTA на SM. Возвращает (ctas, откуда, smem, regs)."""
    key = (BQ, BK, d)
    src = "замерено"
    if key in MEASURED_CFG and smem_override is None and regs_override is None:
        smem, regs = MEASURED_CFG[key]
    else:
        src = "ОЦЕНКА"
        # подгонка по трём замеренным точкам d=128: smem = 16 КБ + BQ*(2d + 32) Б
        smem = (
            smem_override
            if smem_override is not None
            else 16 * 1024 + BQ * (2 * d + 32)
        )
        regs = regs_override if regs_override is not None else 168
        if smem_override is not None and regs_override is not None:
            src = "задано"
    thr = threads_of(BQ)
    by_reg = MACHINE["regfile"] // (regs * thr) if regs * thr else 99
    by_smem = MACHINE["smem_per_sm"] // smem if smem else 99
    by_warp = MACHINE["warps_per_sm"] // (thr // 32)
    return max(1, min(by_reg, by_smem, by_warp)), src, smem, regs


# ---------------------------------------------------------------------------------------------
def tile_steps_per_column(n_qb, BQ, BK, Sq, Sk, causal, window):
    """Число шагов по ключам для каждого блока запросов ОДНОЙ пары (b,h). Список длины n_qb."""
    out = []
    off = Sk - Sq  # причинная маска: запрос i видит ключи <= i+off
    for qb in range(n_qb):
        q_lo = qb * BQ
        q_hi = min(Sq, q_lo + BQ)
        k_hi = min(Sk, q_hi + off) if causal else Sk
        k_lo = 0
        if window:
            k_lo = max(0, q_lo + off - window + 1)
        if k_hi <= k_lo:
            out.append(0)
            continue
        out.append((k_hi + BK - 1) // BK - k_lo // BK)
    return out


def analyze(cfg):
    B, H, Hkv = cfg["B"], cfg["H"], cfg["Hkv"]
    Sq, Sk, d = cfg["Sq"], cfg["Sk"], cfg["d"]
    BQ, BK = cfg["BQ"], cfg["BK"]
    esz = cfg["esz"]
    causal, window = cfg["causal"], cfg["window"]
    sm = cfg["sm"]
    L2 = cfg["l2"]

    r = {"cfg": dict(cfg)}
    if H % Hkv:
        raise SystemExit("H должно делиться на Hkv")
    G = H // Hkv

    n_qb = (Sq + BQ - 1) // BQ
    n_kb = (Sk + BK - 1) // BK
    grid = n_qb * B * H
    ctas_sm, occ_src, smem, regs = (
        (cfg["ctas_per_sm"], "задано", cfg["smem"], cfg["regs"])
        if cfg.get("ctas_per_sm")
        else occupancy(BQ, BK, d, cfg.get("smem"), cfg.get("regs"))
    )
    n_res = min(grid, sm * ctas_sm)
    waves = grid / n_res

    # --- состав волны: какие потоки KV резидентны и сколько CTA на каждый ---------------------
    # порядок запуска: grid = (n_qb, H, B), x -- младшая компонента (attn_fwd_cutlass.cu:655,
    # kernel_forward.h:getBlocksGrid) => линейный id = qb + n_qb*(h + H*b).
    streams = Counter()
    for i in range(n_res):
        bh = i // n_qb
        h, b = bh % H, bh // H
        streams[(b, h // G)] += 1
    n_streams = len(streams)
    R = list(streams.values())
    R_avg = n_res / n_streams

    step_bytes = 2 * BK * d * esz  # K-плитка + V-плитка одного шага по ключам
    win_live = sum(min(rs, n_kb) for rs in R) * step_bytes  # ОКНО (модель)
    win_sync = n_streams * 2 * step_bytes  # нижняя: идеальная синхронность
    win_full = n_streams * n_kb * step_bytes  # верхняя: весь KV потоков волны
    q_res = n_res * BQ * d * esz  # плитки Q резидентных CTA

    # --- трафик --------------------------------------------------------------------------------
    steps_col = tile_steps_per_column(n_qb, BQ, BK, Sq, Sk, causal, window)
    steps_tot = sum(steps_col) * B * H  # шагов по ключам во всём запуске
    l2_kv = steps_tot * step_bytes
    q_bytes = B * H * Sq * d * esz
    o_bytes = B * H * Sq * d * esz
    l2_all = l2_kv + q_bytes + o_bytes

    # обязательный (компульсорный) трафик HBM: каждый байт хотя бы раз
    kv_reach = (
        Sk if not window else min(Sk, window + Sq)
    )  # оконное внимание трогает меньше
    hbm_kv_min = 2 * B * Hkv * kv_reach * d * esz
    hbm_min_read = hbm_kv_min + q_bytes

    # переиспользование плитки между CTA: работает, пока окно живёт в L2
    reuse_ideal = R_avg
    if win_live <= L2:
        reuse = reuse_ideal
        regime = "поток резидентен"
    elif win_live > L2 and win_sync <= L2:
        reuse = max(1.0, reuse_ideal * L2 / win_live)  # [МОДЕЛЬНОЕ ДОПУЩЕНИЕ, не замер]
        regime = "окно шире L2 -- переиспользование частичное"
    else:
        reuse = 1.0
        regime = "не влезает даже синхронное окно"
    if not cfg["coherent"]:
        reuse = 1.0  # несогласованные маски: делить нечего
        regime += " + маски НЕ согласованы"

    hbm_kv = max(hbm_kv_min, l2_kv / reuse)
    hbm_read = hbm_kv + q_bytes
    hbm_write = o_bytes
    hit = 1.0 - hbm_read / l2_all if l2_all else 0.0
    # ЗАПАС ПО HBM: во сколько раз чтение из HBM выше обязательного минимума. Именно он, а не
    # "влезло/не влезло", решает, есть ли что сокращать на уровне HBM: при запасе 1.0 плитки уже
    # ходят по одному разу, и любое сокращение ЧТЕНИЙ ПЛИТОК упирается в объём самих тензоров.
    slack_hbm = hbm_kv / hbm_kv_min if hbm_kv_min else 1.0

    # --- каналы времени -------------------------------------------------------------------------
    flops = 0.0
    for qb, ns in enumerate(steps_col):
        flops += ns * 2.0 * (2 * BQ * BK * d)  # QK^T и P*V, 2 флопа на МАС
    flops *= B * H
    t_hbm = (hbm_read + hbm_write) / MACHINE["hbm_rw"]
    t_l2 = l2_all / cfg["l2_bw"]
    t_tensor = flops / MACHINE["tensor_peak"]
    chan = {"HBM": t_hbm, "L2": t_l2, "тензорный": t_tensor}
    binding = max(chan, key=chan.get)
    t_model = chan[binding]

    # потолок от сокращения трафика ПЛИТОК в cut раз (cut=inf -> до обязательного минимума)
    def ceiling(cut):
        kv2 = l2_kv / cut if cut != float("inf") else 0.0
        l2b = kv2 + q_bytes + o_bytes
        hb = max(hbm_kv_min, kv2 / reuse) + q_bytes
        t2 = max((hb + hbm_write) / MACHINE["hbm_rw"], l2b / cfg["l2_bw"], t_tensor)
        return t_model / t2 if t2 else float("inf")

    # загрузка каналов относительно ЗАМЕРЕННОГО времени, если оно дано. Без него модель может
    # сказать только "канал X самый нагруженный", но не "канал X в упоре".
    tmeas = cfg.get("tmeas")
    load = {k: v / tmeas for k, v in chan.items()} if tmeas else None

    r.update(
        dict(
            slack_hbm=slack_hbm,
            tmeas=tmeas,
            load=load,
            G=G,
            n_qb=n_qb,
            n_kb=n_kb,
            grid=grid,
            ctas_sm=ctas_sm,
            occ_src=occ_src,
            smem=smem,
            regs=regs,
            threads=threads_of(BQ),
            n_res=n_res,
            waves=waves,
            n_streams=n_streams,
            R=sorted(R, reverse=True),
            R_avg=R_avg,
            step_bytes=step_bytes,
            win_live=win_live,
            win_sync=win_sync,
            win_full=win_full,
            q_res=q_res,
            steps_tot=steps_tot,
            l2_kv=l2_kv,
            l2_all=l2_all,
            hbm_kv_min=hbm_kv_min,
            hbm_read=hbm_read,
            hbm_write=hbm_write,
            hit=hit,
            reuse=reuse,
            regime=regime,
            flops=flops,
            chan=chan,
            binding=binding,
            t_model=t_model,
            ceil2=ceiling(2.0),
            ceil13=ceiling(13.4),
            ceilinf=ceiling(float("inf")),
            fits=win_live <= L2,
            fits_full=win_full <= L2,
            fits_sync=win_sync <= L2,
            ktile_smem=BK * d * esz,
        )
    )
    return r


# ---------------------------------------------------------------------------------------------
def mb(x):
    return f"{x / 1024**2:8.2f} МБ"


def render(r):
    c = r["cfg"]
    L = []
    P = L.append
    P("=" * 96)
    P(
        f"ФОРМА  B={c['B']} H={c['H']} Hkv={c['Hkv']} (GQA {r['G']}:1)  Sq={c['Sq']} Sk={c['Sk']} "
        f"d={c['d']}  плитка BQ={c['BQ']} BK={c['BK']}  элемент {c['esz']} Б  "
        f"{'причинная' if c['causal'] else 'полная'}"
        + (f" окно={c['window']}" if c["window"] else "")
    )
    P("=" * 96)
    if c["Sq"] < c["BQ"]:
        P(
            "!!! Sq < BQ: это ФОРМА ДЕКОДА, а модель написана под ПРЕФИЛЛ. Плитки KV и волна считаются"
        )
        P(
            "!!! верно, но тензорный канал БЕССМЫСЛЕН (считается полная плитка запросов), а связывающим"
        )
        P(
            "!!! у декода ЗАМЕРЕНА задержка, которой в модели нет. Вердикт по времени НЕДЕЙСТВИТЕЛЕН."
        )
    P("")
    P(
        "--- ВОЛНА ---------------------------------------------------------------------------------"
    )
    P(
        f"сетка                       {r['grid']} CTA  ({r['n_qb']} блоков запросов x {c['B'] * c['H']} (b,h))"
    )
    P(
        f"CTA на SM                   {r['ctas_sm']}   [{r['occ_src']}: smem {r['smem'] / 1024:.1f} КБ, "
        f"{r['regs']} рег, {r['threads']} нитей]"
    )
    P(
        f"резидентно одновременно     {r['n_res']} CTA на {c['sm']} SM;  волн {r['waves']:.2f}"
    )
    P(
        f"потоков KV в волне          {r['n_streams']}  (CTA на поток: {r['R'][:8]}"
        f"{' ...' if len(r['R']) > 8 else ''}, среднее {r['R_avg']:.1f})"
    )
    P(
        f"плиток ключей в потоке      {r['n_kb']}   шаг по ключам = K+V = {r['step_bytes'] / 1024:.0f} КБ"
    )
    P("")
    P(
        "--- ОКНО (рабочее множество KV, живое в один момент) --------------------------------------"
    )
    P(
        f"нижняя  (идеальная синхронность, 2 плитки/поток)   {mb(r['win_sync'])}"
        f"   {'влезает' if r['fits_sync'] else 'НЕ влезает'}"
    )
    P(
        f"МОДЕЛЬ  (min(CTA потока, плиток) x шаг)            {mb(r['win_live'])}"
        f"   {'ВЛЕЗАЕТ' if r['fits'] else 'НЕ ВЛЕЗАЕТ'}   <-- вердикт по нему"
    )
    P(
        f"верхняя (весь KV потоков волны)                    {mb(r['win_full'])}"
        f"   {'влезает' if r['fits_full'] else 'НЕ влезает'}"
    )
    P(f"плитки Q резидентных CTA (тоже в L2)               {mb(r['q_res'])}")
    P(
        f"окно + резидентные Q                              {mb(r['win_live'] + r['q_res'])}"
        f"   {'влезает' if r['win_live'] + r['q_res'] <= c['l2'] else 'НЕ влезает'}"
        f"   (Q читается один раз -- вердикт ведётся по KV)"
    )
    P(f"L2 = {mb(c['l2'])}                                     режим: {r['regime']}")
    P("")
    P(
        "--- РАЗДЕЛЯЕМАЯ И L1 ----------------------------------------------------------------------"
    )
    P(
        f"плитка K целиком (BK*d*esz)  {r['ktile_smem'] / 1024:7.1f} КБ  из 96 КБ разделяемой  -> "
        f"{
            'влезает'
            if r['ktile_smem'] <= MACHINE['smem_per_sm']
            else 'НЕ ВЛЕЗАЕТ: ядро дробит плитку по d, '
            'и один и тот же K читается из L2 несколько раз'
        }"
    )
    P(
        f"окно шагов резидентных CTA одного SM  {r['ctas_sm'] * r['step_bytes'] / 1024:7.1f} КБ  против "
        f"L1 {MACHINE['l1_per_sm'] / 1024:.0f} КБ  -> "
        f"{'могло бы жить в L1' if r['ctas_sm'] * r['step_bytes'] <= MACHINE['l1_per_sm'] else 'L1 не держит; всё переиспользование обязано идти через L2'}"
    )
    P("")
    P(
        "--- ТРАФИК --------------------------------------------------------------------------------"
    )
    P(f"шагов по ключам во всём запуске   {r['steps_tot']}")
    P(f"обращения к L2 (плитки KV)        {mb(r['l2_kv'])}")
    P(f"обращения к L2 (всего, +Q,+O)     {mb(r['l2_all'])}")
    P(
        f"множитель переиспользования       x{r['reuse']:.1f}   (CTA потока делят одну плитку)"
    )
    P(
        f"чтение из HBM (модель)            {mb(r['hbm_read'])}   при обязательном минимуме "
        f"{mb(r['hbm_kv_min'] + c['B'] * c['H'] * c['Sq'] * c['d'] * c['esz'])}"
    )
    P(
        f"ЗАПАС ПО HBM (чтение KV / минимум)  x{r['slack_hbm']:.2f}"
        f"   <-- вот что решает, есть ли что сокращать"
    )
    P(f"ПРЕДСКАЗАННАЯ ДОЛЯ ПОПАДАНИЙ L2   {100 * r['hit']:.2f} %")
    P("")
    P(
        "--- КАНАЛЫ (грубая крыша; связывающий канал у нашего d<=128 префилла ЗАМЕРЕН ИНОЙ, см. ниже)"
    )
    for k, v in sorted(r["chan"].items(), key=lambda kv: -kv[1]):
        s = f"  {k:<12} {v * 1e6:9.1f} мкс"
        if r["load"]:
            s += f"   загрузка {100 * r['load'][k]:5.1f} % замеренного времени"
        if k == r["binding"]:
            s += "   <-- максимум по модели"
        P(s)
    P("")
    P(
        "--- ВЕРДИКТ -------------------------------------------------------------------------------"
    )
    if r["fits"]:
        P(
            f"1. ОКНО ВЛЕЗАЕТ В L2 ({r['win_live'] / 1024**2:.2f} из {c['l2'] / 1024**2:.0f} МБ): плитки отдаёт L2, "
            f"переиспользование x{r['reuse']:.0f}."
        )
    else:
        P(
            f"1. ОКНО НЕ ВЛЕЗАЕТ В L2 ({r['win_live'] / 1024**2:.2f} против {c['l2'] / 1024**2:.0f} МБ): "
            f"переиспользование срезано до x{r['reuse']:.0f}."
        )
    if r["slack_hbm"] < 1.05:
        P(
            "2. ЧТЕНИЕ KV ИЗ HBM УЖЕ НА ОБЯЗАТЕЛЬНОМ МИНИМУМЕ (каждый байт едет один раз)."
        )
        P(
            "   СОКРАЩЕНИЕ ЧИСЛА ЧТЕНИЙ ПЛИТОК ВО ВРЕМЯ НЕ ПЕРЕХОДИТ. Живёт только сокращение ОБЪЁМА"
        )
        P("   (меньше байт на элемент, меньше самих ключей) -- это другая правка.")
    else:
        P(
            f"2. ЧТЕНИЕ KV ИЗ HBM В x{r['slack_hbm']:.1f} ВЫШЕ ОБЯЗАТЕЛЬНОГО: есть что сокращать на HBM."
        )
    if r["load"]:
        mx = max(r["load"].values())
        if mx < 0.60:
            P(
                f"3. НИ ОДИН МОДЕЛЬНЫЙ КАНАЛ НЕ В УПОРЕ (самый нагруженный на {100 * mx:.0f} % замеренного"
            )
            P(
                "   времени). Связывает ресурс, которого в модели НЕТ -- см. п.5 раздела 'не разобрано'."
            )
            P(
                "   ВЕРДИКТ ПО ТРАФИКУ: не переходит, потолок x1.00 независимо от строки ниже."
            )
        else:
            P(
                f"3. Самый нагруженный канал ({r['binding']}) занимает {100 * mx:.0f} % замеренного времени."
            )
    else:
        P(
            "3. Замеренное время не задано (--tmeas). Без него модель различает каналы между собой, но"
        )
        P(
            "   НЕ МОЖЕТ сказать, находится ли хоть один в упоре. Потолок ниже -- ВЕРХНЯЯ ГРАНИЦА."
        )
    P(
        f"потолок ускорения (ВЕРХНЯЯ ГРАНИЦА, не предсказание) при сокращении чтений плиток: "
        f"x2 -> {r['ceil2']:.3f} | x13.4 -> {r['ceil13']:.3f} | до минимума -> {r['ceilinf']:.3f}"
    )
    if r["binding"] != "HBM":
        P(
            f"ВНИМАНИЕ: по модели связывает не HBM, а {r['binding']}. Любой ход по трафику ограничен сверху"
        )
        P("этим каналом, каким бы большим ни было сокращение.")
    return "\n".join(L)


BLIND = """
--- НЕ РАЗОБРАНО / ЧЕГО ЭТОТ РАСЧЁТ НЕ ЗНАЕТ (печатается ВСЕГДА) ---------------------------------
 1. ПОЛИТИКА ВЫТЕСНЕНИЯ L2. Принят идеальный LRU: влезло окно -- значит переиспользуется. На V100
    L2 секционирован (2 секции по 3 МБ, адреса раскиданы по секциям), есть streaming-подсказки,
    которых мы не выставляем. Реальная граница может быть НИЖЕ 6 МБ, и модель этого не увидит.
 2. ПОРЯДОК ЗАПУСКА CTA принят линейным (x -- младшая). Это верно для нашей сетки, но аппаратный
    диспетчер не обязан выдавать блоки строго по порядку, а при доигрывании волны состав
    резидентных CTA другой (перекрытие волн). Модель считает волну МГНОВЕННЫМ СНИМКОМ.
 3. РАСХОЖДЕНИЕ ПОЗИЦИЙ внутри потока принято равным числу резидентных CTA потока. Это ставка,
    выведенная из очереди "доиграл -- на его место встал следующий с нуля", а НЕ замер.
    Прямой замер (счётчик попаданий L2 против предсказания) сделан только на одной форме -- см.
    якорь A в --selftest.
 4. ПРЕФИКС-КЭШ И ЧАНКОВЫЙ ПРЕФИЛЛ. Часть K/V может уже лежать в L2 от предыдущего вызова; здесь
    каждый запуск считается холодным. На чанковом префилле (q_len << Sk) это занижает попадания.
 5. ЭТО НЕ МОДЕЛЬ ВРЕМЕНИ. Каналов ровно три: HBM, L2, тензорный. У нашего d<=128 префилла
    ЗАМЕРЕНО, что связывает КОНВЕЙЕР РАЗДЕЛЯЕМОЙ ПАМЯТИ (mio_throttle 1.54 + short_scoreboard 0.87
    = половина простоев; smem-волны 38 %, выдача 45 %) -- этого канала в модели НЕТ. Поэтому
    "трафик переходит во время" -- условие НЕОБХОДИМОЕ, а не достаточное; обратный вердикт
    ("не переходит") надёжен, прямой -- нет.
 6. ПРОПУСКНАЯ СПОСОБНОСТЬ L2 НЕ ЗАМЕРЕНА на этой машине (2155 ГБ/с -- [ОЦЕНКА] из вендорских
    данных). Канал L2 в таблице поэтому мягкий; HBM и тензорный опираются на замеры.
 7. BACKWARD НЕ МОДЕЛИРУЕТСЯ. Там обход по КЛЮЧАМ, а не по запросам, и 90 % трафика DRAM -- это
    круговые рейсы накопителей dK/dV, а не сам KV. Считать его этой формулой нельзя.
 8. РАЗРЕЖЕННОСТЬ учитывается только окном (--window) и флагом --incoherent (маски не
    согласованы -> переиспользование = 1). Настоящий отбор блоков не моделируется.
 9. ФОРМАТ KV входит только размером элемента (--esz). Цена распаковки e4m3/int8 (у Volta нет
    fp8 в железе) -- не здесь, она в счёте команд.
10. ЗАНЯТОСТЬ (CTA/SM) взята из замеров только для трёх конфигураций d=128; для d=256/512 это
    ОЦЕНКА по подгонке smem = 16 КБ + BQ*(2d+32). Ставьте --ctas-per-sm / --smem / --regs, если
    знаете точно. Вердикты якорей от этого не зависят (проверено перебором 1..4 CTA/SM).
11. РАСЩЕПЛЕНИЕ ПО КЛЮЧАМ (split-K, num_splits>1) меняет сетку и состав волны -- не учтено.
"""

# ---------------------------------------------------------------------------------------------
PRESETS = {
    # d=512 глобальные (full_attention) слои Gemma-4, MQA: Hkv=1, 8 слоёв из 48.
    "gemma4-global": dict(B=1, H=16, Hkv=1, d=512, Sq=16384, Sk=16384, causal=True),
    # профиль ncu, по которому получены 97.7 % попаданий (SM70_KERNEL_PLAYBOOK §46)
    "ncu-fwd128": dict(
        B=1,
        H=4,
        Hkv=4,
        d=128,
        Sq=4096,
        Sk=4096,
        BQ=32,
        BK=32,
        causal=True,
        ctas_per_sm=3,
        smem=25 * 1024,
        regs=168,
    ),
    "decode-128k": dict(B=32, H=16, Hkv=8, d=128, Sq=1, Sk=131072, causal=True),
}


def make_cfg(a):
    cfg = dict(
        B=a.B,
        H=a.H,
        Hkv=a.Hkv,
        Sq=a.Sq,
        Sk=a.Sk,
        d=a.d,
        esz=a.esz,
        causal=a.causal,
        window=a.window,
        sm=a.sm,
        l2=int(a.l2 * 1024**2),
        l2_bw=a.l2_bw * 1e9,
        coherent=not a.incoherent,
        ctas_per_sm=a.ctas_per_sm,
        smem=a.smem,
        regs=a.regs,
        tmeas=(a.tmeas * 1e-6 if a.tmeas else None),
    )
    if a.preset:
        cfg.update(PRESETS[a.preset])
        for k, v in (
            ("Sq", a.Sq_set),
            ("Sk", a.Sk_set),
            ("B", a.B_set),
            ("H", a.H_set),
        ):
            if v is not None:
                cfg[k] = v
        if cfg["Sq"] == 16384 and a.Sk_set and not a.Sq_set:
            cfg["Sq"] = cfg["Sk"]
    if a.BQ:
        cfg["BQ"] = a.BQ
    if a.BK:
        cfg["BK"] = a.BK
    if "BQ" not in cfg or "BK" not in cfg:
        bq, bk = shipped_tile(cfg["d"], cfg["Sq"], cfg["Sk"])
        cfg.setdefault("BQ", bq)
        cfg.setdefault("BK", bk)
    cfg.setdefault("ctas_per_sm", None)
    cfg.setdefault("smem", None)
    cfg.setdefault("regs", None)
    cfg.setdefault("coherent", True)
    cfg.setdefault("window", None)
    cfg.setdefault("sm", MACHINE["sm"])
    cfg.setdefault("l2", MACHINE["l2"])
    cfg.setdefault("l2_bw", MACHINE["l2_bw"])
    cfg.setdefault("esz", 2)
    return cfg


def base_cfg(**kw):
    cfg = dict(
        B=1,
        H=8,
        Hkv=8,
        Sq=4096,
        Sk=4096,
        d=128,
        esz=2,
        causal=True,
        window=None,
        sm=MACHINE["sm"],
        l2=MACHINE["l2"],
        l2_bw=MACHINE["l2_bw"],
        coherent=True,
        ctas_per_sm=None,
        smem=None,
        regs=None,
        tmeas=None,
    )
    cfg.update(kw)
    if "BQ" not in cfg or "BK" not in cfg:
        bq, bk = shipped_tile(cfg["d"], cfg["Sq"], cfg["Sk"])
        cfg.setdefault("BQ", bq)
        cfg.setdefault("BK", bk)
    return cfg


# ---------------------------------------------------------------------------------------------
def sk_crit(B, H, Hkv, d, esz, lo=256, hi=524288):
    """Наименьший Sk (квадратный префилл), при котором окно ПЕРЕСТАЁТ влезать в L2.

    Функция 'влезает(Sk)' НЕ монотонна: диспетчер меняет плитку с длиной, и переключение может
    вернуть форму обратно под границу. Поэтому ищем ПЕРВОЕ пересечение сканом по кратным 256,
    а не бисекцией -- бисекция по немонотонной функции даёт уверенный неверный ответ.
    """
    S, prev = lo, None
    while S <= hi:
        f = analyze(base_cfg(B=B, H=H, Hkv=Hkv, d=d, Sq=S, Sk=S, esz=esz))["fits"]
        if prev is True and f is False:
            return S
        prev = f
        S += 256
    return None


def scan(a):
    """ГРАНИЦА ПО ФОРМЕ: при каком Sk окно перестаёт влезать в L2."""
    print("=" * 96)
    print(
        "ГРАНИЦА ПО ФОРМЕ: окно KV против L2 (6 МБ). Плитка -- ОТГРУЖЕННАЯ политика диспетчера."
    )
    print(
        f"B={a.B} H={a.H} Hkv={a.Hkv}, Sq=Sk (квадратный префилл), причинная маска, элемент {a.esz} Б"
    )
    print("=" * 96)
    lens = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    for d in (64, 128, 256, 512):
        print(f"\nd = {d}")
        print(
            f"  {'Sk':>8} {'BQxBK':>9} {'CTA/SM':>7} {'резид.':>7} {'потоков':>8} "
            f"{'шаг КБ':>8} {'ОКНО МБ':>9}  вердикт"
        )
        prev = None
        for S in lens:
            cfg = base_cfg(B=a.B, H=a.H, Hkv=a.Hkv, d=d, Sq=S, Sk=S, esz=a.esz)
            r = analyze(cfg)
            v = "влезает" if r["fits"] else "НЕ ВЛЕЗАЕТ"
            mark = ""
            if prev is not None and prev != r["fits"]:
                mark = "   <-- ГРАНИЦА"
            prev = r["fits"]
            print(
                f"  {S:>8} {cfg['BQ']:>4}x{cfg['BK']:<4} {r['ctas_sm']:>7} {r['n_res']:>7} "
                f"{r['n_streams']:>8} {r['step_bytes'] / 1024:>8.0f} {r['win_live'] / 1024**2:>9.2f}"
                f"  {v}{mark}"
            )
        # предельное число резидентных CTA на поток при этой геометрии
        cfg = base_cfg(B=a.B, H=a.H, Hkv=a.Hkv, d=d, Sq=65536, Sk=65536, esz=a.esz)
        step = 2 * cfg["BK"] * d * a.esz
        crit = sk_crit(a.B, a.H, a.Hkv, d, a.esz)
        print(
            f"  предел: в L2 помещается {MACHINE['l2'] // step} одновременно живых шагов по ключам "
            f"(шаг {step / 1024:.0f} КБ);  ПЕРВОЕ ПЕРЕСЕЧЕНИЕ ГРАНИЦЫ Sk = "
            + (f"{crit}" if crit else "нет на 256..524288")
        )
        if crit:
            print(
                "          (функция немонотонна: переключение плитки диспетчером может вернуть "
                "форму ПОД границу -- смотрите таблицу)"
            )
    print(
        "\nЧитать так: окно растёт с Sk, пока число плиток ключей меньше числа резидентных CTA"
    )
    print(
        "потока; дальше упирается в число резидентных CTA. Поэтому ПЕРЕХОД ЧЕРЕЗ ГРАНИЦУ ЛЕЧИТСЯ"
    )
    print(
        "НЕ СОКРАЩЕНИЕМ ТРАФИКА, А УМЕНЬШЕНИЕМ ЧИСЛА РЕЗИДЕНТНЫХ CTA (большая плитка запросов) --"
    )
    print("именно это отгруженный диспетчер и делает при Sk>=16384 (BQ 32 -> 64).")


# ---------------------------------------------------------------------------------------------
def selftest():
    ok = True

    def check(name, cond, got):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'OK ' if cond else 'ПРОВАЛ'}] {name}: {got}")

    print("#" * 96)
    print(
        "# САМОПРОВЕРКА: ЯКОРЬ A -- ВОСПРОИЗВЕДЕНИЕ ИЗВЕСТНОГО ЧИСЛА (97.7 % попаданий в L2)"
    )
    print("#" * 96)
    print(
        "Источник: docs/SM70_KERNEL_PLAYBOOK.md §46 -- ncu на volta_fwd_mainloop, S=4096 H=4 d=128,"
    )
    print(
        "рабочая точка 32x32: попадания в L2 97.7 % (16.95 М из 17.35 М секторов = 555.2 МБ),"
    )
    print("чтение из HBM 12.6 МБ при минимально нужных 16 МБ (Q+K+V+O).")
    cfg = base_cfg(**PRESETS["ncu-fwd128"])
    r = analyze(cfg)
    print(
        f"  модель: обращений к L2 {r['l2_all'] / 1024**2:.1f} МБ (замерено 555.2), "
        f"чтение HBM {r['hbm_read'] / 1024**2:.1f} МБ (замерено 12.6),"
    )
    print(f"          доля попаданий {100 * r['hit']:.2f} % (замерено 97.70)")
    err_hit = abs(100 * r["hit"] - 97.70)
    err_l2 = abs(r["l2_all"] / 1024**2 - 555.2) / 555.2
    err_hbm = abs(r["hbm_read"] / 1024**2 - 12.6) / 12.6
    check(
        "попадания L2", err_hit < 0.5, f"расхождение {err_hit:.2f} пункта (порог 0.5)"
    )
    check(
        "объём обращений к L2",
        err_l2 < 0.10,
        f"расхождение {100 * err_l2:.1f} % (порог 10)",
    )
    check(
        "чтение из HBM", err_hbm < 0.15, f"расхождение {100 * err_hbm:.1f} % (порог 15)"
    )

    print("\n" + "#" * 96)
    print("# ЯКОРЬ B1 -- d=128, Sk=4096, ОБЫЧНЫЕ B*H: обязано быть 'ВЛЕЗАЕТ В L2'")
    print("#" * 96)
    allfit = True
    print(
        f"  {'B':>3} {'H':>3} {'Hkv':>4} {'BQxBK':>9} {'резид':>6} {'потоков':>8} {'ОКНО МБ':>9} "
        f"{'попад. L2':>10}  вердикт"
    )
    for B, H, Hkv in [
        (1, 8, 8),
        (1, 32, 8),
        (2, 16, 4),
        (4, 32, 8),
        (8, 8, 1),
        (1, 4, 4),
        (16, 40, 10),
    ]:
        cfg = base_cfg(B=B, H=H, Hkv=Hkv, d=128, Sq=4096, Sk=4096)
        r = analyze(cfg)
        allfit = allfit and r["fits"]
        print(
            f"  {B:>3} {H:>3} {Hkv:>4} {cfg['BQ']:>4}x{cfg['BK']:<4} {r['n_res']:>6} "
            f"{r['n_streams']:>8} {r['win_live'] / 1024**2:>9.2f} {100 * r['hit']:>9.1f} %  "
            f"{'ВЛЕЗАЕТ' if r['fits'] else 'НЕ ВЛЕЗАЕТ'}"
        )
    check(
        "d=128 Sk=4096 влезает при всех обычных B*H",
        allfit,
        "7 форм из 7" if allfit else "есть промах",
    )
    # устойчивость к неизвестной занятости
    rob = all(
        analyze(
            base_cfg(
                B=1,
                H=32,
                Hkv=8,
                d=128,
                Sq=4096,
                Sk=4096,
                ctas_per_sm=k,
                smem=25 * 1024,
                regs=168,
            )
        )["fits"]
        for k in (1, 2, 3, 4)
    )
    check("вердикт не зависит от CTA/SM (1..4)", rob, "устойчив" if rob else "зависит")

    print("\n" + "#" * 96)
    print("# ЯКОРЬ B2 -- d=512, боевая Gemma-4 (Hkv=1, MQA): обязано быть 'НЕ ВЛЕЗАЕТ'")
    print("#" * 96)
    nofit = True
    print(
        f"  {'Sk':>8} {'BQxBK':>9} {'резид':>6} {'потоков':>8} {'шаг КБ':>8} {'ОКНО МБ':>9} "
        f"{'+Q МБ':>7}  вердикт"
    )
    for S in (1024, 2048, 4096, 8192, 16384, 65536, 262144):
        cfg = base_cfg(B=1, H=16, Hkv=1, d=512, Sq=S, Sk=S)
        r = analyze(cfg)
        if S >= 4096:
            nofit = nofit and (not r["fits"])
        print(
            f"  {S:>8} {cfg['BQ']:>4}x{cfg['BK']:<4} {r['n_res']:>6} {r['n_streams']:>8} "
            f"{r['step_bytes'] / 1024:>8.0f} {r['win_live'] / 1024**2:>9.2f} "
            f"{(r['win_live'] + r['q_res']) / 1024**2:>7.2f}  "
            f"{'ВЛЕЗАЕТ' if r['fits'] else 'НЕ ВЛЕЗАЕТ'}"
        )
    check(
        "d=512 Gemma-4 не влезает на боевых длинах (Sk>=4096)",
        nofit,
        "4 длины из 4" if nofit else "где-то влезло",
    )
    print(
        f"  ГРАНИЦА (расчётная): Sk = {sk_crit(1, 16, 1, 512, 2)}. НИЖЕ неё d=512 ВЛЕЗАЕТ, и это не"
    )
    print(
        "  ошибка модели: при Sk=2048 весь K+V одной MQA-головы -- 4 МБ, он и правда помещается"
    )
    print(
        "  в 6 МБ. Боевой Gemma-4 работает на 16K..256K, то есть выше границы вчетверо и более."
    )
    r = analyze(base_cfg(B=1, H=16, Hkv=1, d=512, Sq=16384, Sk=16384))
    print(
        f"  сверка с ручным счётом из наряда: плитка K = 128*512*2 = "
        f"{128 * 512 * 2 // 1024} КБ; шаг (K+V) = {r['step_bytes'] // 1024} КБ; "
        f"волна {r['n_res']} CTA -> окно {r['win_live'] / 1024**2:.1f} МБ >> 6 МБ"
    )
    check(
        "потолок ускорения при d=512 БОЛЬШЕ единицы (трафик переходит)",
        r["ceilinf"] > 1.05,
        f"x{r['ceilinf']:.2f} до обязательного минимума, запас по HBM x{r['slack_hbm']:.1f}",
    )

    print("\n" + "#" * 96)
    print(
        "# ЯКОРЬ C -- ЛЕСТНИЦА ПЛИТКИ ЗАПРОСОВ: трафик падает вчетверо, ВРЕМЯ РАСТЁТ в 1.41 раза"
    )
    print("#" * 96)
    print(
        "Замерено (SM70_KERNEL_PLAYBOOK §25.2): BQ 32/64/128 -> 353.1/376.1/495.1 мкс."
    )
    print(
        "ФОРМА В §25.2 НЕ НАПИСАНА -- восстановлена и ПРОВЕРЕНА: §21.1 даёт S=2048 H=8 причинно,"
    )
    print(
        "358.2 мкс на той же конфигурации и 17.4 М тензорных команд; 17.4e6 x 256 МАС x 2 ="
    )
    print(
        "8.9 ГФЛОП, а B=1 H=8 S=2048 d=128 причинно даёт 8.6 ГФЛОП -- сходится. Формы S=4096"
    )
    print(
        "ОТВЕРГНУТЫ: они потребовали бы 78 % тензорного пика при замеренных 29 % занятости пайпа."
    )
    meas = {32: 353.1, 64: 376.1, 128: 495.1}
    rs = {}
    print(
        f"  {'BQ':>4} {'шагов':>8} {'трафик L2 МБ':>13} {'ОКНО МБ':>9} {'влез':>6} "
        f"{'замер мкс':>10} {'самый нагруж. канал':>21}"
    )
    for BQ in (32, 64, 128):
        cfg = base_cfg(
            B=1,
            H=8,
            Hkv=8,
            d=128,
            Sq=2048,
            Sk=2048,
            BQ=BQ,
            BK=128,
            tmeas=meas[BQ] * 1e-6,
        )
        r = rs[BQ] = analyze(cfg)
        mx = max(r["load"].values())
        print(
            f"  {BQ:>4} {r['steps_tot']:>8} {r['l2_kv'] / 1024**2:>13.1f} "
            f"{r['win_live'] / 1024**2:>9.2f} {('да' if r['fits'] else 'НЕТ'):>6} "
            f"{meas[BQ]:>10.1f} {r['binding']:>12} {100 * mx:>6.1f} %"
        )
    check(
        "трафик при BQ=128 вчетверо меньше, чем при BQ=32",
        abs(rs[32]["l2_kv"] / rs[128]["l2_kv"] - 4.0) < 0.05,
        f"x{rs[32]['l2_kv'] / rs[128]['l2_kv']:.2f}",
    )
    check(
        "все три точки влезают в L2 и чтение KV на обязательном минимуме",
        all(rs[b]["fits"] and rs[b]["slack_hbm"] < 1.05 for b in rs),
        "запас по HBM x" + "/".join(f"{rs[b]['slack_hbm']:.2f}" for b in (32, 64, 128)),
    )
    loads = [max(rs[b]["load"].values()) for b in (32, 64, 128)]
    check(
        "ни один канал не в упоре -> вердикт 'сокращение трафика во время НЕ переходит'",
        all(l < 0.60 for l in loads),
        "загрузка самого нагруженного "
        + " -> ".join(f"{100 * l:.0f}%" for l in loads)
        + f" при росте времени в {meas[128] / meas[32]:.2f} раза",
    )
    print(
        f"  Читать так: трафик срезан вчетверо, загрузка самого нагруженного канала упала с "
        f"{100 * loads[0]:.0f} % до {100 * loads[2]:.0f} %,"
    )
    print(
        f"  а ВРЕМЯ ВЫРОСЛО в {meas[128] / meas[32]:.2f} раза. Инструмент обязан был отказать -- и отказал."
    )
    print(
        "  ПОБОЧНО: замер ОГРАНИЧИВАЕТ СНИЗУ пропускную L2. При BQ=32 ядро прогнало "
        f"{rs[32]['l2_all'] / 1024**2:.0f} МБ"
    )
    print(
        f"  через L2 за {meas[32]} мкс = {rs[32]['l2_all'] / meas[32] * 1e6 / 1e9:.0f} ГБ/с -- это НИЖНЯЯ "
        "оценка, а принятые 2155 ГБ/с ([ОЦЕНКА])"
    )
    print(
        "  ей не противоречат. Если бы вышло больше 2155, параметр был бы опровергнут."
    )

    print("\n" + "#" * 96)
    print(
        "# ЯКОРЬ D -- ДЕКОД: ЗАФИКСИРОВАННЫЙ ПРОМАХ ИНСТРУМЕНТА (печатается намеренно)"
    )
    print("#" * 96)
    r = analyze(base_cfg(**PRESETS["decode-128k"]))
    print(
        f"  B=32 H=16 Hkv=8 Sq=1 Sk=131072 d=128: потоков {r['n_streams']}, "
        f"окно {r['win_live'] / 1024**2:.1f} МБ ({'влезает' if r['fits'] else 'не влезает'}), "
        f"переиспользование x{r['reuse']:.0f},"
    )
    print(
        f"  попадания L2 {100 * r['hit']:.1f} %, запас по HBM x{r['slack_hbm']:.2f}, "
        f"связывает по модели {r['binding']} "
        f"(HBM/тензор = {r['chan']['HBM'] / r['chan']['тензорный']:.0f}x)"
    )
    check(
        "модель называет декод HBM-связанным",
        r["binding"] == "HBM",
        f"канал {r['binding']}",
    )
    print(
        "  ЭТО РАСХОДИТСЯ С ЗАМЕРОМ, и расхождение НЕ сглаживается. SM70_KERNEL_PLAYBOOK §22:"
    )
    print(
        "  на phi-4 int8-KV (ВДВОЕ меньше байт) НЕ быстрее fp16 (0.105 против 0.104 мс при 8192;"
    )
    print(
        "  0.175 против 0.177 при 16384), а int4 (вчетверо меньше) МЕДЛЕННЕЕ. Будь декод связан"
    )
    print(
        "  полосой, int8 дал бы ~x2. §24.2 даёт DRAM 62 %, а не 90+ %. Значит связывает ЗАДЕРЖКА"
    )
    print(
        "  при низкой занятости -- канала 'задержка' в модели НЕТ (см. п.5 'не разобрано')."
    )
    print(
        "  Вывод: на декоде вердикт этого инструмента НЕДЕЙСТВИТЕЛЕН; он для префилла."
    )

    print("\n" + "#" * 96)
    print(
        "# ЯКОРЬ E (НЕ ЗАКАЗАН, найден расчётом) -- порог диспетчера Sk>=16384 совпал с границей L2"
    )
    print("#" * 96)
    for S in (8192, 16384):
        a = analyze(base_cfg(B=1, H=8, Hkv=8, d=128, Sq=S, Sk=S, BQ=32, BK=128))
        b = analyze(base_cfg(B=1, H=8, Hkv=8, d=128, Sq=S, Sk=S, BQ=64, BK=128))
        print(
            f"  Sk={S:>6}: BQ=32 -> окно {a['win_live'] / 1024**2:5.2f} МБ "
            f"({'влезает' if a['fits'] else 'НЕ ВЛЕЗАЕТ'}), "
            f"BQ=64 -> окно {b['win_live'] / 1024**2:5.2f} МБ "
            f"({'влезает' if b['fits'] else 'НЕ ВЛЕЗАЕТ'})"
        )
    a8 = analyze(base_cfg(B=1, H=8, Hkv=8, d=128, Sq=8192, Sk=8192, BQ=32, BK=128))
    a16 = analyze(base_cfg(B=1, H=8, Hkv=8, d=128, Sq=16384, Sk=16384, BQ=32, BK=128))
    b16 = analyze(base_cfg(B=1, H=8, Hkv=8, d=128, Sq=16384, Sk=16384, BQ=64, BK=128))
    check(
        "расчётная граница BQ=32 лежит между 8192 и 16384 -- ровно там, где отгруженный "
        "диспетчер переключает плитку",
        a8["fits"] and (not a16["fits"]) and b16["fits"],
        "совпало" if (a8["fits"] and not a16["fits"] and b16["fits"]) else "не совпало",
    )
    print(
        "  ЭТО НЕ ДОКАЗАТЕЛЬСТВО ПРИЧИНЫ: порог 16384 был найден замером, совпадение с расчётной"
    )
    print(
        "  границей -- согласие двух независимых источников, а не вывод одного из другого."
    )

    print("\n" + "=" * 96)
    print(
        "ИТОГ САМОПРОВЕРКИ: "
        + ("ВСЕ ЯКОРЯ СОШЛИСЬ" if ok else "ЕСТЬ ПРОВАЛЫ -- см. выше")
    )
    print("=" * 96)
    print(BLIND)
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description="калькулятор резидентности KV (sm_70)")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--scan", action="store_true")
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--Hkv", type=int, default=8)
    p.add_argument("--Sq", type=int, default=4096)
    p.add_argument("--Sk", type=int, default=4096)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--BQ", type=int)
    p.add_argument("--BK", type=int)
    p.add_argument(
        "--esz", type=int, default=2, help="байт на элемент KV (fp16=2, e4m3/int8=1)"
    )
    p.add_argument("--causal", action="store_true", default=True)
    p.add_argument(
        "--full", dest="causal", action="store_false", help="без причинной маски"
    )
    p.add_argument(
        "--window", type=int, default=None, help="скользящее окно (Gemma: 1024)"
    )
    p.add_argument(
        "--incoherent",
        action="store_true",
        help="маски резидентных CTA НЕ согласованы (разреженный отбор без группировки)",
    )
    p.add_argument("--sm", type=int, default=MACHINE["sm"])
    p.add_argument("--l2", type=float, default=MACHINE["l2"] / 1024**2, help="L2, МБ")
    p.add_argument(
        "--l2-bw",
        dest="l2_bw",
        type=float,
        default=MACHINE["l2_bw"] / 1e9,
        help="ГБ/с [ОЦЕНКА, НЕ ЗАМЕРЕНО]",
    )
    p.add_argument(
        "--tmeas",
        type=float,
        help="ЗАМЕРЕННОЕ время ядра, мкс -- включает проверку "
        "'в упоре ли канал'; без него вердикт слабее",
    )
    p.add_argument("--ctas-per-sm", dest="ctas_per_sm", type=int)
    p.add_argument("--smem", type=int, help="байт разделяемой на CTA")
    p.add_argument("--regs", type=int)
    a = p.parse_args()
    a.Sq_set = a.Sk_set = a.B_set = a.H_set = None
    for i, tok in enumerate(sys.argv):
        if tok == "--Sq":
            a.Sq_set = a.Sq
        if tok == "--Sk":
            a.Sk_set = a.Sk
        if tok == "--B":
            a.B_set = a.B
        if tok == "--H":
            a.H_set = a.H

    if a.selftest:
        return selftest()
    if a.scan:
        scan(a)
        print(BLIND)
        return 0
    cfg = make_cfg(a)
    print(render(analyze(cfg)))
    print(BLIND)
    return 0


if __name__ == "__main__":
    sys.exit(main())
