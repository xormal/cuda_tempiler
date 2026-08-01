# АУДИТ КОНФЛИКТНОСТИ БАНКОВ ПО ВСЕМ ОТГРУЖЕННЫМ ЯДРАМ.
#
# ЗАЧЕМ. Замерено (tempo, ncu 17/17): цена разделяемой памяти = ВАЙВФРОНТЫ, и при неизменном объёме
# данных одна только раскладка двигает её в 32 раза. Дополнение до некратности 32 словам стоит
# четыре байта на строку и не меняет ответ ни на бит. Конфликтность банков мы не аудировали НИ РАЗУ.
#
# ЧТО МЕРИМ. Счётчик конфликтов отделяет «вайвфронты из-за ШИРИНЫ» (неустранимо: LDS.128 стоит >=2,
# обратный путь отдаёт 8 Б на полосу за такт) от «вайвфронтов из-за БАНКОВ» (устранимо дополнением):
#
#     вайвфронты = идеальные + конфликты      ->     доля_возвратимого = конфликты / вайвфронты
#
# Это ПРЯМОЕ разложение, а не расчёт: на Volta счётчик конфликтов считает именно ДОБАВОЧНЫЕ
# вайвфронты, вызванные конфликтом.
#
# КАК ЧИТАТЬ. Доля возвратимого - это верхняя оценка выигрыша ПО ТРАФИКУ разделяемой памяти, а не по
# времени ядра. Во время она переходит настолько, насколько разделяемая память связывает: на
# volta_fwd внутри фаз умножения она загружена на 76-85%, на декоде - неизвестно, для того и меряем.
import os
import subprocess
import sys

# ИМЕННО ЭТОТ путь, а не .../bin/ncu: тот launcher отвечает "Nsight Compute is not installed"
# и даёт ПУСТОЙ вывод, который таблица читала как "конфликтов нет". 2025.x на Volta не работает.
NCU = (
    "/opt/conda/miniconda3/pkgs/nsight-compute-2024.1.1.4-0/nsight-compute/2024.1.1/ncu"
)
PY = "/opt/conda/miniconda3/envs/vllm/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))

METRICS = ",".join(
    [
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    ]
)
# ЧУЖИЕ ЯДРА (torch): их в профиле большинство, и они забивают таблицу нулями.
OURS = (
    "split_defer",
    "reduce_defer",
    "flash",
    "fmha",
    "attn",
    "volta",
    "fwd",
    "bwd",
    "kernel_",
)
SKIP = (
    "at::native",
    "at::detail",
    "cutlass::Kernel",
    "elementwise",
    "reduce_kernel",
    "vectorized",
)


def workload(which):
    """Один вызов боевого ядра.

    ФОРМЫ МАЛЕНЬКИЕ НАМЕРЕННО. Мерим ДОЛЮ конфликтов в трафике разделяемой памяти -- она свойство
    РАСКЛАДКИ, а не размера задачи (раскладка плитки от S и B не зависит). Боевые формы под ncu
    означают десятки минут на replay каждого ядра и ничего не добавляют к ответу.
    """
    import torch
    import fa2_sm70 as F

    dev = "cuda"
    torch.manual_seed(0)
    if which == "fwd128":  # cutlass-мейнлуп, d<=128
        B, H, S, D = 1, 2, 512, 128
        q = torch.randn(B, S, H, D, device=dev, dtype=torch.float16)
        k = torch.randn(B, S, H, D, device=dev, dtype=torch.float16)
        v = torch.randn(B, S, H, D, device=dev, dtype=torch.float16)
        F.flash_attn_func(q, k, v, causal=True)
    elif which == "fwd256":  # рукописный мейнлуп Volta, d=256
        B, H, S, D = 1, 2, 512, 256
        q = torch.randn(B, S, H, D, device=dev, dtype=torch.float16)
        k = torch.randn(B, S, H, D, device=dev, dtype=torch.float16)
        v = torch.randn(B, S, H, D, device=dev, dtype=torch.float16)
        F.flash_attn_func(q, k, v, causal=True)
    elif which.startswith("bytes"):  # ядро с разделением варпов по ролям, байтовый KV
        fmt = int(which[5:])  # публично отгружен только формат 8
        B, H, Hkv, S, D = 1, 4, 1, 512, 256
        q = torch.randn(B, S, H, D, device=dev, dtype=torch.float16)
        k = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.float16)
        v = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.float16)
        cache = F.quantize_kv_prefill(k, v, fmt=fmt)
        F.prefill_bytes(q, cache, causal=True)
    elif which == "decode":  # split-KV декод, СВОЯ точка входа (не flash_attn_func:
        B, H, Hkv, S, D = (
            2,
            4,
            1,
            4096,
            128,
        )  # та при Sq=1 уходит в префилл и отвергает GQA)
        q = torch.randn(B, H, 1, D, device=dev, dtype=torch.float16)
        k = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.float16)
        v = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.float16)
        F.decode(q, k, v)
    elif which == "bwd":  # backward d=128
        B, H, S, D = 1, 2, 512, 128
        q = torch.randn(B, S, H, D, device=dev, dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, S, H, D, device=dev, dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, S, H, D, device=dev, dtype=torch.float16, requires_grad=True)
        o = F.flash_attn_func(q, k, v, causal=True)
        o.backward(torch.randn_like(o))
    else:
        raise SystemExit(f"неизвестное тело: {which}")
    torch.cuda.synchronize()


def parse(csv_text):
    """ncu --csv: одна строка на (ядро, метрика). Складываем по ядрам."""
    import csv
    import io

    rows = list(csv.reader(io.StringIO(csv_text)))
    hdr = None
    out = {}
    for r in rows:
        if not r:
            continue
        if hdr is None:
            if "Metric Name" in r:
                hdr = {n: i for i, n in enumerate(r)}
            continue
        try:
            kern = r[hdr["Kernel Name"]]
            name = r[hdr["Metric Name"]]
            # РАЗДЕЛИТЕЛЬ ТЫСЯЧ У ncu -- НЕРАЗРЫВНЫЙ ПРОБЕЛ (локаль ru): "5 238". Снятие
            # обычного пробела его не берёт, float() падает, строка молча выбрасывается --
            # и проходят ТОЛЬКО значения МЕНЬШЕ ТЫСЯЧИ, то есть чужие мелкие ядра.
            # Таблица при этом выглядит полной и врёт.
            raw = r[hdr["Metric Value"]]
            val = "".join(c for c in raw if c.isdigit() or c in ".-eE+")
            out.setdefault(kern, {})[name] = float(val)
        except (KeyError, IndexError, ValueError):
            continue
    return out


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--run":
        workload(sys.argv[2])
        return 0

    bodies = sys.argv[1:] or ["fwd128", "fwd256", "bytes0", "bytes8", "decode", "bwd"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", LC_ALL="C", LANG="C")
    print(
        f"{'тело':>9} | {'ядро':<34} | {'вайвфр LD':>10} | {'конфл LD':>9} |"
        f" {'вайвфр/ком':>10} | {'ВОЗВРАТИМО':>10}"
    )
    print("-" * 104)
    for b in bodies:
        cmd = [
            NCU,
            "--csv",
            "--target-processes",
            "all",
            "--metrics",
            METRICS,
            "--kernel-name-base",
            "demangled",
            PY,
            os.path.abspath(__file__),
            "--run",
            b,
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
        got = parse(p.stdout)
        if not got:
            # МОЛЧАЛИВАЯ ПУСТАЯ ТАБЛИЦА -- ХУЖЕ ОШИБКИ: она читается как "конфликтов нет".
            tail = [l for l in p.stderr.strip().splitlines() if l.strip()][-2:]
            print(
                f"{b:>9} | НЕТ ДАННЫХ (rc={p.returncode}, строк CSV {len(p.stdout.splitlines())}): "
                f"{' | '.join(t[:60] for t in tail)}"
            )
            continue
        for kern, m in got.items():
            if any(x in kern for x in SKIP):
                continue
            wf = m.get(
                "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", 0.0
            ) + m.get("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum", 0.0)
            cf = m.get(
                "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", 0.0
            ) + m.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", 0.0)
            if wf < 1.0:  # ядро вовсе не трогает разделяемую память
                continue
            print(
                f"{b:>9} | {kern[:34]:<34} | {wf:10.3e} | {cf:9.3e} |"
                f" {'':>10} | {100.0 * cf / wf:9.1f} %"
            )
    print(
        "\nВОЗВРАТИМО = конфликты/вайвфронты: доля трафика разделяемой памяти, которую снимает"
    )
    print(
        "ДОПОЛНЕНИЕ РАСКЛАДКИ, не трогая ни одной команды и не меняя ответ. Остаток - ширина,"
    )
    print("он неустраним (обратный путь отдаёт 8 Б на полосу за такт).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
