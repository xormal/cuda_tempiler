# ФАЛЬСИФИКАТОР ЛИНТЕРА НА ЖЕЛЕЗЕ. Линтер утверждает: декод ЧИСТ при d=128 (EPT=4) и КОНФЛИКТУЕТ
# ВДВОЕ при d=256 (EPT=8) -- тот же самый код, разный шаг полосы. Метрика ОБЯЗАНА сдвинуться.
# Если не сдвинулась -- модель вырождена, и все её вердикты недействительны (см. документацию, §3).
# Считаются только СЧЁТЧИКИ (замеры времени запрещены -- карта делится с другим нарядом).
import os, subprocess, sys

# КАТАЛОГ СКРИПТА УХОДИТ ИЗ sys.path ПЕРВЫМ ДЕЙСТВИЕМ. В tools/ лежит чужой timeit.py, и он
# перекрывает стандартный: torch падает на `from timeit import default_timer` ещё до первой строки
# полезной работы, а ncu отдаёт ПУСТУЮ таблицу, которая читается как «конфликтов нет».
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") != _here]
NCU = (
    "/opt/conda/miniconda3/pkgs/nsight-compute-2024.1.1.4-0/nsight-compute/2024.1.1/ncu"
)
PY = "/opt/conda/miniconda3/envs/vllm/bin/python"
M = ",".join(
    [
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    ]
)


def run(d):
    import torch, fa2_sm70 as F

    B, H, Hkv, S = 2, 4, 1, 4096
    q = torch.randn(B, H, 1, d, device="cuda", dtype=torch.float16)
    k = torch.randn(B, Hkv, S, d, device="cuda", dtype=torch.float16)
    v = torch.randn(B, Hkv, S, d, device="cuda", dtype=torch.float16)
    F.decode(q, k, v)
    torch.cuda.synchronize()


def parse(txt):
    import csv, io

    rows = list(csv.reader(io.StringIO(txt)))
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
            raw = r[hdr["Metric Value"]]
            val = "".join(
                c for c in raw if c.isdigit() or c in ".-eE+"
            )  # локаль: НЕРАЗРЫВНЫЙ пробел
            out.setdefault(kern, {})[name] = float(val)
        except Exception:
            continue
    return out


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--run":
        run(int(sys.argv[2]))
        raise SystemExit(0)
    # ncu на Volta требует sudo (счётчики), а sudo теряет PATH -- и torch падает на «Ninja is
    # required to load C++ extensions», после чего ncu отдаёт ПУСТУЮ таблицу. Поэтому PATH и все
    # переменные сборки передаются ЯВНО через `sudo env`, как и в уроке про tmux.
    keep = [
        "PATH",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "CC",
        "CXX",
        "CUDAHOSTCXX",
        "TORCH_CUDA_ARCH_LIST",
        "PYTHONPATH",
        "FA2SM70_BUILD_DIR",
        "HOME",
        "LD_LIBRARY_PATH",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", LC_ALL="C", LANG="C")
    pre = ["sudo", "-S", "-p", "", "env", "LC_ALL=C", "LANG=C"] + [
        f"{k}={env[k]}" for k in keep if env.get(k)
    ]
    print(
        f"{'d':>5} | {'ядро':<28} | {'вайвфр':>10} | {'конфл':>10} | {'доля':>7} | линтер"
    )
    for d in (128, 256):
        p = subprocess.run(
            pre
            + [
                NCU,
                "--csv",
                "--target-processes",
                "all",
                "--metrics",
                M,
                "--kernel-name-base",
                "demangled",
                PY,
                os.path.abspath(__file__),
                "--run",
                str(d),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=2400,
            input=os.environ.get("SUDO_PW", "") + "\n",
        )
        got = parse(p.stdout)
        if not got:
            print(
                f"{d:>5} | НЕТ ДАННЫХ rc={p.returncode}: {p.stderr.strip().splitlines()[-1][:70] if p.stderr.strip() else ''}"
            )
            continue
        for kern, m in got.items():
            if not any(x in kern for x in ("defer", "volta", "attn", "fmha")):
                continue
            wf = m.get(
                "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", 0
            ) + m.get("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum", 0)
            cf = m.get(
                "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", 0
            ) + m.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", 0)
            if wf < 1:
                continue
            pred = "ЧИСТО (EPT=4)" if d == 128 else "КОНФЛИКТ x2 (EPT=8)"
            print(
                f"{d:>5} | {kern[:28]:<28} | {wf:10.3e} | {cf:10.3e} | {100 * cf / wf:6.1f} % | {pred}"
            )
