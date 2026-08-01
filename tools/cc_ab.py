# -*- coding: utf-8 -*-
"""КОМПИЛЯЦИОННЫЙ A/B: регистры, разлив, разделяемая, занятость -- НОЛЬ ТАКТОВ GPU.

ЗАЧЕМ. Половина вопросов про ядро решается без карты: сколько регистров, есть ли разлив, влезает
ли в разделяемую, не появился ли кадр стека. Рецепт (`nvcc -cubin` + `cuobjdump -res-usage` +
поиск LDL/STL в SASS) лежал текстом в журнале, а не инструментом, и поэтому применялся через раз.
Правило проекта прямое: ПОСЛЕ КАЖДОЙ СБОРКИ проверять кадр стека и SASS на LDL/STL, потому что
ОТЧЁТ КОМПИЛЯТОРА О РАЗЛИВЕ НЕПОЛОН (замерено: кадр стека при нуле разливов по отчёту).

ЧТО МЕРЯЕТ (единицы):
  * регистров на поток            -- штук          (cuobjdump -res-usage: REG)
  * кадр стека                    -- байт          (res-usage: STACK; ptxas -v: "bytes stack frame")
  * разлив по отчёту компилятора  -- байт          (ptxas -v: spill stores / spill loads)
  * LDL/STL В ТЕЛЕ                -- штук команд   (разбор SASS; пролог и циклы -- отдельными колонками)
  * статическая разделяемая       -- байт          (res-usage: SHARED)
  * динамическая разделяемая      -- байт          (sizeof(SharedStorage) через компиляционный зонд;
                                                    в cubin её НЕТ -- это параметр запуска)
  * занятость                     -- варпов на SM  (floor(65536/regs/32), плюс поправка на
                                                    гранулярность выделения 256 регистров на варп)

ЧЕГО НЕ УМЕЕТ (читать обязательно, см. также раздел "НЕ РАЗОБРАНО" в выводе):
  * НЕ меряет время. Ни одного такта GPU не тратится и ни одного не предсказывается.
  * НЕ видит, СКОЛЬКО РАЗ исполнится LDL/STL. Один разлив в 200-итерационном цикле хуже сорока
    в прологе, а инструмент печатает штуки. Колонка "в циклах" -- максимум, что даёт статика.
  * НЕ знает динамическую разделяемую из cubin. Она приходит зондом sizeof(...), то есть только
    для тех типов, которые ему назвали; если тип не назван -- строка пустая, а не "0".
  * НЕ различает конфликтность банков (это `smem_lint.py`) и не считает трафик.
  * НЕ знает MaxLive. Из одной сборки видно только REG -- то, сколько ptxas ВЗЯЛ при данном
    бюджете, а это верхняя оценка потребности. Порог берётся перекомпиляцией (`regsweep`).

ЗАМЕРЕННЫЕ ОСОБЕННОСТИ ptxas НА ЭТОЙ МАШИНЕ (встроены в вердикт):
  1. БЕЗ ОБЪЯВЛЕННОЙ ЗАНЯТОСТИ (`__launch_bounds__` со вторым аргументом) ptxas НЕ РАЗЛИВАЕТ
     ВООБЩЕ: он берёт столько регистров, сколько хочет, и вместо замедления вы получаете отказ
     запуска. Значит "разлива нет" В ТАКОЙ СБОРКЕ НЕ ЗНАЧИТ "влезает": цену регистров видно
     только там, где бюджет сообщён компилятору. Инструмент ищет `__launch_bounds__` в исходнике
     и, не найдя, ставит пометку НЕТ-БЮДЖЕТА и снимает доверие к вердикту "влезает".
  2. Порог разлива: R_required = MaxLive + 7 (совпал с ptxas 4/4, data/knee_fit.txt).
     ВНИМАНИЕ, ЗАМЕРЕННАЯ ПОПРАВКА (data/ccab_regsweep*.txt): при нулевом разливе REG -- это
     ВЕРХНЯЯ оценка MaxLive, а не сама MaxLive. Боевое ядро bwd<128,64,256,noESK> держит 173
     регистра при любом бюджете 176..183 и НЕ разливает даже при 176, хотя REG+7 = 180. То есть
     ptxas берёт больше регистров, чем нужно, когда бюджет позволяет. Поэтому "бюджет без
     разлива >= REG+7" печатается как ГРАНИЦА СВЕРХУ, а истинный порог ищется перекомпиляцией
     (команда `regsweep`), а не выводится из одной сборки.
  3. Q(W) = min(255, 8*floor(256/W)) -- бюджет регистров при W варпах на SM.
  4. Кадр стека бывает НЕНУЛЕВЫМ при НУЛЕ разливов по отчёту компилятора -- поэтому STACK и
     spill печатаются РАЗНЫМИ колонками и никогда не складываются.

ЗАПУСК:
  python3 tools/cc_ab.py qtable
  python3 tools/cc_ab.py selftest                      # оба якоря
  python3 tools/cc_ab.py compile --src F.cu -D FOO=1 [--profile fa2] [--filter RE] [--json OUT]
  python3 tools/cc_ab.py ab --src F.cu --a "" --b "-DFOO=1"
  python3 tools/cc_ab.py sizeof --pre-file P.txt --expr "sizeof(T)" [--expr ...]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time

# --------------------------------------------------------------------------------------------
# 0. Железо sm_70 (константы, не подгоняемые)
# --------------------------------------------------------------------------------------------
REGFILE_PER_SM = 65536  # регистров (слов) на SM
REG_ALLOC_GRAN = 256  # гранулярность выделения регистров на варп
WARPS_PER_SM_MAX = 64
BLOCKS_PER_SM_MAX = 32
SMEM_PER_BLOCK_MAX = (
    96 * 1024
)  # байт, sm_70, с opt-in cudaFuncAttributeMaxDynamicSharedMemorySize
SMEM_PER_BLOCK_DEFAULT = 48 * 1024
SMEM_PER_SM = 96 * 1024
REG_ISA_MAX = 255
SPILL_KNEE = 7  # R_required = MaxLive + 7 (замерено, data/knee_fit.txt)

CUDA_HOME_DEFAULT = "/opt/conda/miniconda3/envs/cuda128"
REPO_DEFAULT = "../VLLM_fa2/solutions/fa2_sm70_cutlass_grade"


def q_budget(warps: int) -> int:
    """Бюджет регистров на поток при `warps` варпах на SM.  Q(W)=min(255, 8*floor(256/W))."""
    if warps <= 0:
        raise ValueError("warps must be >= 1")
    return min(REG_ISA_MAX, 8 * (256 // warps))


def warps_from_regs_naive(regs: int) -> int:
    """Занятость по формуле наряда: floor(65536 / regs / 32) варпов."""
    if regs <= 0:
        return WARPS_PER_SM_MAX
    return min(WARPS_PER_SM_MAX, REGFILE_PER_SM // regs // 32)


def warps_from_regs_gran(regs: int) -> int:
    """То же с поправкой на гранулярность: варп занимает ceil(regs*32/256)*256 регистров."""
    if regs <= 0:
        return WARPS_PER_SM_MAX
    per_warp = -(-(regs * 32) // REG_ALLOC_GRAN) * REG_ALLOC_GRAN
    return min(WARPS_PER_SM_MAX, REGFILE_PER_SM // per_warp)


def max_warps_no_spill(regs: int) -> int:
    """Максимум варпов, при котором ptxas ещё не обязан разливать: max{W : Q(W) >= regs+7}."""
    need = regs + SPILL_KNEE
    best = 0
    for w in range(1, WARPS_PER_SM_MAX + 1):
        if q_budget(w) >= need:
            best = w
    return best


# --------------------------------------------------------------------------------------------
# 1. Профили сборки
# --------------------------------------------------------------------------------------------
def _first_existing(paths):
    for p in paths:
        if os.path.isdir(p):
            return p
    return None


def profile_fa2(repo=REPO_DEFAULT, cuda_home=CUDA_HOME_DEFAULT):
    """Профиль боевой сборки backward/prefill (снят с build.ninja торчевого JIT)."""
    torch_inc = _first_existing(
        [
            "/opt/conda/miniconda3/envs/py311/lib/python3.11/site-packages/torch/include",
            "/opt/conda/miniconda3/envs/vllm/lib/python3.11/site-packages/torch/include",
        ]
    )
    inc = [
        os.path.join(repo, "fa2_src/cutlass/include"),
        os.path.join(repo, "fa2_src/cutlass/tools/util/include"),
        os.path.join(repo, "fa2_src/fmha_kernel"),
    ]
    sysinc = []
    if torch_inc:
        sysinc += [torch_inc, os.path.join(torch_inc, "torch/csrc/api/include")]
    sysinc += [os.path.join(cuda_home, "include")]
    flags = [
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr",
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "-DHAS_PYTORCH",
    ]
    return {
        "name": "fa2",
        "inc": inc,
        "sysinc": sysinc,
        "flags": flags,
        "cuda_home": cuda_home,
        "repo": repo,
        "missing_torch_include": torch_inc is None,
    }


def profile_bare(repo=REPO_DEFAULT, cuda_home=CUDA_HOME_DEFAULT):
    """Профиль без torch и без cutlass -- для самодостаточных ядер (наши volta_*.cu)."""
    return {
        "name": "bare",
        "inc": [os.path.join(repo, "fa2_src/fmha_kernel")],
        "sysinc": [os.path.join(cuda_home, "include")],
        "flags": [
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "-D__CUDA_NO_HALF_OPERATORS__",
        ],
        "cuda_home": cuda_home,
        "repo": repo,
        "missing_torch_include": False,
    }


PROFILES = {"fa2": profile_fa2, "bare": profile_bare}


def nvcc(prof):
    return os.path.join(prof["cuda_home"], "bin", "nvcc")


def cuobjdump(prof):
    return os.path.join(prof["cuda_home"], "bin", "cuobjdump")


def _inc_args(prof):
    a = []
    for p in prof["inc"]:
        a += ["-I" + p]
    for p in prof["sysinc"]:
        a += ["-isystem", p]
    return a


def run(cmd, timeout=3600):
    t0 = time.time()
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=dict(os.environ, LC_ALL="C", LANG="C"),
    )
    return {
        "rc": p.returncode,
        "out": p.stdout,
        "err": p.stderr,
        "sec": time.time() - t0,
        "cmd": " ".join(shlex.quote(c) for c in cmd),
    }


# --------------------------------------------------------------------------------------------
# 2. Компиляция в cubin
# --------------------------------------------------------------------------------------------
def compile_cubin(src, defines, prof, outdir, arch="sm_70", extra=None):
    os.makedirs(outdir, exist_ok=True)
    cubin = os.path.join(outdir, os.path.basename(src) + ".cubin")
    cmd = [
        nvcc(prof),
        "-ccbin",
        "/usr/bin/gcc",
        "-cubin",
        "-o",
        cubin,
        "-gencode",
        "arch=compute_%s,code=%s" % (arch.split("_")[1], arch),
        "-Xptxas",
        "-v",
        "-Wno-deprecated-gpu-targets",
    ]
    cmd += _inc_args(prof) + list(prof["flags"])
    cmd += ["-D" + d for d in defines]
    if extra:
        cmd += extra
    cmd += [src]
    r = run(cmd)
    r["cubin"] = cubin if (r["rc"] == 0 and os.path.exists(cubin)) else None
    return r


# --------------------------------------------------------------------------------------------
# 3. Разбор ptxas -v  (ОТЧЁТ КОМПИЛЯТОРА -- заведомо неполный, сверяется с SASS)
# --------------------------------------------------------------------------------------------
PTXAS_ENTRY = re.compile(
    r"ptxas info\s*:\s*Compiling entry function '([^']+)' for '([^']+)'"
)
PTXAS_FUNC = re.compile(
    r"ptxas info\s*:\s*Function properties for (?:')?([^\s']+)(?:')?\s*\n"
    r"\s*(\d+)\s+bytes stack frame,\s*(\d+)\s+bytes spill stores,\s*(\d+)\s+bytes spill loads"
)
PTXAS_USED = re.compile(
    r"ptxas info\s*:\s*Used\s+(\d+)\s+registers"
    r"(?:.*?(\d+)\s+bytes smem)?",
    re.S,
)


def parse_ptxas(log):
    """Отчёт ptxas -v -> {mangled: {...}} + список неразобранных строк 'ptxas info'."""
    res, unparsed = {}, []
    lines = log.splitlines()
    cur = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = PTXAS_ENTRY.search(ln)
        if m:
            cur = m.group(1)
            res.setdefault(
                cur,
                {
                    "arch": m.group(2),
                    "stack": None,
                    "spill_st": None,
                    "spill_ld": None,
                    "regs": None,
                    "smem_static": None,
                },
            )
            i += 1
            continue
        m = re.search(r"Function properties for (?:')?([^\s']+)", ln)
        if m:
            name = m.group(1)
            body = " ".join(lines[i : i + 3])
            mm = re.search(
                r"(\d+)\s+bytes stack frame,\s*(\d+)\s+bytes spill stores,"
                r"\s*(\d+)\s+bytes spill loads",
                body,
            )
            if mm:
                d = res.setdefault(
                    name,
                    {
                        "arch": None,
                        "stack": None,
                        "spill_st": None,
                        "spill_ld": None,
                        "regs": None,
                        "smem_static": None,
                    },
                )
                d["stack"] = int(mm.group(1))
                d["spill_st"] = int(mm.group(2))
                d["spill_ld"] = int(mm.group(3))
                if name == cur:
                    pass
            else:
                unparsed.append(ln.strip())
            i += 1
            continue
        m = re.search(r"Used\s+(\d+)\s+registers", ln)
        if m and cur:
            res[cur]["regs"] = int(m.group(1))
            ms = re.search(r"(\d+)\s+bytes smem", ln)
            if ms:
                res[cur]["smem_static"] = int(ms.group(1))
            i += 1
            continue
        if (
            "ptxas info" in ln
            and "Compiling entry" not in ln
            and "Function properties" not in ln
            and "Used" not in ln
            and "Compile time" not in ln
            and "bytes gmem" not in ln
            and "bytes cmem" not in ln
        ):
            unparsed.append(ln.strip())
        i += 1
    return res, unparsed


# --------------------------------------------------------------------------------------------
# 4. Разбор cuobjdump -res-usage
# --------------------------------------------------------------------------------------------
RES_FUNC = re.compile(r"^\s*Function\s+(\S+)\s*:\s*$")
RES_LINE = re.compile(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)")


def parse_res_usage(text):
    """cuobjdump -res-usage -> {mangled: {...}} + неразобранные строки."""
    res, unparsed = {}, []
    cur = None
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        m = RES_FUNC.match(ln)
        if m:
            cur = m.group(1)
            continue
        m = RES_LINE.search(ln)
        if m:
            if cur is None:
                unparsed.append("ресурсы без имени функции: " + s)
                continue
            res[cur] = {
                "reg": int(m.group(1)),
                "stack": int(m.group(2)),
                "shared": int(m.group(3)),
                "local": int(m.group(4)),
            }
            cur = None
            continue
        if s.startswith(
            (
                "Fatbin",
                "===",
                "arch =",
                "code version",
                "host =",
                "compile_size",
                "identifier =",
                "producer =",
                "ptxasOptions",
                "compressed",
                "Resource usage:",
                "Common:",
                "GLOBAL:",
                "..........",
            )
        ):
            continue
        unparsed.append(s)
    return res, unparsed


# --------------------------------------------------------------------------------------------
# 5. Разбор SASS: LDL/STL, пролог, циклы
# --------------------------------------------------------------------------------------------
SASS_FUNC = re.compile(r"^\s*(?:Function|\.text\.)\s*:?\s*(\S+)\s*:?\s*$")
SASS_FUNC2 = re.compile(r"^\s*Function : (\S+)")
SASS_INSN = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(.*?);")
BRANCH_OPS = (
    "BRA",
    "BRX",
    "JMP",
    "JMX",
    "CAL",
    "RET",
    "EXIT",
    "SSY",
    "BSSY",
    "BSYNC",
    "PBK",
    "BRK",
    "PCNT",
    "CONT",
    "PRET",
    "SYNC",
    "NANOSLEEP",
)
BRA_TARGET = re.compile(r"\bBRA\b[^;]*?0x([0-9a-fA-F]+)")


def parse_sass(text):
    """SASS -> {mangled: {insns, ldl, stl, ...}} + счётчик нераспознанных строк."""
    funcs = {}
    cur = None
    unrecognised = 0
    body_lines = 0
    for ln in text.splitlines():
        m = SASS_FUNC2.match(ln) or SASS_FUNC.match(ln)
        if m and ("Function" in ln or ".text." in ln):
            cur = m.group(1)
            funcs[cur] = []
            continue
        m = SASS_INSN.match(ln)
        if m:
            if cur is None:
                unrecognised += 1
                continue
            addr = int(m.group(1), 16)
            txt = m.group(2).strip()
            funcs[cur].append((addr, txt))
            body_lines += 1
            continue
        s = ln.strip()
        if not s:
            continue
        if s.startswith(
            (
                "/*",
                "code for",
                "Fatbin",
                "====",
                "arch =",
                "code version",
                "host =",
                "compile_size",
                "identifier",
                "producer",
                "compressed",
                ".headerflags",
                "//",
                "Function",
                "..........",
            )
        ):
            continue
        unrecognised += 1
    out = {}
    for name, insns in funcs.items():
        out[name] = analyse_func(insns)
    return out, unrecognised, body_lines


def analyse_func(insns):
    """Классификация LDL/STL: пролог / тело / внутри циклов (обратные переходы)."""
    addrs = [a for a, _ in insns]
    aset = set(addrs)
    # пролог = всё до ПЕРВОЙ команды передачи управления
    prologue_end = len(insns)
    for i, (_, t) in enumerate(insns):
        op = t.split()[0].lstrip("@!").split(".")[0]
        if t.startswith("@"):
            parts = t.split()
            op = parts[1].split(".")[0] if len(parts) > 1 else op
        if op in BRANCH_OPS:
            prologue_end = i
            break
    # циклы = диапазоны [target, addr] для BRA с целью НАЗАД
    loops = []
    bad_targets = 0
    for a, t in insns:
        if "BRA" not in t:
            continue
        m = BRA_TARGET.search(t)
        if not m:
            if re.search(r"\bBRA\b", t) and "0x" not in t:
                bad_targets += 1
            continue
        tgt = int(m.group(1), 16)
        if tgt <= a:
            if tgt not in aset:
                bad_targets += 1
            loops.append((tgt, a))

    def in_loop(a):
        return any(lo <= a <= hi for lo, hi in loops)

    ldl = stl = ldl_body = stl_body = ldl_loop = stl_loop = 0
    first_body_spill = None
    hmma = bar = 0
    for i, (a, t) in enumerate(insns):
        head = t.split()[0]
        if head.startswith("@"):
            head = t.split()[1] if len(t.split()) > 1 else head
        base = head.split(".")[0]
        if base == "LDL" or base == "STL":
            is_ldl = base == "LDL"
            if is_ldl:
                ldl += 1
            else:
                stl += 1
            if i >= prologue_end:
                if is_ldl:
                    ldl_body += 1
                else:
                    stl_body += 1
                if first_body_spill is None:
                    first_body_spill = a
            if in_loop(a):
                if is_ldl:
                    ldl_loop += 1
                else:
                    stl_loop += 1
        if base.startswith("HMMA"):
            hmma += 1
        if base == "BAR":
            bar += 1
    return {
        "n_insn": len(insns),
        "ldl": ldl,
        "stl": stl,
        "ldl_body": ldl_body,
        "stl_body": stl_body,
        "ldl_loop": ldl_loop,
        "stl_loop": stl_loop,
        "prologue_insn": prologue_end,
        "first_body_spill": first_body_spill,
        "n_loops": len(loops),
        "hmma": hmma,
        "bar": bar,
        "bad_branch_targets": bad_targets,
    }


def demangle(names, prof):
    if not names:
        return {}
    p = subprocess.run(
        ["c++filt"], input="\n".join(names), capture_output=True, text=True
    )
    if p.returncode != 0:
        return {}
    out = p.stdout.splitlines()
    if len(out) != len(names):
        return {}
    return dict(zip(names, out))


# --------------------------------------------------------------------------------------------
# 6. Зонд sizeof: динамическая разделяемая и любые compile-time константы
# --------------------------------------------------------------------------------------------
PROBE_ERR = re.compile(r"CCAB_SIZEOF<\s*(\d+)\s*U?L?L?\s*>")


def probe_constants(preamble, exprs, prof, outdir, arch="sm_70", extra=None):
    """Компиляционный зонд: печатает значения constexpr-выражений БЕЗ запуска и БЕЗ линковки.

    Механика: `template<unsigned long> struct CCAB_SIZEOF;` объявлен, но не определён; объявление
    объекта `CCAB_SIZEOF<expr> x;` даёт ошибку "incomplete type CCAB_SIZEOF<41232UL>", в которой
    значение напечатано. Ошибки независимы, поэтому за один проход снимаются все выражения.
    Выражение, не давшее ошибки нужного вида, попадает в НЕ РАЗОБРАНО, а НЕ в нули.
    """
    os.makedirs(outdir, exist_ok=True)
    src = os.path.join(outdir, "ccab_probe.cu")
    with open(src, "w") as f:
        f.write(preamble)
        f.write("\ntemplate<unsigned long CCAB_N> struct CCAB_SIZEOF;\n")
        for i, e in enumerate(exprs):
            f.write("CCAB_SIZEOF<(unsigned long)(%s)> ccab_probe_%d;\n" % (e, i))
    cmd = [
        nvcc(prof),
        "-ccbin",
        "/usr/bin/gcc",
        "-c",
        "-o",
        "/dev/null",
        "-gencode",
        "arch=compute_%s,code=%s" % (arch.split("_")[1], arch),
        "-Wno-deprecated-gpu-targets",
    ]
    cmd += _inc_args(prof) + list(prof["flags"])
    if extra:
        cmd += extra
    cmd += [src]
    r = run(cmd)
    log = r["out"] + r["err"]
    vals = {}
    unparsed = []
    for i, e in enumerate(exprs):
        got = None
        for ln_i, ln in enumerate(log.splitlines()):
            if "ccab_probe_%d;" % i in ln or re.search(r"ccab_probe_%d\b" % i, ln):
                ctx = "\n".join(log.splitlines()[max(0, ln_i - 3) : ln_i + 2])
                m = PROBE_ERR.search(ctx)
                if m:
                    got = int(m.group(1))
                    break
        if got is None:
            # запасной путь: n-я по счёту ошибка о неполном типе
            ms = PROBE_ERR.findall(log)
            if len(ms) == len(exprs):
                got = int(ms[i])
        if got is None:
            unparsed.append((e, "зонд не дал числа (см. лог компиляции)"))
        else:
            vals[e] = got
    r["values"] = vals
    r["unparsed"] = unparsed
    r["log"] = log
    r["src"] = src
    return r


# --------------------------------------------------------------------------------------------
# 7. Вердикт
# --------------------------------------------------------------------------------------------
def verdict(
    regs,
    stack,
    spill_st,
    spill_ld,
    ldl_body,
    stl_body,
    smem_total,
    threads=None,
    has_launch_bounds=True,
):
    flags = []
    spilled = bool(
        (stack or 0) > 0
        or (spill_st or 0) > 0
        or (spill_ld or 0) > 0
        or (ldl_body or 0) > 0
        or (stl_body or 0) > 0
    )
    if spilled:
        flags.append("РАЗЛИВ")
    if regs is not None and regs >= REG_ISA_MAX:
        flags.append("СТЕНА-255")
    if smem_total is not None and smem_total > SMEM_PER_BLOCK_MAX:
        flags.append("СТЕНА-СМЕМ")
    if threads is not None and regs is not None and threads * regs > REGFILE_PER_SM:
        flags.append("СТЕНА-БЛОК-НЕ-ЗАПУСТИТСЯ")
    if not flags:
        flags.append("ВЛЕЗАЕТ")
    if not has_launch_bounds:
        flags.append("НЕТ-БЮДЖЕТА")
    return "+".join(flags)


def occupancy(regs, threads, smem_total):
    """Варпов на SM: по регистрам (две формулы) и по разделяемой."""
    out = {
        "warps_reg_naive": warps_from_regs_naive(regs) if regs else None,
        "warps_reg_gran": warps_from_regs_gran(regs) if regs else None,
        "blocks_smem": None,
        "warps_smem": None,
        "warps_final": None,
    }
    if smem_total and smem_total > 0:
        out["blocks_smem"] = min(BLOCKS_PER_SM_MAX, SMEM_PER_SM // smem_total)
    if threads and regs:
        wpb = max(1, threads // 32)
        blocks_reg = REGFILE_PER_SM // (max(1, regs) * threads) if regs * threads else 0
        cand = [blocks_reg]
        if out["blocks_smem"] is not None:
            cand.append(out["blocks_smem"])
        blocks = min(cand) if cand else 0
        out["warps_smem"] = (
            out["blocks_smem"] * wpb if out["blocks_smem"] is not None else None
        )
        out["warps_final"] = min(WARPS_PER_SM_MAX, blocks * wpb)
        out["blocks_final"] = blocks
        out["warps_per_block"] = wpb
    return out


# --------------------------------------------------------------------------------------------
# 8. Сбор таблицы по одной сборке
# --------------------------------------------------------------------------------------------
def analyse_build(
    src,
    defines,
    prof,
    outdir,
    filt=None,
    arch="sm_70",
    extra=None,
    threads_hint=None,
    dyn_smem=None,
    reuse=None,
    log_text="",
):
    notes = []
    unparsed = []
    if reuse:
        r = {
            "cubin": reuse,
            "out": "",
            "err": log_text,
            "sec": 0.0,
            "cmd": "(готовый cubin)",
        }
        log = log_text
        if not log_text:
            unparsed.append(
                "отчёт ptxas -v недоступен (разбирается готовый cubin) -> "
                "колонка 'отчёт' пуста; это НЕ значит 'разлива нет по отчёту'"
            )
    else:
        r = compile_cubin(src, defines, prof, outdir, arch=arch, extra=extra)
        log = r["out"] + r["err"]
    if r["cubin"] is None:
        return {
            "ok": False,
            "compile": r,
            "log": log,
            "unparsed": ["КОМПИЛЯЦИЯ НЕ ПРОШЛА -- ниже НИЧЕГО не разобрано"],
        }
    ptx, ptx_unparsed = parse_ptxas(log)
    unparsed += ["ptxas -v: " + u for u in ptx_unparsed]

    ru = run([cuobjdump(prof), "-res-usage", r["cubin"]])
    res, res_unparsed = parse_res_usage(ru["out"] + ru["err"])
    unparsed += ["res-usage: " + u for u in res_unparsed[:20]]
    if len(res_unparsed) > 20:
        unparsed.append("res-usage: ... ещё %d строк" % (len(res_unparsed) - 20))

    sa = run([cuobjdump(prof), "-sass", r["cubin"]])
    sass_txt = sa["out"]
    sass, sass_unrec, sass_insns = parse_sass(sass_txt)
    if sass_unrec:
        unparsed.append(
            "SASS: %d строк не распознано как команда/заголовок" % sass_unrec
        )

    # объявленная занятость: ищем __launch_bounds__ ПОИМЕННО (исходник + локальные заголовки)
    lb_files, lb_names, lb_found = launch_bounds_scan(src, prof)
    if not lb_found:
        notes.append(
            "__launch_bounds__ НЕ НАЙДЕН НИ У ОДНОГО ядра в %d просмотренных файлах -> "
            "ptxas выбирал число регистров САМ и РАЗЛИВАТЬ НЕ ОБЯЗАН: "
            "'разлива нет' здесь НЕ доказывает, что ядро влезает." % len(lb_files)
        )

    names = sorted(set(list(res.keys()) + list(sass.keys()) + list(ptx.keys())))
    dm = demangle(names, prof)
    if not dm:
        unparsed.append("c++filt: демангл не выполнен (имена печатаются искажёнными)")

    rows = []
    for n in names:
        pretty = dm.get(n, n)
        if filt and not (re.search(filt, n) or re.search(filt, pretty)):
            continue
        ru_e = res.get(n)
        sa_e = sass.get(n)
        px_e = ptx.get(n)
        if ru_e is None:
            unparsed.append(
                "нет res-usage для функции %s (есть SASS/ptxas)" % pretty[:80]
            )
        if sa_e is None:
            unparsed.append("нет SASS для функции %s" % pretty[:80])
        regs = (ru_e or {}).get("reg") or (px_e or {}).get("regs")
        stack = (ru_e or {}).get("stack")
        if stack is None:
            stack = (px_e or {}).get("stack")
        smem_static = (ru_e or {}).get("shared")
        threads = threads_hint(pretty) if callable(threads_hint) else threads_hint
        dyn = dyn_smem(pretty) if callable(dyn_smem) else dyn_smem
        smem_total = (
            (smem_static or 0) + (dyn or 0)
            if (smem_static is not None or dyn)
            else None
        )
        row = {
            "mangled": n,
            "name": pretty,
            "regs": regs,
            "stack": stack,
            "spill_st": (px_e or {}).get("spill_st"),
            "spill_ld": (px_e or {}).get("spill_ld"),
            "smem_static": smem_static,
            "smem_dyn": dyn,
            "smem_total": smem_total,
            "threads": threads,
            "n_insn": (sa_e or {}).get("n_insn"),
            "ldl": (sa_e or {}).get("ldl"),
            "stl": (sa_e or {}).get("stl"),
            "ldl_body": (sa_e or {}).get("ldl_body"),
            "stl_body": (sa_e or {}).get("stl_body"),
            "ldl_loop": (sa_e or {}).get("ldl_loop"),
            "stl_loop": (sa_e or {}).get("stl_loop"),
            "prologue_insn": (sa_e or {}).get("prologue_insn"),
            "n_loops": (sa_e or {}).get("n_loops"),
            "hmma": (sa_e or {}).get("hmma"),
        }
        row["occ"] = occupancy(regs, threads, smem_total)
        row["min_budget_no_spill"] = (regs + SPILL_KNEE) if regs else None
        row["max_warps_no_spill"] = max_warps_no_spill(regs) if regs else None
        row["has_lb"] = kernel_basename(pretty) in lb_names
        row["verdict"] = verdict(
            regs,
            stack,
            row["spill_st"],
            row["spill_ld"],
            row["ldl_body"],
            row["stl_body"],
            smem_total,
            threads,
            row["has_lb"],
        )
        # СВЕРКА: отчёт компилятора против SASS
        rep = (row["spill_st"] or 0) + (row["spill_ld"] or 0)
        sass_sp = (row["ldl_body"] or 0) + (row["stl_body"] or 0)
        if rep == 0 and sass_sp > 0:
            row["disagree"] = (
                "отчёт компилятора: разлива НЕТ; в SASS %d LDL/STL в теле" % sass_sp
            )
        elif rep == 0 and (stack or 0) > 0:
            row["disagree"] = "отчёт компилятора: разлива НЕТ; кадр стека %d Б" % stack
        else:
            row["disagree"] = None
        if (sa_e or {}).get("bad_branch_targets"):
            unparsed.append(
                "SASS %s: %d переходов с неразобранной целью (границы циклов неполны)"
                % (pretty[:60], sa_e["bad_branch_targets"])
            )
        rows.append(row)

    if filt and not rows:
        unparsed.append(
            "фильтр %r не выбрал НИ ОДНОЙ функции из %d найденных" % (filt, len(names))
        )

    # ядра, объявленные в исходнике, но отсутствующие в cubin
    declared = declared_kernels(src)
    if declared:
        missing = [d for d in declared if not any(d in (dm.get(n) or n) for n in names)]
        if missing:
            unparsed.append(
                "объявлены в исходнике, но НЕ найдены в cubin (шаблон не инстанцирован "
                "или имя не совпало): " + ", ".join(sorted(missing)[:10])
            )
    return {
        "ok": True,
        "rows": rows,
        "unparsed": unparsed,
        "notes": notes,
        "compile_sec": r["sec"],
        "cubin": r["cubin"],
        "log": log,
        "n_funcs_total": len(names),
        "sass_insns": sass_insns,
        "cmd": r["cmd"],
    }


DECL_GLOBAL = re.compile(
    r"__global__\s+(?:void\s+)?(?:__launch_bounds__\([^)]*\)\s*)?"
    r"(?:void\s+)?([A-Za-z_]\w*)\s*\("
)


def declared_kernels(src):
    try:
        txt = open(src, encoding="utf-8", errors="replace").read()
    except OSError:
        return []
    return sorted(set(DECL_GLOBAL.findall(txt)))


def launch_bounds_scan(src, prof):
    """Собираем ИМЕНА ядер, у которых объявлен __launch_bounds__ (исходник + НАШИ заголовки).

    Возвращает (файлы, множество имён с бюджетом, было ли хоть одно). Проверка ИМЕННАЯ, а не
    пофайловая: соседнее ядро с бюджетом ничего не говорит про это.
    """
    files = [src]
    for d in prof["inc"]:
        if "cutlass/include" in d or "util/include" in d:
            continue  # чужое дерево -- не наш бюджет
        for root, _dirs, names in os.walk(d):
            for nm in names:
                if nm.endswith((".h", ".cuh", ".hpp", ".cu")):
                    files.append(os.path.join(root, nm))
    pat = re.compile(
        r"__launch_bounds__\s*\([^)]*\)\s*(?:\w[\w:<>,\s*&]*?\s+)?"
        r"([A-Za-z_]\w*)\s*[(<]"
    )
    with_lb = set()
    scanned = []
    for f in files:
        try:
            txt = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        scanned.append(f)
        if "__launch_bounds__" not in txt:
            continue
        with_lb.update(pat.findall(txt))
    return scanned, with_lb, bool(with_lb)


BASENAME = re.compile(r"([A-Za-z_]\w*)\s*(?:<|\()")


def kernel_basename(pretty):
    """Имя ядра из демангленной подписи: последний идентификатор перед '<' или '('."""
    head = pretty.split("(")[0]
    head = re.sub(r"<.*", "", head)  # шаблонные аргументы отбросить
    head = head.strip().split()[-1] if head.strip() else pretty
    return head.split("::")[-1]


# --------------------------------------------------------------------------------------------
# 9. Печать
# --------------------------------------------------------------------------------------------
def shorten(name, width=52):
    # ядра backward различаются ТОЛЬКО шаблонными аргументами -- обрезание по длине сделало бы
    # все строки одинаковыми (то есть таблицу нечитаемой и молча вводящей в заблуждение)
    t = bwd_tile_from_name(name)
    if t and "attention_kernel_backward" in name:
        flags = re.search(
            r"Sm70,\s*(?:cutlass::)?half_t,\s*(true|false),\s*(true|false),\s*"
            r"(true|false),\s*\d+,\s*\d+,\s*\d+,\s*(true|false),\s*(true|false)",
            name,
        )
        sfx = ""
        if flags:
            sfx = ",drop" if flags.group(2) == "true" else ""
            sfx += ",алигн" if flags.group(4) == "true" else ""
            sfx += ",noESK" if flags.group(5) == "false" else ""
        return "bwd<%d,%d,%d%s>" % (t[0], t[1], t[2], sfx)
    n = re.sub(r"\bcutlass::", "", name)
    n = re.sub(r"\bvoid\s+", "", n)
    if len(n) <= width:
        return n
    return n[: width - 3] + "..."


def print_table(res, title):
    print("=" * 118)
    print(title)
    print("=" * 118)
    if not res.get("ok"):
        print("КОМПИЛЯЦИЯ НЕ ПРОШЛА. Последние строки лога:")
        for ln in res.get("log", "").splitlines()[-25:]:
            print("   " + ln)
        return
    hdr = "%-40s %4s %5s %9s %7s %7s %7s %8s %5s %s" % (
        "ядро",
        "рег",
        "стек",
        "отчёт Б",
        "LDL/STL",
        "в цикле",
        "смем-ст",
        "смем-дин",
        "варпы",
        "вердикт",
    )
    print(hdr)
    print("-" * 118)
    for r in sorted(res["rows"], key=lambda x: -(x["regs"] or 0)):
        occ = r["occ"]
        w = occ.get("warps_final")
        wn = occ.get("warps_reg_naive")
        rep = (
            "%s/%s" % (r["spill_st"], r["spill_ld"])
            if r["spill_st"] is not None
            else "?"
        )
        print(
            "%-40s %4s %5s %9s %3s/%-3s %3s/%-3s %7s %8s %2s|%-2s %s"
            % (
                shorten(r["name"], 40),
                r["regs"],
                r["stack"],
                rep,
                r["ldl_body"],
                r["stl_body"],
                r["ldl_loop"],
                r["stl_loop"],
                r["smem_static"],
                "" if r["smem_dyn"] is None else r["smem_dyn"],
                "" if wn is None else wn,
                "" if w is None else w,
                r["verdict"],
            )
        )
    print("-" * 118)
    print(
        "колонки: рег = регистров/поток; стек = кадр стека, Б; отчёт = разлив ПО ОТЧЁТУ ptxas "
        "(store/load, Б);"
    )
    print(
        "         LDL/STL = ШТУК КОМАНД в теле (после пролога); в цикле = из них внутри "
        "обратных переходов;"
    )
    print(
        "         варпы = floor(65536/рег/32) | с учётом смем и потоков блока. "
        "'отчёт 0/0' при ненулевых LDL/STL -- отчёт неполон."
    )
    for r in res["rows"]:
        if r.get("disagree"):
            print("  !! %s: %s" % (shorten(r["name"], 40), r["disagree"]))
        if r.get("regs"):
            spilled = "РАЗЛИВ" in r["verdict"]
            if spilled:
                print(
                    "     %s: разлив есть -> REG(%s) НЕ равно MaxLive, предсказание "
                    "'бюджет без разлива' НЕПРИМЕНИМО (нужна сборка с большим бюджетом)."
                    % (shorten(r["name"], 36), r["regs"])
                )
            else:
                w = r["max_warps_no_spill"] or 1
                print(
                    "     %s: MaxLive <= %d (REG -- ВЕРХНЯЯ оценка, ptxas берёт больше, чем "
                    "нужно, когда бюджет позволяет). Отсюда бюджет без разлива <= %d рег и "
                    "варпов >= %d (Q(%d)=%d) -- это ГРАНИЦА, а не значение; истинный порог "
                    "ищет `regsweep`."
                    % (
                        shorten(r["name"], 36),
                        r["regs"],
                        r["min_budget_no_spill"],
                        w,
                        w,
                        q_budget(w),
                    )
                )
            if not r.get("has_lb", True):
                print(
                    "     %s: у ЭТОГО ядра __launch_bounds__ НЕ найден -> бюджет компилятору "
                    "не сообщён, отсутствие разлива ничего не доказывает."
                    % shorten(r["name"], 36)
                )
    for n in res.get("notes", []):
        print("  ЗАМЕЧАНИЕ: " + n)
    print()
    print("НЕ РАЗОБРАНО (%d):" % len(res.get("unparsed", [])))
    if not res.get("unparsed"):
        print(
            "  -- пусто. Функций в cubin: %d, команд SASS разобрано: %d."
            % (res.get("n_funcs_total", 0), res.get("sass_insns", 0))
        )
    else:
        for u in res["unparsed"]:
            print("  * " + u)
    print(
        "ПУСТОЙ СПИСОК ПРОБЛЕМ ПРИ НЕПУСТОМ СПИСКЕ НЕРАЗОБРАННОГО НЕ ОЗНАЧАЕТ 'ЧИСТО'."
    )


# --------------------------------------------------------------------------------------------
# 10. Якоря / самопроверка
# --------------------------------------------------------------------------------------------
ANCHOR_Q = {12: 168, 16: 128, 24: 80, 32: 64}
ANCHOR_BWD = {"regs": 255, "smem_dyn": 41232, "threads": 128}

BWD_PROBE_PREAMBLE = """
#define HAS_PYTORCH 1
#include <kernel_backward.h>
template <int BI, int BJ, int MK, bool ALIGNED = false, bool ESK = true>
using CCAB_AK = AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t,
    /*kIsAligned*/ true, /*kApplyDropout*/ false, /*kPreload*/ false,
    BI, BJ, MK, ALIGNED, ESK>;
"""


def cmd_qtable(args):
    print(
        "ЯКОРЬ 1: Q(W) = min(255, 8*floor(256/W)) -- бюджет регистров при W варпах на SM"
    )
    print("         (совпал с ptxas 4/4, см. data/knee_fit.txt)")
    print()
    print("%6s %8s %10s %10s" % ("W", "Q(W)", "ожидалось", "сходится"))
    ok = True
    for w in sorted(ANCHOR_Q):
        got = q_budget(w)
        exp = ANCHOR_Q[w]
        good = got == exp
        ok &= good
        print("%6d %8d %10d %10s" % (w, got, exp, "ДА" if good else "НЕТ"))
    print()
    print("полная лестница:")
    for w in (1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64):
        print(
            "   W=%-3d Q=%-4d  порог MaxLive (Q-7) = %-4d  временной порог (Q-2) = %d"
            % (w, q_budget(w), q_budget(w) - 7, q_budget(w) - 2)
        )
    print()
    print("ЯКОРЬ 1: " + ("СОШЁЛСЯ 4/4" if ok else "НЕ СОШЁЛСЯ"))
    return 0 if ok else 1


def bwd_threads_from_name(pretty):
    """kNumThreads = 32 * kBI*kBJ/(32*32) = kBI*kBJ/32, вытащенное из демангленного имени."""
    t = bwd_tile_from_name(pretty)
    if not t:
        return None
    return t[0] * t[1] // 32


def bwd_tile_from_name(pretty):
    m = re.search(
        r"Sm70,\s*(?:cutlass::)?half_t,\s*(?:true|false),\s*(?:true|false),\s*"
        r"(?:true|false),\s*(\d+),\s*(\d+),\s*(\d+)",
        pretty,
    )
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(
        r"kBlockSizeI_?\s*=\s*(\d+).*?kBlockSizeJ_?\s*=\s*(\d+).*?kMaxK_?\s*=\s*(\d+)",
        pretty,
    )
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


BWD_TYPE_TU = """
#define HAS_PYTORCH 1
#include <kernel_backward.h>
using CCAB_AK = AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t,
    /*kIsAligned*/ true, /*kApplyDropout*/ %(drop)s, /*kPreload*/ false,
    /*kBlockSizeI*/ %(bi)d, /*kBlockSizeJ*/ %(bj)d, /*kMaxK*/ %(mk)d,
    /*kKeysQueriesAlignedToBlockSize*/ %(aligned)s, /*kEnableSplitKeys*/ %(esk)s>;
"""

BWD_ONE_TU = (
    BWD_TYPE_TU
    + """
template __global__ void attention_kernel_backward_batched_impl<CCAB_AK>(CCAB_AK::Params);
"""
)

# Точка входа С НАШИМ бюджетом. Нужна потому, что `-maxrregcount` ЗАМЕРЕННО НЕ ДЕЙСТВУЕТ на ядро
# с `__launch_bounds__` (первый свип дал 173 регистра при бюджете 96 -- физически невозможно, то
# есть флаг был молча проигнорирован). Единственный способ сменить бюджет боевого ядра -- сменить
# сам `__launch_bounds__`, а для этого нужна своя обёртка с тем же телом.
BWD_ENTRY_TU = (
    BWD_TYPE_TU
    + """
__global__ void __launch_bounds__(%(threads)d, %(minb)d)
    ccab_bwd_entry(CCAB_AK::Params p) {
  if (!p.advance_to_block()) { return; }
  CCAB_AK::attention_kernel(p);
}
"""
)


def write_bwd_one(outdir, bi, bj, mk, drop=False, aligned=False, esk=True):
    """Минимальная единица трансляции с ОДНИМ инстанцированным ядром backward.

    Зачем: полный attn_bwd_cutlass.cu собирает десятки инстанциаций и идёт десятки минут; для
    вопроса "сколько регистров у ЭТОЙ плитки" достаточно одной. Кодогенерация ядра от соседей по
    единице трансляции не зависит, но это ПРЕДПОЛОЖЕНИЕ -- команда `bwd --full` собирает боевой
    файл целиком и сверяет числа (см. README, раздел про сверку one-vs-full).
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "bwd_one_%d_%d_%d.cu" % (bi, bj, mk))
    with open(path, "w") as f:
        f.write(
            BWD_ONE_TU
            % {
                "bi": bi,
                "bj": bj,
                "mk": mk,
                "drop": "true" if drop else "false",
                "aligned": "true" if aligned else "false",
                "esk": "true" if esk else "false",
            }
        )
    return path


def cmd_bwd_one(args):
    prof = PROFILES[args.profile](repo=args.repo, cuda_home=args.cuda_home)
    bi, bj, mk = [int(x) for x in args.tile.split(",")]
    src = write_bwd_one(
        args.outdir,
        bi,
        bj,
        mk,
        drop=args.dropout,
        aligned=args.aligned,
        esk=not args.no_esk,
    )
    pre = BWD_PROBE_PREAMBLE
    pr = probe_constants(
        pre,
        [
            "sizeof(CCAB_AK<%d,%d,%d>::SharedStorage)" % (bi, bj, mk),
            "CCAB_AK<%d,%d,%d>::kNumThreads" % (bi, bj, mk),
            "CCAB_AK<%d,%d,%d>::kMinBlocksPerSm" % (bi, bj, mk),
        ],
        prof,
        args.outdir,
    )
    dyn = pr["values"].get("sizeof(CCAB_AK<%d,%d,%d>::SharedStorage)" % (bi, bj, mk))
    thr = pr["values"].get("CCAB_AK<%d,%d,%d>::kNumThreads" % (bi, bj, mk))
    minb = pr["values"].get("CCAB_AK<%d,%d,%d>::kMinBlocksPerSm" % (bi, bj, mk))
    if thr and minb:
        print(
            "объявленный бюджет: __launch_bounds__(%d, %d) -> %d рег/поток"
            % (thr, minb, min(REG_ISA_MAX, REGFILE_PER_SM // (thr * minb)))
        )
    res = analyse_build(
        src,
        list(args.define),
        prof,
        args.outdir,
        filt=args.filter or "attention_kernel_backward",
        threads_hint=thr,
        dyn_smem=dyn,
    )
    res.setdefault("unparsed", []).extend(
        ["зонд: " + e + " -- " + w for e, w in pr["unparsed"]]
    )
    print_table(
        res,
        "backward одной плиткой BI=%d BJ=%d MK=%d  [%s]"
        % (bi, bj, mk, " ".join("-D" + d for d in args.define) or "без макросов"),
    )
    if args.json and res.get("ok"):
        with open(args.json, "w") as f:
            json.dump(res["rows"], f, ensure_ascii=False, indent=1)
    return 0 if res.get("ok") else 1


def cmd_regsweep(args):
    """Кривая 'объявленный бюджет -> разлив' для ОДНОГО боевого ядра.

    Бюджет меняется ЧЕРЕЗ `__launch_bounds__(threads, minb)` собственной обёртки, а НЕ через
    `-maxrregcount`: последний ЗАМЕРЕННО не действует на ядро, у которого launch_bounds уже есть
    (свип 96..172 дал 173 регистра на всех точках -- флаг молча проигнорирован). Инструмент
    проверяет применение бюджета арифметически: REG > бюджета означает, что бюджет НЕ применён,
    и такая строка помечается, а не выдаётся за результат.

    Это прямая проверка замеренного порога R_required = MaxLive + 7. Времени НЕ меряет.
    """
    prof = PROFILES[args.profile](repo=args.repo, cuda_home=args.cuda_home)
    bi, bj, mk = [int(x) for x in args.tile.split(",")]
    threads = bi * bj // 32
    minbs = [int(x) for x in args.minb.split(",")] if args.minb else [1, 2, 3, 4, 5, 6]
    print(
        "ядро: bwd<%d,%d,%d%s>, %d потоков. Бюджет = min(255, 65536/(потоки*minb))."
        % (bi, bj, mk, ",noESK" if args.no_esk else "", threads)
    )
    print(
        "%5s %8s %6s %6s %9s %9s %s"
        % ("minb", "бюджет", "рег", "стек", "отчёт Б", "LDL/STL", "вердикт")
    )
    rows = []
    ignored = []
    for mb in minbs:
        budget = min(REG_ISA_MAX, REGFILE_PER_SM // (threads * mb))
        src = os.path.join(args.outdir, "bwd_entry_%d_%d_%d_m%d.cu" % (bi, bj, mk, mb))
        os.makedirs(args.outdir, exist_ok=True)
        with open(src, "w") as f:
            f.write(
                BWD_ENTRY_TU
                % {
                    "bi": bi,
                    "bj": bj,
                    "mk": mk,
                    "drop": "false",
                    "aligned": "true" if args.aligned else "false",
                    "esk": "false" if args.no_esk else "true",
                    "threads": threads,
                    "minb": mb,
                }
            )
        res = analyse_build(
            src,
            list(args.define),
            prof,
            os.path.join(args.outdir, "m%d" % mb),
            filt="ccab_bwd_entry",
            threads_hint=threads,
        )
        if not res.get("ok") or not res["rows"]:
            print("%5d %8d  КОМПИЛЯЦИЯ/РАЗБОР НЕ ДАЛИ СТРОКИ (см. лог)" % (mb, budget))
            continue
        r = res["rows"][0]
        mark = ""
        if r["regs"] and r["regs"] > budget:
            mark = "  <<< БЮДЖЕТ НЕ ПРИМЕНЁН (REG > бюджета)"
            ignored.append(mb)
        print(
            "%5d %8d %6s %6s %5s/%-3s %4s/%-4s %s%s"
            % (
                mb,
                budget,
                r["regs"],
                r["stack"],
                r["spill_st"],
                r["spill_ld"],
                r["ldl_body"],
                r["stl_body"],
                r["verdict"],
                mark,
            )
        )
        rows.append(
            {
                "minb": mb,
                "budget": budget,
                "applied": not mark,
                **{
                    k: r[k]
                    for k in (
                        "regs",
                        "stack",
                        "spill_st",
                        "spill_ld",
                        "ldl_body",
                        "stl_body",
                        "verdict",
                    )
                },
            }
        )
    if ignored:
        print(
            "!! на %d точках бюджет НЕ применился -- эти строки НЕ являются кривой цены."
            % len(ignored)
        )
    rows = [x for x in rows if x["applied"]]
    clean = [x for x in rows if "РАЗЛИВ" not in x["verdict"]]
    dirty = [x for x in rows if "РАЗЛИВ" in x["verdict"]]
    print()
    if clean and dirty:
        lo = max(x["budget"] for x in dirty)
        hi = min(x["budget"] for x in clean)
        print(
            "ИЗЛОМ в вилке (%d, %d]: при бюджете %d разлив есть, при %d уже нет."
            % (lo, hi, lo, hi)
        )
        reg_free = min(x["regs"] for x in clean)
        print(
            "  по правилу R_required = MaxLive+7: разлив при %d => MaxLive > %d; "
            "чисто при %d с REG=%d => MaxLive <= %d. ИТОГО MaxLive в (%d, %d]."
            % (lo, lo - SPILL_KNEE, hi, reg_free, reg_free, lo - SPILL_KNEE, reg_free)
        )
        print(
            "  ВНИМАНИЕ: ось бюджета КВАНТОВАНА -- при %d потоках доступны только "
            "65536/(%d*minb) = %s. Уже вилки не получить: это ограничение МЕТОДА, а не "
            "свойство ядра (`-maxrregcount` тут не работает, см. заголовок команды)."
            % (threads, threads, ", ".join(str(x["budget"]) for x in rows))
        )
    elif clean:
        print(
            "Разлива нет НИ ПРИ ОДНОМ из бюджетов %s -- излом ниже, свип надо продолжить вниз."
            % ",".join(str(x["budget"]) for x in rows)
        )
    elif dirty:
        print(
            "Разлив ПРИ ВСЕХ бюджетах %s -- излом выше, свип надо продолжить вверх."
            % ",".join(str(x["budget"]) for x in rows)
        )
    else:
        print("НЕ РАЗОБРАНО: ни одна точка свипа не дала строки.")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
    return 0


def cmd_bwd(args):
    """ЯКОРЬ 2: отгруженный backward -- 255 регистров, 41232 Б динамической, 128 потоков."""
    prof = PROFILES[args.profile](repo=args.repo, cuda_home=args.cuda_home)
    outdir = args.outdir
    src = os.path.join(args.repo, "fa2_sm70/csrc/attn_bwd_cutlass.cu")
    defines = ["FMHA_BWD_R7", "FMHA_BWD_COAL_FO"] + list(args.define)

    # 1) динамическая разделяемая = sizeof(SharedStorage) -- зондом, без запуска
    tiles = [
        (64, 64, 64),
        (64, 64, 128),
        (64, 64, 256),
        (64, 64, 512),
        (64, 128, 64),
        (64, 128, 128),
        (64, 128, 256),
        (64, 128, 512),
        (128, 64, 64),
        (128, 64, 128),
        (128, 64, 256),
        (128, 64, 512),
        (128, 64, 65536),
        (128, 128, 256),
        (128, 128, 512),
        (128, 32, 256),
        (128, 32, 512),
    ]
    exprs = ["sizeof(CCAB_AK<%d,%d,%d>::SharedStorage)" % t for t in tiles]
    exprs += ["CCAB_AK<%d,%d,%d>::kNumThreads" % t for t in tiles]
    print(
        "Зонд sizeof(SharedStorage) по всем плиткам backward (компиляция, без GPU) ..."
    )
    pr = probe_constants(BWD_PROBE_PREAMBLE, exprs, prof, outdir)
    smem_by_tile, thr_by_tile = {}, {}
    for t in tiles:
        e = "sizeof(CCAB_AK<%d,%d,%d>::SharedStorage)" % t
        if e in pr["values"]:
            smem_by_tile[t] = pr["values"][e]
        e2 = "CCAB_AK<%d,%d,%d>::kNumThreads" % t
        if e2 in pr["values"]:
            thr_by_tile[t] = pr["values"][e2]
    print("%-18s %10s %8s" % ("плитка BI,BJ,MK", "смем-дин Б", "потоков"))
    for t in tiles:
        print(
            "%-18s %10s %8s"
            % (str(t), smem_by_tile.get(t, "?"), thr_by_tile.get(t, "?"))
        )
    hits = [
        t
        for t in tiles
        if smem_by_tile.get(t) == ANCHOR_BWD["smem_dyn"]
        and thr_by_tile.get(t) == ANCHOR_BWD["threads"]
    ]
    print()
    print(
        "плитки, дающие ЯКОРНЫЕ (%d потоков, %d Б динамической): %s"
        % (ANCHOR_BWD["threads"], ANCHOR_BWD["smem_dyn"], hits or "НЕТ НИ ОДНОЙ")
    )
    probe_unparsed = ["зонд: " + e + " -- " + why for e, why in pr["unparsed"]]

    # 2) регистры/разлив -- из cubin
    print()
    dyn_fn = lambda p: smem_by_tile.get(bwd_tile_from_name(p) or ())
    if args.reuse_cubin:
        print("Разбор ГОТОВОГО cubin: " + args.reuse_cubin)
        res = analyse_build(
            src,
            defines,
            prof,
            outdir,
            filt=args.filter or "attention_kernel_backward",
            reuse=args.reuse_cubin,
            log_text=open(args.ptxas_log, errors="replace").read()
            if args.ptxas_log
            else "",
            threads_hint=bwd_threads_from_name,
            dyn_smem=dyn_fn,
        )
    elif args.full:
        print(
            "Компиляция ВСЕГО боевого attn_bwd_cutlass.cu (~11 мин на этой машине) ..."
        )
        res = analyse_build(
            src,
            defines,
            prof,
            outdir,
            filt=args.filter or "attention_kernel_backward",
            threads_hint=bwd_threads_from_name,
            dyn_smem=dyn_fn,
        )
    else:
        t = hits[0] if hits else (64, 64, 256)
        one = write_bwd_one(outdir, *t)
        print(
            "Компиляция ОДНОЙ якорной плитки %s отдельной единицей трансляции (~4 мин). "
            "Полный файл: --full." % (t,)
        )
        res = analyse_build(
            one,
            defines,
            prof,
            outdir,
            filt=args.filter or "attention_kernel_backward",
            threads_hint=bwd_threads_from_name,
            dyn_smem=dyn_fn,
        )
    res.setdefault("unparsed", []).extend(probe_unparsed)
    print_table(
        res, "ЯДРА BACKWARD (sm_70), сборка: " + " ".join("-D" + d for d in defines)
    )

    # 3) сверка с якорем
    print()
    print(
        "ЯКОРЬ 2 (ncu на боевом ядре): 255 регистров, 41232 Б динамической, блок 128 потоков."
    )
    if not res.get("ok"):
        print("  НЕ ПРОВЕРЕН: компиляция не прошла.")
        return 1
    matched = [
        r
        for r in res["rows"]
        if r["regs"] == ANCHOR_BWD["regs"]
        and r["threads"] == ANCHOR_BWD["threads"]
        and r["smem_dyn"] == ANCHOR_BWD["smem_dyn"]
    ]
    show = (
        matched if matched else sorted(res["rows"], key=lambda x: -(x["regs"] or 0))[:8]
    )
    for r in show:
        print(
            "   %-32s рег=%-4s потоков=%-5s смем-дин=%-7s стек=%-4s LDL/STL=%s/%s %s"
            % (
                shorten(r["name"], 32),
                r["regs"],
                r["threads"],
                r["smem_dyn"],
                r["stack"],
                r["ldl_body"],
                r["stl_body"],
                "<-- ЯКОРЬ" if r in matched else "",
            )
        )
    print(
        "  ИТОГ: "
        + (
            "СОШЁЛСЯ (%d ядер попали в тройку чисел)" % len(matched)
            if matched
            else "НЕ СОШЁЛСЯ -- см. таблицу выше, НЕ ПОДКРУЧИВАТЬ"
        )
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "rows": res["rows"],
                    "smem_by_tile": {str(k): v for k, v in smem_by_tile.items()},
                    "threads_by_tile": {str(k): v for k, v in thr_by_tile.items()},
                    "unparsed": res["unparsed"],
                    "anchor_matched": len(matched),
                },
                f,
                ensure_ascii=False,
                indent=1,
            )
        print("  JSON: " + args.json)
    return 0 if matched else 1


def cmd_compile(args):
    prof = PROFILES[args.profile](repo=args.repo, cuda_home=args.cuda_home)
    if prof.get("missing_torch_include"):
        print("ЗАМЕЧАНИЕ: заголовки torch не найдены -- профиль fa2 неполон.")
    res = analyse_build(
        args.src,
        list(args.define),
        prof,
        args.outdir,
        filt=args.filter,
        threads_hint=args.threads,
        dyn_smem=args.dyn_smem,
    )
    print_table(
        res,
        "%s  [%s]"
        % (args.src, " ".join("-D" + d for d in args.define) or "без макросов"),
    )
    if args.json and res.get("ok"):
        with open(args.json, "w") as f:
            json.dump(res["rows"], f, ensure_ascii=False, indent=1)
    return 0 if res.get("ok") else 1


def cmd_cubin(args):
    """Разобрать ГОТОВЫЙ cubin (например, оставшийся от боевой сборки). Компиляции нет."""
    prof = PROFILES[args.profile](repo=args.repo, cuda_home=args.cuda_home)
    log_text = open(args.ptxas_log, errors="replace").read() if args.ptxas_log else ""
    smem_map = {}
    if args.smem_json:
        raw = json.load(open(args.smem_json))
        for k, v in raw.get("smem_by_tile", raw).items():
            smem_map[tuple(int(x) for x in re.findall(r"\d+", k))] = v
    res = analyse_build(
        args.src or "(нет исходника)",
        [],
        prof,
        args.outdir,
        filt=args.filter,
        reuse=args.cubin,
        log_text=log_text,
        threads_hint=bwd_threads_from_name if args.bwd else args.threads,
        dyn_smem=(lambda p: smem_map.get(bwd_tile_from_name(p) or ()))
        if smem_map
        else None,
    )
    print_table(res, "готовый cubin: " + args.cubin)
    if args.json and res.get("ok"):
        with open(args.json, "w") as f:
            json.dump(res["rows"], f, ensure_ascii=False, indent=1)
    return 0 if res.get("ok") else 1


def cmd_ab(args):
    prof = PROFILES[args.profile](repo=args.repo, cuda_home=args.cuda_home)
    da = shlex.split(args.a)
    db = shlex.split(args.b)
    ra = analyse_build(
        args.src,
        [d[2:] for d in da if d.startswith("-D")],
        prof,
        os.path.join(args.outdir, "A"),
        filt=args.filter,
        threads_hint=args.threads,
    )
    rb = analyse_build(
        args.src_b or args.src,
        [d[2:] for d in db if d.startswith("-D")],
        prof,
        os.path.join(args.outdir, "B"),
        filt=args.filter,
        threads_hint=args.threads,
    )
    print_table(ra, "A: " + (args.a or "без макросов"))
    print_table(rb, "B: " + (args.b or "без макросов"))
    if not (ra.get("ok") and rb.get("ok")):
        return 1
    ia = {r["name"]: r for r in ra["rows"]}
    ib = {r["name"]: r for r in rb["rows"]}
    print("=" * 118)
    print("РАЗНИЦА B-A (только изменившееся)")
    print("=" * 118)
    only_a = sorted(set(ia) - set(ib))
    only_b = sorted(set(ib) - set(ia))
    changed = 0
    for n in sorted(set(ia) & set(ib)):
        a, b = ia[n], ib[n]
        d = []
        for k, lab in (
            ("regs", "рег"),
            ("stack", "стек"),
            ("ldl_body", "LDL"),
            ("stl_body", "STL"),
            ("smem_static", "смем-ст"),
            ("n_insn", "команд"),
        ):
            if a.get(k) != b.get(k):
                d.append("%s %s->%s" % (lab, a.get(k), b.get(k)))
        if a["verdict"] != b["verdict"]:
            d.append("вердикт %s->%s" % (a["verdict"], b["verdict"]))
        if d:
            changed += 1
            print("%-50s %s" % (shorten(n, 50), "; ".join(d)))
    if not changed:
        print(
            "  ни одно общее ядро не изменилось по регистрам/разливу/смем/числу команд."
        )
    if only_a or only_b:
        print("  только в A: %s" % ", ".join(shorten(x, 40) for x in only_a[:5]))
        print("  только в B: %s" % ", ".join(shorten(x, 40) for x in only_b[:5]))
    print()
    print(
        "НЕ РАЗОБРАНО (A: %d, B: %d) -- перечислено в таблицах выше."
        % (len(ra.get("unparsed", [])), len(rb.get("unparsed", [])))
    )
    return 0


def cmd_sizeof(args):
    prof = PROFILES[args.profile](repo=args.repo, cuda_home=args.cuda_home)
    pre = open(args.pre_file).read() if args.pre_file else (args.pre or "")
    r = probe_constants(pre, list(args.expr), prof, args.outdir)
    print("%-70s %12s" % ("выражение", "значение"))
    for e in args.expr:
        print("%-70s %12s" % (e[:70], r["values"].get(e, "?")))
    print()
    print("НЕ РАЗОБРАНО (%d):" % len(r["unparsed"]))
    for e, why in r["unparsed"]:
        print("  * %s -- %s" % (e, why))
    if args.log:
        open(args.log, "w").write(r["log"])
    return 0 if not r["unparsed"] else 1


def cmd_tiny(args):
    """Микротест разбора и правила 'без бюджета ptxas не разливает'. ~10 с, без cutlass."""
    prof = profile_bare(repo=args.repo, cuda_home=args.cuda_home)
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ccab_tiny.cu")
    outdir = os.path.join(args.outdir, "tiny")
    res = analyse_build(src, [], prof, outdir, threads_hint=256)
    print_table(
        res, "МИКРОТЕСТ tools/ccab_tiny.cu (два одинаковых тела, разный бюджет)"
    )
    if not res.get("ok"):
        return 1
    by = {r["name"].split("(")[0].split()[-1]: r for r in res["rows"]}
    free, tight = by.get("k_free"), by.get("k_tight")
    ok = True
    if not free or not tight:
        print("  НЕ РАЗОБРАНО: в cubin нет k_free/k_tight -- микротест не состоялся.")
        return 1
    if (
        free["regs"]
        and free["regs"] > 64
        and not free["ldl_body"]
        and not free["stl_body"]
    ):
        print(
            "  ОК: без __launch_bounds__ ptxas взял %d регистров и НЕ разлил."
            % free["regs"]
        )
    else:
        ok = False
        print(
            "  НЕ ПОДТВЕРДИЛОСЬ: ожидали 'много регистров, нулевой разлив', получили "
            "рег=%s LDL/STL=%s/%s" % (free["regs"], free["ldl_body"], free["stl_body"])
        )
    if (
        tight["regs"] == 32
        and (tight["ldl_body"] or 0) + (tight["stl_body"] or 0) > 100
    ):
        print(
            "  ОК: __launch_bounds__(256,8) -> бюджет 32 -> %d рег, %d LDL / %d STL в теле."
            % (tight["regs"], tight["ldl_body"], tight["stl_body"])
        )
    else:
        ok = False
        print(
            "  НЕ ПОДТВЕРДИЛОСЬ: при бюджете 32 получили рег=%s LDL/STL=%s/%s"
            % (tight["regs"], tight["ldl_body"], tight["stl_body"])
        )
    if not free.get("has_lb", True) and tight.get("has_lb"):
        print(
            "  ОК: поимённое определение __launch_bounds__ различило два ядра одного файла."
        )
    else:
        ok = False
        print(
            "  НЕ ПОДТВЕРДИЛОСЬ: поимённое определение __launch_bounds__ (k_free=%s, k_tight=%s)"
            % (free.get("has_lb"), tight.get("has_lb"))
        )
    return 0 if ok else 1


def cmd_selftest(args):
    rc0 = cmd_tiny(args)
    print()
    rc1 = cmd_qtable(args)
    print()
    rc2 = cmd_bwd(args)
    print()
    print("=" * 118)
    print(
        "САМОПРОВЕРКА: микротест ptxas -- %s; якорь Q(W) -- %s; "
        "якорь backward (255/41232/128) -- %s"
        % (
            "ПРОЙДЕН" if rc0 == 0 else "НЕ ПРОЙДЕН",
            "СОШЁЛСЯ" if rc1 == 0 else "НЕ СОШЁЛСЯ",
            "СОШЁЛСЯ" if rc2 == 0 else "НЕ СОШЁЛСЯ",
        )
    )
    return rc0 | rc1 | rc2


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--profile", default="fa2", choices=sorted(PROFILES))
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--cuda-home", default=CUDA_HOME_DEFAULT)
    ap.add_argument("--outdir", default="./build/ccab")
    ap.add_argument("--filter", default=None, help="регэксп по имени ядра")
    ap.add_argument("--json", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("qtable", help="якорь 1: Q(W)")
    p.set_defaults(func=cmd_qtable)

    p = sub.add_parser("compile", help="разобрать одну сборку")
    p.add_argument("--src", required=True)
    p.add_argument("-D", "--define", action="append", default=[])
    p.add_argument(
        "--threads", type=int, default=None, help="потоков в блоке (для занятости)"
    )
    p.add_argument(
        "--dyn-smem", type=int, default=None, help="динамическая разделяемая, Б"
    )
    p.set_defaults(func=cmd_compile)

    p = sub.add_parser("ab", help="A/B двух наборов макросов")
    p.add_argument("--src", required=True)
    p.add_argument("--src-b", default=None)
    p.add_argument("--a", default="")
    p.add_argument("--b", default="")
    p.add_argument("--threads", type=int, default=None)
    p.set_defaults(func=cmd_ab)

    p = sub.add_parser("sizeof", help="compile-time константы зондом")
    p.add_argument("--pre", default=None)
    p.add_argument("--pre-file", default=None)
    p.add_argument("--expr", action="append", required=True)
    p.add_argument("--log", default=None)
    p.set_defaults(func=cmd_sizeof)

    p = sub.add_parser("bwd", help="якорь 2: боевой backward")
    p.add_argument("-D", "--define", action="append", default=[])
    p.add_argument(
        "--full", action="store_true", help="собрать ВЕСЬ attn_bwd_cutlass.cu (~11 мин)"
    )
    p.add_argument(
        "--reuse-cubin", default=None, help="взять готовый cubin, не компилировать"
    )
    p.add_argument("--ptxas-log", default=None)
    p.set_defaults(func=cmd_bwd)

    p = sub.add_parser("bwd-one", help="backward ОДНОЙ плиткой (быстро)")
    p.add_argument("--tile", default="64,64,256", help="BI,BJ,MK")
    p.add_argument("--dropout", action="store_true")
    p.add_argument("--aligned", action="store_true")
    p.add_argument(
        "--no-esk", action="store_true", help="kEnableSplitKeys=false (торчев вариант)"
    )
    p.add_argument("-D", "--define", action="append", default=[])
    p.set_defaults(func=cmd_bwd_one)

    p = sub.add_parser(
        "tiny", help="микротест разбора и правила 'без бюджета нет разлива'"
    )
    p.set_defaults(func=cmd_tiny)

    p = sub.add_parser("cubin", help="разобрать готовый cubin (без компиляции)")
    p.add_argument("--cubin", required=True)
    p.add_argument("--src", default=None)
    p.add_argument(
        "--ptxas-log",
        default=None,
        help="лог сборки с -Xptxas -v (иначе колонка 'отчёт' пуста)",
    )
    p.add_argument("--smem-json", default=None, help="JSON от 'bwd' с smem_by_tile")
    p.add_argument(
        "--bwd", action="store_true", help="имена ядер backward -> потоки из плитки"
    )
    p.add_argument("--threads", type=int, default=None)
    p.set_defaults(func=cmd_cubin)

    p = sub.add_parser(
        "regsweep", help="кривая бюджет -> разлив (через __launch_bounds__)"
    )
    p.add_argument("--tile", default="64,64,256")
    p.add_argument(
        "--minb",
        default=None,
        help="список minBlocksPerSm для __launch_bounds__ (по умолчанию 1..6)",
    )
    p.add_argument("--aligned", action="store_true")
    p.add_argument("--no-esk", action="store_true")
    p.add_argument("-D", "--define", action="append", default=[])
    p.set_defaults(func=cmd_regsweep)

    p = sub.add_parser("selftest", help="оба якоря")
    p.add_argument("-D", "--define", action="append", default=[])
    p.add_argument("--full", action="store_true")
    p.add_argument("--reuse-cubin", default=None)
    p.add_argument("--ptxas-log", default=None)
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
