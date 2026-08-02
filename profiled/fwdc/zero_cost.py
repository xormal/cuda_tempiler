# -*- coding: utf-8 -*-
"""ЦЕНА ОСНАСТКИ ПРИ МАСКЕ 0 -- ПРОТИВ БОЕВОГО ЗАГОЛОВКА, А НЕ ПРОТИВ САМОЙ СЕБЯ.

Собирает ОДНУ И ТУ ЖЕ единицу трансляции дважды: (а) с БОЕВЫМ kernel_forward.h, (б) с двойником
при FMHA_STRIP_MASK=0, -- и требует совпадения. Сравнение идёт по трём уровням, потому что
совпадения одного мало:

  1. SASS (cuobjdump -sass) -- побайтово. Это про КОД.
  2. Регистры и кадр стека (ptxas -v) -- отдельно, потому что равный SASS при разном кадре
     невозможен, но проверка дешёвая, а «скомпилил -- проверь кадр и LDL/STL» здесь правило.
  3. Весь cubin, собранный БЕЗ -lineinfo, -- побайтово ПОСЛЕ подстановки пути. Единственная
     законная разница -- строка с путём исходника, которую nvcc зашивает в .nv.global.init;
     проверка требует, чтобы после её замены файлы совпали БАЙТ В БАЙТ. Так исключается разница
     в таблицах, которую diff по SASS не видит.

Почему ноль вообще достижим: разметка живёт в `if constexpr`, а подстановки -- в его ветке `else`.
При маске 0 ветка `else` не порождает кода, а сам инструментальный заголовок (fmha_phase.h с его
prmt-пломбами и __constant__-символом) при FMHA_STRIP_MASK==0 даже не подключается -- fwd_phase.h
закрывает боевой путь заглушками.
"""

import os
import re
import subprocess
import sys

# --- ПУТИ ОКРУЖЕНИЯ: единственное место -- tempo/cli/env.py (правило Р8 спецификации) ---
def _tempo_env_load():
    import importlib.util as _u, os as _o

    _p = _o.path.join(
        _o.path.dirname(_o.path.abspath(__file__)), "..", "..", "tempo", "cli", "env.py"
    )
    try:
        _s = _u.spec_from_file_location("tempo_env", _p)
        _m = _u.module_from_spec(_s)
        _s.loader.exec_module(_m)
        return _m
    except Exception:  # инструмент, вынесенный из дерева, обязан остаться запускаемым

        class _Stub:
            def __getattr__(self, _n):
                return lambda *a, **k: None

        return _Stub()


_ENV = _tempo_env_load()


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PROD = "../VLLM_fa2/solutions/fa2_sm70_cutlass_grade"
NVCC = _ENV.nvcc() or "nvcc"
CUOBJ = _ENV.cuobjdump() or "cuobjdump"
PROD_HDR = os.path.join(PROD, "fa2_src/fmha_kernel/kernel_forward.h")
TWIN_HDR = "./profiled/fwd_cutlass/fa2_src/fmha_kernel/kernel_forward.h"


def nvcc(out, twin, lineinfo):
    cmd = [NVCC, "-arch=sm_70", "-O3", "-std=c++17", "-cubin", "-ccbin", "/usr/bin/g++"]
    if lineinfo:
        cmd += ["-lineinfo", "-Xptxas", "-v"]
    else:
        cmd += ["-Xptxas", "-v"]
    if twin:
        cmd += ["-I", os.path.dirname(TWIN_HDR)]
    cmd += [
        "-I",
        os.path.join(PROD, "fa2_src/cutlass/include"),
        "-I",
        os.path.join(PROD, "fa2_src/fmha_kernel"),
        "-DFMHA_STRIP_MASK=0u",
        "-o",
        out,
        os.path.join(HERE, "inst_fwd.cu"),
    ]
    r = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
    )
    if r.returncode:
        sys.stdout.write(r.stdout[-3000:])
        raise SystemExit("сборка провалилась")
    return r.stdout


def resline(log):
    out = []
    cur = None
    for ln in log.splitlines():
        m = re.search(r"entry function '(\S+)'", ln)
        if m:
            cur = m.group(1)
        m = re.search(r"Used (\d+) registers", ln)
        if m and cur:
            out.append((cur, "рег", int(m.group(1))))
        m = re.search(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
            ln,
        )
        if m and cur:
            out.append(
                (
                    cur,
                    "кадр/разлив",
                    (int(m.group(1)), int(m.group(2)), int(m.group(3))),
                )
            )
    return out


def sass(cub):
    return subprocess.run(
        [CUOBJ, "-sass", cub], stdout=subprocess.PIPE, universal_newlines=True
    ).stdout


def secbytes(f, name=".nv.global.init"):
    out = subprocess.run(
        ["readelf", "-x", name, f], capture_output=True, text=True
    ).stdout
    b = bytearray()
    for ln in out.splitlines():
        m = re.match(r"\s+0x[0-9a-f]+ ((?:[0-9a-f]{2,8} ){1,4})", ln)
        if m:
            b += bytes.fromhex(m.group(1).replace(" ", ""))
    return bytes(b)


def secnames(f):
    out = subprocess.run(["readelf", "-SW", f], capture_output=True, text=True).stdout
    return [
        m.group(1)
        for m in re.finditer(r"\[\s*\d+\]\s+(\S+)\s+(?:PROGBITS|NOBITS)", out)
    ]


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = True
    lp = nvcc(os.path.join(OUT, "zc_prod.cubin"), twin=False, lineinfo=True)
    lt = nvcc(os.path.join(OUT, "zc_twin.cubin"), twin=True, lineinfo=True)

    sp, st = (
        sass(os.path.join(OUT, "zc_prod.cubin")),
        sass(os.path.join(OUT, "zc_twin.cubin")),
    )
    print(
        "1) SASS побайтово: %s (%d строк)"
        % ("РАВЕН" if sp == st else "РАЗОШЁЛСЯ", len(sp.splitlines()))
    )
    ok &= sp == st

    rp, rt = resline(lp), resline(lt)
    print("2) регистры/кадр: %s" % ("РАВНЫ" if rp == rt else "РАЗОШЛИСЬ"))
    for a in rp:
        print("     %-60s %s = %s" % (a[0][:60], a[1], a[2]))
    ok &= rp == rt

    fp = os.path.join(OUT, "zc_prod_nl.cubin")
    ft = os.path.join(OUT, "zc_twin_nl.cubin")
    nvcc(fp, twin=False, lineinfo=False)
    nvcc(ft, twin=True, lineinfo=False)
    P, T = PROD_HDR.encode(), TWIN_HDR.encode()
    names = [n for n in secnames(fp) if n in secnames(ft)]
    print("3) cubin БЕЗ -lineinfo, посекционно (%d секций с данными):" % len(names))
    bad = []
    for n in names:
        x, y = secbytes(fp, n), secbytes(ft, n)
        if x == y:
            continue
        if x.replace(P, T) == y:
            print(
                "   %-24s различается ТОЛЬКО зашитым путём исходника (%+d Б)"
                % (n, len(T) - len(P))
            )
            continue
        bad.append(n)
        print("   %-24s РАЗОШЛАСЬ НЕ ПО ПУТИ: %d -> %d Б" % (n, len(x), len(y)))
    print("   секций, разошедшихся по существу: %d" % len(bad))
    print(
        "   (длина файла %d -> %d: %+d Б = %d Б пути + выравнивание секций)"
        % (
            os.path.getsize(fp),
            os.path.getsize(ft),
            os.path.getsize(ft) - os.path.getsize(fp),
            len(T) - len(P),
        )
    )
    ok &= not bad

    print()
    print(
        "ИТОГ: "
        + (
            "ЦЕНА ОСНАСТКИ ПРИ МАСКЕ 0 = НОЛЬ (различается только зашитый путь исходника)"
            if ok
            else "ОСНАСТКА НЕ БЕСПЛАТНА -- разложение по фазам недействительно"
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
