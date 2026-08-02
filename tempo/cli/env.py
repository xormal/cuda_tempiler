#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ЕДИНСТВЕННОЕ МЕСТО В ДЕРЕВЕ, ГДЕ ЖИВУТ ПУТИ ОКРУЖЕНИЯ (правило Р8 спецификации).

ЗАЧЕМ ОТДЕЛЬНЫЙ ФАЙЛ.  До 2026-08-02 путь `/opt/conda/miniconda3/...` был вшит в 31 файл.
Тулчейн переехал в `/home/alex/miniconda3`, и разом умерло ВСЁ, что компилирует: cc_ab,
padsweep, twin build, phaseprof static, ncu, bwd_phase_ext.  Диагноз занял больше времени,
чем правка, потому что каждый инструмент падал по-своему.

ПРАВИЛО, КОТОРОЕ ЭТОТ ФАЙЛ ВВОДИТ И КОТОРОЕ НЕЛЬЗЯ ОБХОДИТЬ:

    ИСПОЛНЯЕМЫЙ путь (то, что будет запущено) -- только отсюда.
    ЗАПИСЬ О ЗАМЕРЕ (где замер был снят) -- НЕ ТРОГАТЬ НИКОГДА.

Второе -- не педантизм.  `data/ncu/fwd128.json`, `*.spec.json`-снимки, таблицы происхождения
в `docs/journal/` содержат путь как ЧАСТЬ ПРОТОКОЛА: он говорит, на какой машине и каким
бинарём снято число.  Переписать его = подделать происхождение замера.  Гейт G7 проверяет
ровно это различие: он смотрит только на исполняемые файлы.

ПОРЯДОК РАЗРЕШЕНИЯ (первое найденное побеждает):
    1. переменная окружения (TEMPO_PY, CUDA_HOME, TEMPO_NCU, TEMPO_TORCH_INCLUDE);
    2. кандидаты из TEMPO_ROOTS (двоеточие-разделённый список корней conda/окружений);
    3. встроенный список кандидатов (см. _ROOTS) -- сюда дописывается новая машина;
    4. PATH.
Не найдено -> возвращается None, а вызывающий обязан сказать об этом ЧЕЛОВЕКУ, а не упасть
через 200 строк с FileNotFoundError на середине сборки.

ЗАПУСК
    python3 tempo/cli/env.py             -- таблица «что найдено и откуда»
    python3 tempo/cli/env.py --sh        -- строки export для shell-скриптов
    python3 tempo/cli/env.py --json      -- то же машинно
    python3 tempo/cli/env.py --selftest  -- самопроверка резолвера (карта не нужна)
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys

# --------------------------------------------------------------------------------------------
# Корни, в которых искать окружения.  ЭТО ЕДИНСТВЕННЫЙ СПИСОК; новая машина = одна строка сюда
# (или TEMPO_ROOTS в окружении, чтобы не править файл вовсе).
# --------------------------------------------------------------------------------------------
_ROOTS = (
    os.path.expanduser("~/miniconda3"),
    "/home/alex/miniconda3",
    "/opt/conda/miniconda3",  # прежняя машина; оставлено, чтобы старые записи воспроизводились
    "/opt/conda",
    os.environ.get("CONDA_PREFIX", "") or "/nonexistent",
)


def _roots():
    extra = [r for r in os.environ.get("TEMPO_ROOTS", "").split(":") if r]
    seen, out = set(), []
    for r in list(extra) + list(_ROOTS):
        r = os.path.abspath(os.path.expanduser(r))
        if r and r not in seen and os.path.isdir(r):
            seen.add(r)
            out.append(r)
    return out


def _first(paths):
    for p in paths:
        for hit in sorted(glob.glob(p)):
            if os.path.exists(hit):
                return hit
    return None


# --------------------------------------------------------------------------------------------
# ИНТЕРПРЕТАТОР.  Замерено: базовый conda-python имеет gcc>14, и nvcc его отвергает; а
# residency.py не разбирается Python 3.11 (PEP 701 во вложенных f-строках).  Отсюда ДВА
# разных ответа, и спрашивать надо тот, который нужен инструменту.
# --------------------------------------------------------------------------------------------
def python_vllm():
    """Интерпретатор с torch (боевой): им гоняются замеры и сборки расширений."""
    v = os.environ.get("TEMPO_PY")
    if v and os.path.exists(v):
        return v
    return _first([os.path.join(r, "envs", "vllm", "bin", "python") for r in _roots()])


def python_312():
    """Интерпретатор >= 3.12 (нужен residency.py: PEP 701)."""
    v = os.environ.get("TEMPO_PY312")
    if v and os.path.exists(v):
        return v
    cands = []
    for r in _roots():
        cands.append(os.path.join(r, "bin", "python"))
        cands += [os.path.join(r, "envs", "*", "bin", "python")]
    for c in cands:
        for hit in sorted(glob.glob(c)):
            try:
                import subprocess

                out = subprocess.run(
                    [hit, "-c", "import sys;print(sys.version_info[:2])"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if out.returncode == 0 and "(3, 1" in out.stdout:
                    major_minor = out.stdout.strip().strip("()").split(",")
                    if int(major_minor[1]) >= 12:
                        return hit
            except Exception:
                continue
    return None


# --------------------------------------------------------------------------------------------
# ТУЛЧЕЙН
# --------------------------------------------------------------------------------------------
def cuda_home():
    v = os.environ.get("CUDA_HOME")
    if v and os.path.isdir(v):
        return v
    hit = _first(
        [os.path.join(r, "envs", "cuda128") for r in _roots()]
        + [os.path.join(r, "envs", "cuda*") for r in _roots()]
        + ["/usr/local/cuda"]
    )
    return hit


def _cuda_bin(name):
    ch = cuda_home()
    if ch:
        p = os.path.join(ch, "bin", name)
        if os.path.exists(p):
            return p
    return shutil.which(name)


def nvcc():
    return _cuda_bin("nvcc")


def cuobjdump():
    return _cuda_bin("cuobjdump")


def nvdisasm():
    return _cuda_bin("nvdisasm")


def ptxas():
    return _cuda_bin("ptxas")


def cuda_include():
    ch = cuda_home()
    if not ch:
        return None
    for cand in (
        os.path.join(ch, "targets", "x86_64-linux", "include"),
        os.path.join(ch, "include"),
    ):
        if os.path.isdir(cand):
            return cand
    return None


def ncu_candidates():
    """Кандидаты в порядке доверия.  ОБЫЧНЫЙ bin/ncu ВРЁТ 'not installed' -- он последний."""
    pats = []
    for r in _roots():
        pats += [
            os.path.join(r, "pkgs", "nsight-compute-*", "nsight-compute", "*", "ncu"),
            os.path.join(r, "pkgs", "nsight-compute-*", "nsight-compute-*", "ncu"),
            os.path.join(r, "pkgs", "nsight-compute-*", "bin", "ncu"),
        ]
    pats += [
        "/usr/local/cuda/nsight-compute/*/ncu",
        "/usr/local/NVIDIA-Nsight-Compute*/ncu",
    ]
    for r in _roots():
        pats.append(os.path.join(r, "envs", "*", "bin", "ncu"))  # заведомо подозрительные
    out = []
    for p in pats:
        out += sorted(glob.glob(p))
    w = shutil.which("ncu")
    if w:
        out.append(w)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def ncu():
    v = os.environ.get("TEMPO_NCU")
    if v and os.path.exists(v):
        return v
    c = ncu_candidates()
    return c[0] if c else None


def torch_include():
    """Каталог заголовков torch (нужен, чтобы собрать расширение без импорта torch)."""
    v = os.environ.get("TEMPO_TORCH_INCLUDE")
    if v and os.path.isdir(v):
        return v
    pats = []
    for r in _roots():
        pats += [
            os.path.join(r, "envs", "*", "lib", "python3.*", "site-packages", "torch", "include")
        ]
    return _first(pats)


def torch_includes():
    """Все найденные каталоги torch/include -- cc_ab перебирает их по очереди."""
    out = []
    for r in _roots():
        out += sorted(
            glob.glob(
                os.path.join(
                    r, "envs", "*", "lib", "python3.*", "site-packages", "torch", "include"
                )
            )
        )
    return out


# --------------------------------------------------------------------------------------------
# ОТЧЁТ
# --------------------------------------------------------------------------------------------
_ITEMS = (
    ("TEMPO_PY", python_vllm, "интерпретатор с torch (замеры, сборка расширений)"),
    ("CUDA_HOME", cuda_home, "корень CUDA (nvcc его отвергнет, если gcc>14)"),
    ("NVCC", nvcc, "компилятор"),
    ("PTXAS", ptxas, "регистры / разлив / кадр стека"),
    ("CUOBJDUMP", cuobjdump, "SASS для разборщика"),
    ("NVDISASM", nvdisasm, "дизассемблер"),
    ("CUDA_INCLUDE", cuda_include, "заголовки CUDA для линтера"),
    ("NCU", ncu, "ЕДИНСТВЕННЫЙ годный ncu (обычный bin/ncu врёт)"),
    ("TORCH_INCLUDE", torch_include, "заголовки torch"),
)


def table():
    return {name: fn() for name, fn, _ in _ITEMS}


def _selftest():
    """Самопроверка РЕЗОЛВЕРА, не машины.  Карта не нужна, тулчейн может отсутствовать."""
    ok = 0
    total = 0

    def chk(label, cond):
        nonlocal ok, total
        total += 1
        if cond:
            ok += 1
        print(("  ok   " if cond else "  ПАДЁТ ") + label)

    print("САМОПРОВЕРКА tempo/cli/env.py")
    chk("список корней не пуст (иначе резолвер слеп)", len(_roots()) > 0)
    chk("_first на пустом списке возвращает None", _first([]) is None)
    chk("_first на несуществующем возвращает None", _first(["/nonexistent/*/x"]) is None)
    os.environ["TEMPO_PY"] = sys.executable
    chk("переменная окружения имеет приоритет", python_vllm() == sys.executable)
    del os.environ["TEMPO_PY"]
    os.environ["TEMPO_PY"] = "/nonexistent/python"
    chk("несуществующее значение переменной ИГНОРИРУЕТСЯ, а не ломает", python_vllm() != "/nonexistent/python")
    del os.environ["TEMPO_PY"]
    chk("ncu_candidates() -- список без повторов", len(ncu_candidates()) == len(set(ncu_candidates())))
    chk("table() отдаёт все объявленные ключи", set(table()) == {n for n, _, _ in _ITEMS})
    chk("ни один путь не вшит как литерал в ответ", all(
        (v is None or os.path.exists(v)) for v in table().values()
    ))
    print("ИТОГ: %d/%d" % (ok, total))
    return 0 if ok == total else 1


def main(argv):
    if "--selftest" in argv:
        return _selftest()
    t = table()
    if "--json" in argv:
        print(json.dumps(t, ensure_ascii=False, indent=2))
        return 0
    if "--sh" in argv:
        for k, v in t.items():
            if v:
                print('export TEMPO_%s="%s"' % (k.replace("TEMPO_", ""), v))
        return 0
    print("ОКРУЖЕНИЕ (единственное место с путями -- этот файл)")
    miss = 0
    for name, fn, what in _ITEMS:
        v = fn()
        if v is None:
            miss += 1
        print("  %-14s %-70s %s" % (name, v or "НЕ НАЙДЕН", what))
    if miss:
        print(
            "\nНЕ НАЙДЕНО: %d.  Лечение: TEMPO_ROOTS=/путь/к/conda, либо строка в _ROOTS." % miss
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
