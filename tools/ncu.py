#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ОБЁРТКА НАД ncu, КОТОРАЯ НЕ ВРЁТ: поиск рабочего бинаря, честный разбор, ПОСТРОЧНАЯ привязка.

ЗАЧЕМ. `ncu` на этой машине трижды за день выдал ПРАВДОПОДОБНЫЙ НЕВЕРНЫЙ ответ, и все три раза
молча:

  (1) `.../envs/*/bin/ncu` -- это launcher-заглушка: печатает "Nsight Compute is not installed",
      возвращает 0 и ПУСТОЙ stdout. Таблица прочитала пустоту как "конфликтов нет".
  (2) Локаль ru_RU: разделитель тысяч у ncu -- НЕРАЗРЫВНЫЙ ПРОБЕЛ (U+00A0), "5 238".
      `float()` падает, наивный разбор строку выбрасывает -- и проходят ТОЛЬКО значения < 1000,
      то есть чужие мелкие ядра, а наши крупные исчезают. Таблица выглядит полной.
  (3) 2025.x не поддерживает sm_70, но сообщает об этом не сразу и не в первой строке.

Общее у всех трёх: НЕПОЛНЫЙ ИНСТРУМЕНТ ДАЁТ НЕ "МЕНЬШЕ ДАННЫХ", А ДРУГОЙ ОТВЕТ -- и звучит он с
той же уверенностью. Поэтому здесь всё, что не разобрано, ПЕЧАТАЕТСЯ, а отсутствие своего ядра в
профиле -- ОШИБКА, а не пустая таблица.

ЧТО МЕРИТ. Стоимость разделяемой памяти на Volta -- ВАЙВФРОНТЫ (1 вайвфронт за такт на SM):

    вайвфронты = идеальные (по ШИРИНЕ доступа, неустранимо) + ИЗБЫТОЧНЫЕ (конфликт банков, снимает
                                                                          дополнение раскладки)

Два независимых маршрута к одному числу, оба собираются этой обёрткой и СВЕРЯЮТСЯ друг с другом:

    А (на запуск):  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{ld,st}.sum   -- вайвфронты
                    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum -- конфликты
    Б (на команду): memory_l1_wavefronts_shared, memory_l1_wavefronts_shared_ideal
                    (инстансные метрики секции SourceCounters, значение на КАЖДЫЙ PC)

Маршрут Б -- главное: без него 34.6 % конфликтов в бэкварде некуда прикладывать. Обёртка сводит
сумму Б к итогу А и печатает НЕВЯЗКУ. Разошлись -- значит разобрано не всё, и это видно.

ПОЧЕМУ ПОСТОЧЕЧНАЯ ПРИВЯЗКА "НЕ ПРИХОДИТ ЧЕРЕЗ --csv": её ищут не на той СТРАНИЦЕ. Рабочий путь --
снять отчёт с секцией SourceCounters, а потом ИМПОРТИРОВАТЬ его со страницей source:

    ncu --section SourceCounters --import-source yes -o rep <тело>
    ncu -i rep.ncu-rep --csv --page source --print-source cuda,sass

Форматов страницы ДВА, и они разные (разбираются оба, см. parse_csv_source): у `sass` шапка
"Kernel Name" и заголовок с "Address"; у `cuda,sass` шапка "File Path"/"Function Name", заголовок с
"Line No", колонка "Source" встречается ДВАЖДЫ (CUDA и SASS -- брать по позиции, dict их схлопнет),
а строки CUDA -- это СУММЫ своих команд SASS (складывать и то и другое = удвоить).

ЕДИНИЦЫ. Вайвфронты (штуки), конфликты (штуки, ДОБАВОЧНЫЕ вайвфронты), доля = конфликты/вайвфронты
(безразмерная, верхняя оценка выигрыша ПО ТРАФИКУ разделяемой памяти, НЕ по времени ядра).

ЧЕГО НЕ УМЕЕТ -- см. раздел "СЛЕПЫЕ ЗОНЫ" в конце файла и в README.

ЗАПУСК:
    python tools/ncu.py --selftest                       # якорь: bwd, 4.93e6 / 1.71e6 / 34.6 %
    python tools/ncu.py --which                          # какие бинари найдены и почему отвергнуты
    python tools/ncu.py --kernel 'attention_kernel_backward' -- python bankaudit.py --run bwd
как библиотека:
    from ncu import conflicts
    r = conflicts(r'attention_kernel_backward', [PY, 'bankaudit.py', '--run', 'bwd'])
    r.fraction, r.rows, r.unparsed
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys


# --- ПУТИ ОКРУЖЕНИЯ: единственное место -- tempo/cli/env.py (правило Р8 спецификации) ---
def _tempo_env_load():
    import importlib.util as _u
    import os as _o

    _p = _o.path.join(
        _o.path.dirname(_o.path.abspath(__file__)), "..", "tempo", "cli", "env.py"
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


# ЛОВУШКА ОКРУЖЕНИЯ (уже стоила одного ложного вывода в этом же каталоге): в tempo/tools/ лежит
# СВОЙ timeit.py. Каталог скрипта попадает в sys.path ПЕРВЫМ, и любой импорт `timeit` (его делает
# torch) подхватывает чужой файл -- тело падает ДО первой полезной строки, а ncu отдаёт пустую
# таблицу, которая читается как "конфликтов нет". Вычищаем свой каталог первым действием.
_HERE_DIR = os.path.dirname(os.path.abspath(__file__))
if (
    __name__ == "__main__"
):  # как БИБЛИОТЕКУ нас импортируют ИЗ этого каталога -- не рубить
    sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") != _HERE_DIR]

# ---------------------------------------------------------------------------------------------
# 0. Инвариант окружения: ЛОКАЛЬ. Ставится ДО любого запуска ncu и не спрашивается у среды.
# ---------------------------------------------------------------------------------------------
FORCED_ENV = {"LC_ALL": "C", "LANG": "C", "LANGUAGE": "C"}

# Кандидаты в порядке предпочтения. Именно ПОЛНЫЕ пути внутри пакета, а не .../bin/ncu:
# последний -- launcher-заглушка (см. (1) в шапке).
CANDIDATE_GLOBS = [
    os.environ.get("TEMPO_NCU", ""),  # явное указание владельца
    *(_ENV.ncu_candidates() or []),
    "/opt/nvidia/nsight-compute/*/ncu",
    "/usr/local/cuda*/nsight-compute*/ncu",
    "/usr/local/cuda*/bin/ncu",
    # прочие кандидаты (в т.ч. заведомо подозрительные envs/*/bin/ncu) даёт _ENV
]

# Метрики маршрута А (итог на запуск).
M_LAUNCH = [
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
]
# Метрики маршрута Б (инстансные, на команду). Секция SourceCounters их и собирает.
M_SOURCE = [
    "memory_l1_wavefronts_shared",
    "memory_l1_wavefronts_shared_ideal",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
]

# Символы, которыми ncu/локаль разделяют тысячи. Ни один из них НЕ является десятичной точкой,
# когда LC_ALL=C принудительно выставлен нами (см. FORCED_ENV) -- поэтому их снятие законно.
THOUSANDS = "     ,'"  # NBSP, narrow NBSP, thin, figure, обычный, запятая, апостроф


class NcuError(RuntimeError):
    """Любой отказ, который НЕЛЬЗЯ превращать в пустую таблицу."""


class KernelNotFound(NcuError):
    """Ядро, которое заказал вызывающий, в профиле отсутствует. Это ДРУГОЙ ОТВЕТ, а не меньше данных."""


# ---------------------------------------------------------------------------------------------
# 1. ПОИСК РАБОЧЕГО БИНАРЯ
# ---------------------------------------------------------------------------------------------
class Cand(object):
    def __init__(self, path):
        self.path = path
        self.version = None
        self.sm70 = None
        self.ok = False
        self.why = ""

    def __repr__(self):
        mark = "ГОДЕН" if self.ok else "отвергнут"
        return "%-9s %-70s %-12s %s" % (mark, self.path, self.version or "-", self.why)


def _env(extra=None):
    e = dict(os.environ)
    e.update(FORCED_ENV)
    if extra:
        e.update({k: str(v) for k, v in extra.items()})
    return e


def _run(cmd, timeout=1800, extra_env=None, cwd=None):
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_env(extra_env),
        cwd=cwd,
    )


VER_RE = re.compile(r"Version\s+(\d+)\.(\d+)\.(\d+)")


CANARY_SRC = r"""
// КАНАРЕЙКА: минимальное ядро, на котором проверяется, что кандидат ВООБЩЕ снимает счётчики
// с sm_70. Спрашивать у ncu --list-chips бесполезно: 2025.1 перечисляет gv100 (база метрик
// его знает), а ЦЕЛЕВАЯ часть Volta уже не поддерживает -- отказ приходит только в рантайме.
#include <cstdio>
__global__ void tempo_ncu_canary(float* o) {
  __shared__ float s[64];
  s[threadIdx.x & 63] = threadIdx.x;
  __syncthreads();
  o[threadIdx.x & 63] = s[(threadIdx.x * 3) & 63];
}
int main() { float* d; cudaMalloc(&d, 256); tempo_ncu_canary<<<1, 64>>>(d);
             cudaDeviceSynchronize(); printf("canary %d\n", (int)cudaGetLastError()); return 0; }
"""
CANARY_BIN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "build", "ncu_canary"
)


def _src_digest(text):
    """Отпечаток ИСХОДНИКА -- он и есть имя собранного (LAW=L-CACHE-KEY-BY-CONTENT)."""
    import hashlib

    return hashlib.md5(text.encode("utf-8", "replace")).hexdigest()[:8]


def build_canary():
    """Собрать канарейку (один раз). Не собралась -- вернуть None, а НЕ притвориться, что всё ок.

    ИМЯ СОБРАННОГО НЕСЁТ ОТПЕЧАТОК ИСХОДНИКА.  Прежняя редакция брала «файл по этому пути уже
    есть» за «собрано то, что нужно»: правка CANARY_SRC не меняла ни пути, ни времени решения, и
    прибор молча мерил ПРОШЛУЮ канарейку.  Ключ по содержимому снимает вопрос целиком -- другой
    исходник даёт другое имя, и старое остаётся лежать без вреда.
    """
    out = os.path.abspath(CANARY_BIN) + "." + _src_digest(CANARY_SRC)
    if os.path.exists(out):
        return out
    nvcc = os.path.join(
        os.environ.get("CUDA_HOME") or _ENV.cuda_home() or "", "bin", "nvcc"
    )
    if not os.path.exists(nvcc):
        return None
    src = out + ".cu"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(src, "w") as f:
        f.write(CANARY_SRC)
    p = _run(
        [nvcc, "-arch=sm_70", "-o", out, src, "-ccbin", "/usr/bin/gcc"], timeout=600
    )
    return out if p.returncode == 0 and os.path.exists(out) else None


def live_probe(path):
    """ЖИВАЯ проверка: снял ли кандидат хоть один счётчик с РЕАЛЬНОГО sm_70-запуска.
    -> (True/False/None, пояснение). None = проверить не удалось (нет канарейки/нет прав)."""
    bin_ = build_canary()
    if not bin_:
        return None, "канарейка не собралась (нет nvcc?)"
    args = [
        "--csv",
        "--metrics",
        "smsp__inst_executed.sum",
        "--target-processes",
        "all",
    ]
    # ВАЖНО: бьём В ЭТОГО кандидата напрямую (_exec_ncu), а не через run_ncu -- иначе рекурсия
    # в pick_ncu и проверялся бы совсем другой бинарь.
    try:
        p, _ = _exec_ncu(path, args, [bin_], None, False, 600, None)
        blob = (p.stdout or "") + (p.stderr or "")
        if any(m in blob for m in PERM_MARKERS):
            p, _ = _exec_ncu(path, args, [bin_], None, True, 600, None)
    except NcuError as exc:
        return None, "живая проверка не выполнена: %s" % exc
    except Exception as exc:  # noqa: BLE001
        return None, "живая проверка упала: %s" % exc
    data, _ = parse_csv_metrics(p.stdout or "")
    if data:
        return True, "живая проверка: счётчики с sm_70 сняты"
    blob = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()
    return False, "живая проверка ПРОВАЛЕНА (rc=%d): %s" % (
        p.returncode,
        " | ".join(blob[-2:])[:200] or "пустой вывод",
    )


def probe_binary(path, deep=False):
    """Проверка кандидата. Каждый отказ -- со своей причиной в тексте."""
    c = Cand(path)
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        c.why = "не файл или не исполняемый"
        return c
    try:
        p = _run([path, "--version"], timeout=120)
    except Exception as exc:  # noqa: BLE001
        c.why = "--version не запустился: %s" % exc
        return c
    out = (p.stdout or "") + (p.stderr or "")
    if "not installed" in out.lower():
        # ЭТО ТА САМАЯ ЗАГЛУШКА. Она возвращает 0 и пустой stdout при профилировании.
        c.why = "launcher-ЗАГЛУШКА: 'Nsight Compute is not installed' (даёт ПУСТОЙ профиль, rc=0)"
        return c
    if not out.strip():
        c.why = "--version отдал ПУСТОТУ (rc=%d)" % p.returncode
        return c
    m = VER_RE.search(out)
    if not m:
        c.why = "версия не разобрана: %r" % out.strip().splitlines()[:1]
        return c
    c.version = "%s.%s.%s" % m.groups()
    major, minor = int(m.group(1)), int(m.group(2))

    # ПОДДЕРЖКА sm_70 -- три слоя, и слой, который вынес решение, ПЕЧАТАЕТСЯ.
    #
    # Слой 1 (НЕОБХОДИМЫЙ, НЕ ДОСТАТОЧНЫЙ): знает ли бинарь чип gv100. ЗАМЕРЕНО: 2025.1.1 его
    #   перечисляет и отдаёт 2312 строк метрик -- база метрик Volta помнит, а ЦЕЛЕВАЯ часть уже
    #   нет. Полагаться на этот слой = получить пустой профиль с видом успеха.
    # Слой 2 (РЕШАЮЩИЙ офлайн): версия. Volta снята с поддержки начиная с 2025.1.
    # Слой 3 (ЖИВОЙ, по --deep): реально снять счётчик с канарейки на карте.
    try:
        q = _run([path, "--list-chips"], timeout=180)
        c.sm70 = q.returncode == 0 and "gv100" in (q.stdout or "")
    except Exception:  # noqa: BLE001
        c.sm70 = False
    if not c.sm70:
        c.why = "слой 1: бинарь НЕ ЗНАЕТ чип gv100 (--list-chips)"
        return c
    if (major, minor) >= (2025, 1):
        c.why = (
            "слой 2: версия %s >= 2025.1 -- Volta снята с поддержки "
            "(NB: --list-chips ВСЁ РАВНО печатает gv100, слоя 1 недостаточно)"
            % c.version
        )
        return c
    c.why = "слои 1-2: версия %s, чип gv100 известен" % c.version
    if deep:
        live, why = live_probe(path)
        c.why += "; " + why
        if live is False:
            return c
    c.ok = True
    return c


def discover(deep=False):
    seen, cands = set(), []
    for g in CANDIDATE_GLOBS:
        if not g:
            continue
        for path in sorted(glob.glob(g), reverse=True):  # свежая версия пакета первой
            rp = os.path.realpath(path)
            if rp in seen:
                continue
            seen.add(rp)
            cands.append(probe_binary(path, deep))
    w = shutil.which("ncu")
    if w and os.path.realpath(w) not in seen:
        cands.append(probe_binary(w, deep))
    return cands


_NCU_CACHE = [None]


def pick_ncu(verbose=False):
    if _NCU_CACHE[0]:
        return _NCU_CACHE[0]
    cands = discover()
    if verbose:
        for c in cands:
            print("  " + repr(c))
    good = [c for c in cands if c.ok]
    if not good:
        raise NcuError(
            "рабочего ncu НЕТ. Проверены:\n" + "\n".join("  " + repr(c) for c in cands)
        )
    _NCU_CACHE[0] = good[0].path
    return good[0].path


# ---------------------------------------------------------------------------------------------
# 2. РАЗБОР ЧИСЕЛ -- ПОСИМВОЛЬНО, С ПРОТОКОЛОМ НЕРАЗОБРАННОГО
# ---------------------------------------------------------------------------------------------
def parse_number(raw):
    """-> (значение | None, пометка). Пометка непуста, если строка выглядела не как чистое число.

    Разделители тысяч снимаем ЯВНО и только их (см. THOUSANDS); всё прочее -- повод пометить,
    а не молча выбросить.
    """
    if raw is None:
        return None, "нет поля"
    s = raw.strip().strip('"')
    if s == "" or s in ("n/a", "N/A", "-"):
        return None, "пусто/n-a: %r" % raw
    note = ""
    nbsp = any(ch in s for ch in "    ")
    if nbsp:
        note = "локальный разделитель тысяч (NBSP) -- снят"
    # РОЛЬ ЗАПЯТОЙ РЕШАЕТСЯ, А НЕ УГАДЫВАЕТСЯ. При LC_ALL=C запятая -- разделитель тысяч; но если
    # локаль всё же просочилась, она ДЕСЯТИЧНАЯ, и слепое снятие превращает 1,5 в 15 -- ошибка
    # в 10 раз, которая выглядит как нормальное число. Признак: группа после запятой не из 3 цифр,
    # либо в строке уже есть неразрывный пробел (тогда тысячи разделяет он, а не запятая).
    if "," in s:
        groups = s.split(",")
        thousands_shape = re.fullmatch(
            r"[+-]?\d{1,3}", groups[0].strip()
        ) is not None and all(re.fullmatch(r"\d{3}", g) for g in groups[1:])
        if nbsp or not thousands_shape:
            s = s.replace(",", ".")
            note = (
                (note + "; ") if note else ""
            ) + "ЗАПЯТАЯ РАЗОБРАНА КАК ДЕСЯТИЧНАЯ (локаль просочилась мимо LC_ALL=C)"
    cleaned = "".join(ch for ch in s if ch not in THOUSANDS)
    if not re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", cleaned):
        return None, "не число после чистки: %r -> %r" % (raw, cleaned)
    try:
        return float(cleaned), note
    except ValueError:
        return None, "float() отказал: %r" % cleaned


def parse_csv_metrics(text):
    """`ncu --csv`: строка на (ЗАПУСК, метрика). -> (запуски, неразобранное).

    Ключ -- (ID запуска, имя ядра), а НЕ имя ядра: при нескольких запусках одного ядра разбор
    "по имени" молча оставляет ПОСЛЕДНИЙ и печатает его как итог. Здесь запуски видны по одному
    и складываются явно, а их число попадает в отчёт.
    """
    import csv
    import io

    launches, unparsed = {}, []
    hdr = None
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if hdr is None:
            if "Metric Name" in row and "Kernel Name" in row:
                hdr = {n: i for i, n in enumerate(row)}
            continue
        try:
            kern = row[hdr["Kernel Name"]]
            lid = row[hdr["ID"]] if "ID" in hdr else "0"
            name = row[hdr["Metric Name"]]
            raw = row[hdr["Metric Value"]]
        except (IndexError, KeyError):
            unparsed.append(("строка без нужных колонок", ",".join(row)[:120]))
            continue
        val, note = parse_number(raw)
        if val is None:
            unparsed.append(("%s / %s" % (kern[:40], name), note))
            continue
        if note:
            unparsed.append(("%s / %s [ЗАСЧИТАНО]" % (kern[:40], name), note))
        lz = launches.setdefault(
            (lid, kern),
            {
                "kernel": kern,
                "id": lid,
                "grid": row[hdr["Grid Size"]] if "Grid Size" in hdr else "",
                "block": row[hdr["Block Size"]] if "Block Size" in hdr else "",
                "m": {},
            },
        )
        lz["m"][name] = val
    return launches, unparsed


# ---------------------------------------------------------------------------------------------
# 3. ЗАПУСК ПРОФИЛЯ
# ---------------------------------------------------------------------------------------------
def _sudo_prefix(use_sudo):
    """sudo нужен, т.к. RmProfilingAdminOnly=1. Пароль берём из окружения и НИКОГДА не печатаем."""
    if not use_sudo:
        return [], None
    pw = os.environ.get("TEMPO_SUDO_PASS")
    if not pw:
        raise NcuError(
            "нужен sudo (RmProfilingAdminOnly=1), но TEMPO_SUDO_PASS не задан"
        )
    return ["sudo", "-S", "-p", ""], pw


PERM_MARKERS = (
    "ERR_NVGPUCTRPERM",
    "insufficient permissions",
    "The user does not have permission",
)


def _exec_ncu(ncu, ncu_args, cmd, workload_env, use_sudo, timeout, cwd):
    """Один вызов ncu. Под sudo переменные окружения пробрасываем ЯВНО через env(1):
    sudo сбрасывает PYTHONPATH/CUDA_VISIBLE_DEVICES/FA2SM70_BUILD_DIR, и без этого профилируется
    НЕ ТО (а падение выглядит как 'ядро не найдено')."""
    pre, pw = _sudo_prefix(use_sudo)
    envpairs = []
    if use_sudo:
        keep = dict(FORCED_ENV)
        keep.update(workload_env or {})
        for k in (
            "PYTHONPATH",
            "CUDA_VISIBLE_DEVICES",
            "PATH",
            "HOME",
            "LD_LIBRARY_PATH",
            "CUDA_HOME",
            "FA2SM70_BUILD_DIR",
            "CC",
            "CXX",
            "CUDAHOSTCXX",
            "TORCH_CUDA_ARCH_LIST",
            "TORCH_EXTENSIONS_DIR",
            "FA2SM70_EXTRA_NVCC",
            "TMPDIR",
        ):
            if k in os.environ and k not in keep:
                keep[k] = os.environ[k]
        envpairs = ["env"] + ["%s=%s" % (k, v) for k, v in keep.items()]
    full = pre + envpairs + [ncu] + ncu_args + list(cmd)
    p = subprocess.run(
        full,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_env(workload_env),
        cwd=cwd,
        input=(pw + "\n") if pw else None,
    )
    return p, full


def admin_only():
    """Нужен ли root для счётчиков. ЧИТАЕТСЯ У ДРАЙВЕРА, а не выясняется падением.

    Иначе "auto" сперва гоняет тело БЕЗ прав (ncu запускает приложение целиком и только потом
    отказывает), и каждый замер стоит ДВА прогона тела. На этой машине RmProfilingAdminOnly: 1.
    """
    try:
        with open("/proc/driver/nvidia/params") as f:
            for line in f:
                if line.startswith("RmProfilingAdminOnly:"):
                    return line.split(":")[1].strip() != "0"
    except OSError:
        pass
    return None  # не выяснили -- пусть решает отказ по правам


def run_ncu(ncu_args, cmd, workload_env=None, timeout=3600, cwd=None, sudo="auto"):
    """Запуск с автоматическим повтором под sudo при отказе по правам."""
    ncu = pick_ncu()
    if (
        sudo == "auto"
        and cmd
        and admin_only()
        and os.geteuid() != 0
        and os.environ.get("TEMPO_SUDO_PASS")
    ):
        sudo = True  # спросили драйвер: без root счётчиков не будет
    if sudo is True:
        p, full = _exec_ncu(ncu, ncu_args, cmd, workload_env, True, timeout, cwd)
        return p, full
    p, full = _exec_ncu(ncu, ncu_args, cmd, workload_env, False, timeout, cwd)
    blob = (p.stdout or "") + (p.stderr or "")
    if sudo == "auto" and any(m in blob for m in PERM_MARKERS):
        p, full = _exec_ncu(ncu, ncu_args, cmd, workload_env, True, timeout, cwd)
    return p, full


def _redact(argv):
    return " ".join(a for a in argv if not a.startswith("TEMPO_SUDO_PASS"))


# ---------------------------------------------------------------------------------------------
# 4. ПРОВЕРКА ПРИСУТСТВИЯ
# ---------------------------------------------------------------------------------------------
def require_kernel(launches, kernel_regex, ctx):
    rx = re.compile(kernel_regex)
    hit = [key for key in launches if rx.search(launches[key]["kernel"])]
    if hit:
        return sorted(hit)
    names = sorted({v["kernel"] for v in launches.values()})[:25]
    raise KernelNotFound(
        "ЯДРО %r В ПРОФИЛЕ ОТСУТСТВУЕТ -- это НЕ 'меньше данных', это ДРУГОЙ ОТВЕТ.\n"
        "  разобрано ЗАПУСКОВ: %d; РАЗНЫХ ИМЁН ядер найдено%s\n"
        "  %s\n"
        "  причины, которые дают ровно эту картину: (а) профилировался не тот процесс "
        "(--target-processes), (б) под sudo потерялся PYTHONPATH/FA2SM70_BUILD_DIR и упало тело, "
        "(в) регулярка написана под mangled, а имена demangled (или наоборот), "
        "(г) ncu отфильтровал запуск (--launch-count/--kernel-name)."
        % (
            kernel_regex,
            len(launches),
            (":\n    " + "\n    ".join(names)) if names else " (НИ ОДНОГО)",
            ctx,
        )
    )


# ---------------------------------------------------------------------------------------------
# 5. ФАСАД
# ---------------------------------------------------------------------------------------------
class Result(object):
    def __init__(self):
        self.kernel = ""
        self.wavefronts = 0.0  # маршрут А, штук
        self.conflicts = 0.0  # маршрут А, штук (ДОБАВОЧНЫЕ вайвфронты)
        self.rows = []  # маршрут Б: список dict(addr,line,file,sass,wf,ideal,exc)
        self.by_line = []  # маршрут Б, свёртка по строке .cu/.h
        self.src_wavefronts = 0.0  # маршрут Б, сумма
        self.src_excessive = 0.0  # маршрут Б, сумма
        self.src_instr = 0  # сколько команд разобрано
        self.src_mode = ""  # какой вид страницы source сработал
        self.unparsed = []  # ЧТО НЕ РАЗОБРАНО -- печатать ВСЕГДА
        self.rule = ""  # текст правила самого ncu (независимая сверка)
        self.rule_excessive = None  # число из правила ncu (не из нашего разбора)
        self.rule_total = None
        self.launches = 0  # сколько ЗАПУСКОВ попало под регулярку
        self.total_launches = 0  # сколько запусков вообще разобрано в профиле
        self.per_launch = []  # по каждому: id/grid/block/вайвфронты/конфликты
        self.report = ""  # путь к .ncu-rep
        self.cmdline = ""

    @property
    def fraction(self):
        return self.conflicts / self.wavefronts if self.wavefronts else float("nan")

    @property
    def src_fraction(self):
        return (
            self.src_excessive / self.src_wavefronts
            if self.src_wavefronts
            else float("nan")
        )


def _profile_launch(kernel_regex, cmd, workload_env, cwd, timeout, sudo):
    args = [
        "--csv",
        "--target-processes",
        "all",
        "--kernel-name-base",
        "demangled",
        "--metrics",
        ",".join(M_LAUNCH),
    ]
    p, full = run_ncu(args, cmd, workload_env, timeout, cwd, sudo)
    data, unparsed = parse_csv_metrics(p.stdout or "")
    ctx = "rc=%d, строк stdout=%d, хвост stderr: %s" % (
        p.returncode,
        len((p.stdout or "").splitlines()),
        " | ".join((p.stderr or "").strip().splitlines()[-3:])[:300],
    )
    if not data:
        raise NcuError(
            "ncu не отдал НИ ОДНОЙ разобранной метрики -- пустая таблица это ОТКАЗ, "
            "а не 'конфликтов нет'.\n  %s\n  запуск: %s" % (ctx, _redact(full))
        )
    return data, unparsed, ctx, _redact(full)


def parse_csv_source(text):
    """`ncu -i rep --csv --page source --print-source sass | cuda,sass`.

    ЗАМЕРЕНО, что форматов ДВА, и они разные (это и есть причина, по которой посточечная привязка
    "не приходила через --csv" -- её ищут не на той странице и не тем разбором):

      A) --print-source sass          шапка: "Kernel Name","<имя>"
                                      заголовок начинается с "Address"
                                      строки: одна на КОМАНДУ SASS
      B) --print-source cuda,sass     шапка: "File Path","<путь>" / "Function Name","<имя>"
                                      заголовок начинается с "Line No" и содержит "Source" ДВАЖДЫ
                                      строки чередуются: СТРОКА CUDA (агрегат) и её команды SASS

    В формате Б строка CUDA -- это СУММА своих команд SASS. Сложить и то и другое = удвоить.
    Поэтому суммируем ТОЛЬКО команды (есть адрес), а строки CUDA держим как разметку.

    Одноимённые колонки ("Source" дважды) индексируем ПО ПОЗИЦИИ: dict по имени их схлопывает.
    """
    import csv
    import io

    recs, unparsed = [], []
    hdr = None
    cur_kernel, cur_file, cur_func, cur_line, cur_cuda = "", "", "", "", ""
    i_addr = i_sass = i_line = i_cuda = None
    seen_headers = []

    for row in csv.reader(io.StringIO(text)):
        if not row or not any(c.strip() for c in row):
            continue
        first = row[0].strip()
        if first == "Kernel Name" and len(row) > 1:
            cur_kernel = row[1]
            hdr = None
            continue
        if first == "File Path" and len(row) > 1:
            cur_file = row[1]
            hdr = None
            continue
        if first == "Function Name" and len(row) > 1:
            cur_func = row[1]
            cur_kernel = cur_kernel or row[1]
            hdr = None
            continue
        if first in ("Address", "Line No"):
            hdr = [c.strip() for c in row]
            seen_headers.append(first)
            names = list(hdr)
            i_addr = names.index("Address") if "Address" in names else None
            i_line = names.index("Line No") if "Line No" in names else None
            # первая "Source" при формате Б -- текст CUDA, вторая -- текст SASS
            src_at = [k for k, n in enumerate(names) if n == "Source"]
            if first == "Line No":
                i_cuda = src_at[0] if src_at else None
                i_sass = src_at[1] if len(src_at) > 1 else None
            else:
                i_cuda, i_sass = None, (src_at[0] if src_at else None)
            continue
        if hdr is None:
            unparsed.append(
                ("страница source", "строка до заголовка: " + ",".join(row)[:100])
            )
            continue

        addr = row[i_addr].strip() if (i_addr is not None and i_addr < len(row)) else ""
        lineno = (
            row[i_line].strip() if (i_line is not None and i_line < len(row)) else ""
        )
        if addr == "...":
            # ncu СВОРАЧИВАЕТ участки листинга многоточием. Значений в них нет ("-"), но молчать
            # о них нельзя: это ровно тот случай, когда "разобрано не всё" выглядит как "чисто".
            unparsed.append(("страница source", "ncu свернул участок листинга ('...')"))
            continue
        if lineno and (not addr or addr == "-"):
            cur_line = lineno
            cur_cuda = (
                row[i_cuda].strip()
                if (i_cuda is not None and i_cuda < len(row))
                else ""
            )
            continue  # агрегат по строке CUDA -- разметка, НЕ слагаемое
        if not addr or addr == "-":
            unparsed.append(
                (
                    "страница source",
                    "строка без адреса и без номера: " + ",".join(row)[:100],
                )
            )
            continue
        rec = {
            "_kernel": cur_kernel,
            "_file": cur_file,
            "_func": cur_func,
            "_line": cur_line,
            "_cuda": cur_cuda,
            "addr": addr,
            "sass": (
                row[i_sass].strip()
                if (i_sass is not None and i_sass < len(row))
                else ""
            ),
        }
        for k, name in enumerate(hdr):
            if k in (i_addr, i_sass, i_line, i_cuda):
                continue
            rec[name] = row[k] if k < len(row) else ""
        recs.append(rec)
    return recs, (seen_headers[0] if seen_headers else None), unparsed


def _num_col(rec, names):
    for n in names:
        if n in rec:
            v, note = parse_number(rec[n])
            if v is not None:
                return v, note
            return None, note
    return None, "нет колонки из %s" % (names,)


def warmup(cmd, workload_env=None, cwd=None, timeout=7200):
    """ПРОГОН БЕЗ ПРОФИЛИРОВЩИКА ПЕРЕД ЗАМЕРОМ -- не роскошь, а необходимость.

    ЗАМЕРЕНО на этом же наряде: тело (`bankaudit.py`) при первом запуске ДОСОБИРАЕТ расширение
    через ninja/nvcc. Под `--target-processes all` профилировщик цепляется и к ним, прогон
    растягивается на десятки минут, а замер идёт по процессу, который большую часть жизни
    компилировал. Один холостой запуск снимает это целиком.
    """
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_env(workload_env),
        cwd=cwd,
    )
    return p.returncode, (p.stderr or "").strip().splitlines()[-3:]


def conflicts(
    kernel_regex,
    cmd,
    workload_env=None,
    cwd=None,
    timeout=3600,
    sudo="auto",
    report=None,
    per_line=True,
    keep_report=True,
    do_warmup=True,
):
    """ГЛАВНЫЙ ФАСАД.

    kernel_regex -- ОБЯЗАТЕЛЕН; если такого ядра в профиле нет, кидается KernelNotFound.
    -> Result: доля конфликтов (маршрут А) + построчная таблица (маршрут Б) + НЕРАЗОБРАННОЕ.
    """
    r = Result()
    if do_warmup:
        rc, tail = warmup(cmd, workload_env, cwd)
        if rc != 0:
            # Тело падает БЕЗ профилировщика -- значит и под ним упадёт, и "ядро не найдено"
            # будет ложной причиной. Сказать это СЕЙЧАС, а не после часа replay.
            raise NcuError(
                "тело упало ещё БЕЗ профилировщика (rc=%d):\n  %s\n"
                "  ncu тут ни при чём; чинить надо запуск." % (rc, " | ".join(tail))
            )
    launches, unparsed, ctx, shown = _profile_launch(
        kernel_regex, cmd, workload_env, cwd, timeout, sudo
    )
    r.unparsed += unparsed
    r.cmdline = shown
    r.total_launches = len(launches)
    hits = require_kernel(launches, kernel_regex, ctx)
    r.kernel = launches[hits[0]]["kernel"]
    r.launches = len(hits)
    for key in hits:
        lz = launches[key]
        m = lz["m"]
        for name in M_LAUNCH:
            if name not in m:
                r.unparsed.append(
                    (
                        "запуск %s / %s" % (lz["id"], lz["kernel"][:30]),
                        "метрика %s не пришла" % name,
                    )
                )
        r.wavefronts += m.get(M_LAUNCH[0], 0.0) + m.get(M_LAUNCH[1], 0.0)
        r.conflicts += m.get(M_LAUNCH[2], 0.0) + m.get(M_LAUNCH[3], 0.0)
        r.per_launch.append(
            {
                "id": lz["id"],
                "kernel": lz["kernel"],
                "grid": lz["grid"],
                "block": lz["block"],
                "wf": m.get(M_LAUNCH[0], 0.0) + m.get(M_LAUNCH[1], 0.0),
                "cf": m.get(M_LAUNCH[2], 0.0) + m.get(M_LAUNCH[3], 0.0),
            }
        )
    if len(hits) > 1:
        r.unparsed.append(
            (
                "ЗАПУСКОВ ПОД РЕГУЛЯРКУ: %d" % len(hits),
                "числа СЛОЖЕНЫ по всем запускам; сами запуски -- в таблице 'по запускам'",
            )
        )
    if per_line:
        _attach_source(
            r, kernel_regex, cmd, workload_env, cwd, timeout, sudo, report, keep_report
        )
    return r


def _attach_source(
    r, kernel_regex, cmd, workload_env, cwd, timeout, sudo, report, keep
):
    """Маршрут Б: посточечная привязка. Через `--csv` она НЕ приходит -- нужен .ncu-rep + --import."""
    rep = report or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data",
        "ncu",
        "src_%d.ncu-rep" % os.getpid(),
    )
    rep = os.path.abspath(rep)
    os.makedirs(os.path.dirname(rep), exist_ok=True)
    args = [
        "--target-processes",
        "all",
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        "regex:" + kernel_regex,
        "--section",
        "SourceCounters",
        "--import-source",
        "yes",
        "-f",
        "-o",
        rep[: -len(".ncu-rep")] if rep.endswith(".ncu-rep") else rep,
    ]
    p, full = run_ncu(args, cmd, workload_env, timeout, cwd, sudo)
    if not os.path.exists(rep):
        r.unparsed.append(
            (
                "ПОСТРОЧНАЯ ПРИВЯЗКА",
                "отчёт %s не создан (rc=%d): %s"
                % (
                    rep,
                    p.returncode,
                    " | ".join((p.stderr or "").strip().splitlines()[-2:])[:200],
                ),
            )
        )
        return
    _chown_back(rep)
    r.report = rep
    if "No source files were imported" in ((p.stdout or "") + (p.stderr or "")):
        r.unparsed.append(
            (
                "ПРИВЯЗКА К СТРОКЕ .cu",
                "ядро собрано БЕЗ -lineinfo: привязка будет только к АДРЕСУ SASS. "
                "Пересобрать: FA2SM70_EXTRA_NVCC=-lineinfo",
            )
        )
    # 5.1 сама привязка. Сперва cuda,sass (даёт файл+строку), при неудаче -- голый sass.
    rows = []
    for mode in ("cuda,sass", "sass"):
        imp = [
            "-i",
            rep,
            "--csv",
            "--page",
            "source",
            "--print-source",
            mode,
            "--print-units",
            "base",
        ]
        q, _ = run_ncu(imp, [], workload_env, timeout, cwd, sudo=False)
        rows, hdr, up = parse_csv_source(q.stdout or "")
        if rows:
            r.unparsed += up
            r.src_mode = mode
            break
        r.unparsed.append(
            (
                "ПОСТРОЧНАЯ ПРИВЯЗКА (--print-source %s)" % mode,
                "строк не разобрано (rc=%d): %s"
                % (
                    q.returncode,
                    " | ".join((q.stderr or "").strip().splitlines()[-2:])[:200],
                ),
            )
        )
    if not rows:
        return
    _fold_source_rows(r, rows, hdr)
    # 5.2 НЕЗАВИСИМАЯ СВЕРКА: правило UncoalescedSharedAccess самого ncu. Оно считает избыточные
    # вайвфронты своим кодом по своим метрикам -- совпадение с нашей суммой означает, что разбор
    # верен не "по нашей же логике". Правило МОЛЧИТ, когда избыточных нет: это не отказ.
    d, _ = run_ncu(
        ["-i", rep, "--page", "details"], [], workload_env, timeout, cwd, sudo=False
    )
    blob = (d.stdout or "").replace("\n", " ")
    m = re.search(
        r"total of\s+([\d\s.,]+?)\s+excessive wavefronts\s*\(\s*(\d+)%\s*of the total\s+"
        r"([\d\s.,]+?)\s+wavefronts",
        blob,
    )
    if m:
        got_exc, _ = parse_number(m.group(1))
        got_tot, _ = parse_number(m.group(3))
        r.rule = "правило ncu: избыточных %s (%s%% от %s)" % (
            m.group(1).strip(),
            m.group(2),
            m.group(3).strip(),
        )
        r.rule_excessive, r.rule_total = got_exc, got_tot
    else:
        for line in (d.stdout or "").splitlines():
            if "excessive" in line.lower() and "wavefront" in line.lower():
                r.rule = line.strip()
                break


WF_COLS = ("L1 Wavefronts Shared", "memory_l1_wavefronts_shared")
ID_COLS = ("L1 Wavefronts Shared Ideal", "memory_l1_wavefronts_shared_ideal")
EX_COLS = (
    "L1 Wavefronts Shared Excessive",
    "derived__memory_l1_wavefronts_shared_excessive",
)
NW_COLS = ("L1 Conflicts Shared N-Way", "derived__memory_l1_conflicts_shared_nway")


def _fold_source_rows(r, rows, hdr):
    """Свернуть команды SASS в таблицу и в суммы. Считаем ТОЛЬКО команды (не агрегаты CUDA)."""
    have_col = False
    for rec in rows:
        wf, note = _num_col(rec, WF_COLS)
        if wf is None:
            if note and "нет колонки" in note:
                continue
            r.unparsed.append(
                ("точка %s" % rec.get("addr", "?"), "вайвфронты: " + note)
            )
            continue
        have_col = True
        ideal, _ = _num_col(rec, ID_COLS)
        exc, _ = _num_col(rec, EX_COLS)
        nway, _ = _num_col(rec, NW_COLS)
        if exc is None:
            exc = (wf - ideal) if ideal is not None else 0.0
        r.src_wavefronts += wf
        r.src_excessive += exc
        r.src_instr += 1
        if wf <= 0 and exc <= 0:
            continue  # команда не трогает разделяемую память
        r.rows.append(
            {
                "addr": rec["addr"],
                "sass": rec["sass"],
                "file": os.path.basename(rec.get("_file", "") or ""),
                "line": rec.get("_line", ""),
                "cuda": rec.get("_cuda", ""),
                "space": rec.get("Address Space", ""),
                "op": rec.get("Access Operation", ""),
                "size": rec.get("Access Size", ""),
                "nway": nway,
                "wf": wf,
                "ideal": ideal,
                "exc": exc,
            }
        )
    r.rows.sort(key=lambda x: -x["exc"])
    if not have_col:
        r.unparsed.append(
            (
                "ПОСТРОЧНАЯ ПРИВЯЗКА",
                "колонки вайвфронтов нет; тип заголовка: %r" % (hdr,),
            )
        )
        return
    # свёртка по строке исходника (если -lineinfo есть) -- то, к чему прикладывают правку
    by = {}
    for rec in r.rows:
        key = (
            ("%s:%s" % (rec["file"], rec["line"])) if rec["line"] else "(нет -lineinfo)"
        )
        agg = by.setdefault(
            key,
            {
                "key": key,
                "wf": 0.0,
                "exc": 0.0,
                "n": 0,
                "cuda": rec["cuda"],
                "sass": rec["sass"],
            },
        )
        agg["wf"] += rec["wf"]
        agg["exc"] += rec["exc"]
        agg["n"] += 1
    r.by_line = sorted(by.values(), key=lambda x: -x["exc"])


def _chown_back(path):
    """Отчёт, созданный под sudo, принадлежит root -- вернуть владельцу, иначе --import под ним же."""
    try:
        if os.stat(path).st_uid == 0 and os.geteuid() != 0:
            pw = os.environ.get("TEMPO_SUDO_PASS")
            if pw:
                subprocess.run(
                    [
                        "sudo",
                        "-S",
                        "-p",
                        "",
                        "chown",
                        "%d:%d" % (os.getuid(), os.getgid()),
                        path,
                    ],
                    input=pw + "\n",
                    capture_output=True,
                    text=True,
                    env=_env(),
                )
    except OSError:
        pass


# ---------------------------------------------------------------------------------------------
# 6. ПЕЧАТЬ
# ---------------------------------------------------------------------------------------------
def report_text(r, top=25):
    out = []
    out.append(
        "ИТОГ (маршрут А, счётчики l1tex):\n"
        "  ядро            %s\n"
        "  запусков        %d под регулярку из %d разобранных в профиле\n"
        "  вайвфронтов     %.6g\n"
        "  конфликтов      %.6g\n"
        "  ДОЛЯ            %.2f %%\n"
        % (
            r.kernel[:70],
            r.launches,
            r.total_launches,
            r.wavefronts,
            r.conflicts,
            100.0 * r.fraction,
        )
    )
    if len(r.per_launch) > 1:
        out.append("  по запускам:")
        for pl in r.per_launch[:12]:
            out.append(
                "    id=%-4s grid=%-14s block=%-12s вф=%10.6g конф=%10.6g  %.1f %%"
                % (
                    pl["id"],
                    pl["grid"][:14],
                    pl["block"][:12],
                    pl["wf"],
                    pl["cf"],
                    100.0 * pl["cf"] / pl["wf"] if pl["wf"] else float("nan"),
                )
            )
    if r.rows:
        out.append(
            "СВЕРКА С ПОСТРОЧНЫМ (маршрут Б, SourceCounters):\n"
            "  сумма вайвфронтов  %.6g   (невязка к А: %+.3f %%)\n"
            "  сумма избыточных   %.6g   (невязка к А: %+.3f %%)\n"
            "  ДОЛЯ построчная    %.2f %%\n"
            % (
                r.src_wavefronts,
                100.0 * (r.src_wavefronts - r.wavefronts) / r.wavefronts
                if r.wavefronts
                else float("nan"),
                r.src_excessive,
                100.0 * (r.src_excessive - r.conflicts) / r.conflicts
                if r.conflicts
                else float("nan"),
                100.0 * r.src_fraction,
            )
        )
        out.append(
            "  разобрано команд   %d  (страница source: --print-source %s)"
            % (r.src_instr, r.src_mode)
        )
        gap = abs(r.src_excessive - r.conflicts) / r.conflicts if r.conflicts else 0.0
        if gap > 0.01:
            out.append(
                "  ВНИМАНИЕ: А и Б считают РАЗНОЕ и расходиться ОБЯЗАНЫ, а не только от ошибок\n"
                "  разбора. А = l1tex__data_bank_conflicts (добавочные вайвфронты ИМЕННО от\n"
                "  конфликта банков). Б = вайвфронты минус ИДЕАЛ, а идеал ncu считает по ШИРИНЕ\n"
                "  доступа. Где ширина сама по себе требует >1 вайвфронта (LDS.64/128), Б\n"
                "  засчитывает часть ширины в 'избыточное'. Какая из двух цифр воспроизводит\n"
                "  ЯКОРЬ -- решает правило ncu ниже: оно считает ровно Б."
            )
        out.append(
            "\nГДЕ ИМЕННО -- ПО КОМАНДАМ (убывание избыточных вайвфронтов, top %d из %d "
            "трогающих разделяемую память):" % (top, len(r.rows))
        )
        out.append(
            "%-16s %11s %11s %11s %5s %-7s %-6s %s"
            % (
                "адрес",
                "вайвфр",
                "идеал",
                "ИЗБЫТ",
                "N-way",
                "оп",
                "стр",
                "команда SASS",
            )
        )
        for rec in r.rows[:top]:
            out.append(
                "%-16s %11.0f %11s %11.0f %5s %-7s %-6s %s"
                % (
                    rec["addr"][-16:],
                    rec["wf"],
                    ("%.0f" % rec["ideal"]) if rec["ideal"] is not None else "-",
                    rec["exc"],
                    ("%.1f" % rec["nway"]) if rec["nway"] is not None else "-",
                    (rec["op"] or "-")[:7],
                    (rec["line"] or "-")[:6],
                    rec["sass"].strip()[:52],
                )
            )
        if r.by_line:
            out.append(
                "\nГДЕ ИМЕННО -- ПО СТРОКЕ ИСХОДНИКА (top %d из %d):"
                % (top, len(r.by_line))
            )
            out.append(
                "%-34s %6s %11s %11s  %s"
                % ("файл:строка", "команд", "вайвфр", "ИЗБЫТ", "текст")
            )
            for agg in r.by_line[:top]:
                out.append(
                    "%-34s %6d %11.0f %11.0f  %s"
                    % (
                        agg["key"][-34:],
                        agg["n"],
                        agg["wf"],
                        agg["exc"],
                        (agg["cuda"] or agg["sass"]).strip()[:60],
                    )
                )
    else:
        out.append(
            "ПОСТРОЧНОЙ ПРИВЯЗКИ НЕТ -- см. раздел НЕ РАЗОБРАНО ниже. Доля выше остаётся\n"
            "верной как ИТОГ, но ПРИЛОЖИТЬ ЕЁ НЕ К ЧЕМУ."
        )
    if r.rule:
        out.append(
            "\nНЕЗАВИСИМАЯ СВЕРКА (правило ncu считает своим кодом по своим метрикам):"
        )
        out.append("  " + r.rule)
        if r.rule_excessive is not None and r.conflicts:
            out.append(
                "  расхождение с нашим итогом: избыточные %+.3f %%, вайвфронты %+.3f %%"
                % (
                    100.0 * (r.rule_excessive - r.conflicts) / r.conflicts,
                    100.0 * (r.rule_total - r.wavefronts) / r.wavefronts
                    if (r.rule_total and r.wavefronts)
                    else float("nan"),
                )
            )
    else:
        out.append(
            "\nНЕЗАВИСИМАЯ СВЕРКА: правило ncu про избыточные вайвфронты не сработало.\n"
            "  Это НЕ отказ, если избыточных нет вовсе (правило молчит при нуле), и ОТКАЗ,\n"
            "  если избыточные выше ненулевые -- тогда сверять нечем."
        )
    out.append("\nНЕ РАЗОБРАНО (%d):" % len(r.unparsed))
    if not r.unparsed:
        out.append("  -- пусто.")
    for what, why in r.unparsed[:60]:
        out.append("  %-46s %s" % (what, why))
    if len(r.unparsed) > 60:
        out.append("  ... ещё %d" % (len(r.unparsed) - 60))
    out.append(
        "\nПУСТОЙ СПИСОК ПОДОЗРЕНИЙ ПРИ НЕПУСТОМ СПИСКЕ НЕРАЗОБРАННОГО НЕ ОЗНАЧАЕТ 'ЧИСТО'."
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# 7. САМОПРОВЕРКА (ЯКОРЬ)
# ---------------------------------------------------------------------------------------------
ANCHOR = {
    "kernel": r"attention_kernel_backward",
    "wavefronts": 4925440.0,
    "conflicts": 1703936.0,
    "fraction": 0.346,
    "tol": 0.02,  # 2 % относительной невязки: счётчики детерминированы, но не побитово
}

# ВТОРОЙ якорь -- СИНТЕТИЧЕСКИЙ, из АРИФМЕТИКИ, а не из прошлого замера (см. tools/
# ncu_conflict_canary.cu). Он не зависит ни от сборки ядер FA2, ни от чьей-либо памяти о числе.
SYNTH = {
    "tempo_bank_dirty": {
        "wavefronts": 64.0,
        "conflicts": 62.0,
        "nway": 32.0,
        "почему": "шаг 32 слова -> весь столбец в банк 0: 32 вайвфронта на "
        "команду при идеале 1, две команды (STS+LDS)",
    },
    "tempo_bank_clean": {
        "wavefronts": 2.0,
        "conflicts": 0.0,
        "nway": 1.0,
        "почему": "шаг 33 слова -> столбец на 32 разных банка: 1 вайвфронт на "
        "команду, две команды",
    },
}


def build_synth():
    """ИМЯ СОБРАННОГО НЕСЁТ ОТПЕЧАТОК ИСХОДНИКА (LAW=L-CACHE-KEY-BY-CONTENT): «файл есть» не
    означает «собрано из ЭТОГО исходника», а прибор меряет то, что лежит."""
    src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ncu_conflict_canary.cu"
    )
    try:
        with open(src, encoding="utf-8", errors="replace") as fh:
            stamp = _src_digest(fh.read())
    except OSError as e:
        raise NcuError("исходник канарейки конфликтов не читается: %s" % e)
    out = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "build",
            "ncu_conflict_canary." + stamp,
        )
    )
    if os.path.exists(out):
        return out
    nvcc = os.path.join(
        os.environ.get("CUDA_HOME") or _ENV.cuda_home() or "", "bin", "nvcc"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    p = _run(
        [
            nvcc,
            "-arch=sm_70",
            "-lineinfo",
            "-Wno-deprecated-gpu-targets",
            "-o",
            out,
            src,
            "-ccbin",
            "/usr/bin/gcc",
        ],
        timeout=600,
    )
    if p.returncode != 0:
        raise NcuError("канарейка конфликтов не собралась:\n" + (p.stderr or "")[-800:])
    return out


def selftest_synth(verbose=True):
    """САМОПРОВЕРКА БЕЗ ВНЕШНИХ ЗАВИСИМОСТЕЙ: числа известны из арифметики банков."""
    binp = build_synth()
    ok = True
    print("=" * 96)
    print("СИНТЕТИЧЕСКИЙ ЯКОРЬ (предсказан арифметикой 32 банков, не памятью о замере)")
    for kern, want in SYNTH.items():
        r = conflicts(
            kern,
            [binp],
            workload_env={
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            },
        )
        if verbose:
            print("\n" + report_text(r, top=6))
        nway = max(
            [x["nway"] for x in r.rows if x["nway"] is not None] or [float("nan")]
        )
        print(
            "\n  %-18s ждали вф=%g конф=%g N-way=%g | получили вф=%g конф=%g N-way=%g  -> %s"
            % (
                kern,
                want["wavefronts"],
                want["conflicts"],
                want["nway"],
                r.wavefronts,
                r.conflicts,
                nway,
                "СОШЛОСЬ"
                if (
                    r.wavefronts == want["wavefronts"]
                    and r.conflicts == want["conflicts"]
                )
                else "НЕ СОШЛОСЬ",
            )
        )
        print("  почему именно столько: " + want["почему"])
        ok &= r.wavefronts == want["wavefronts"] and r.conflicts == want["conflicts"]
    print("=" * 96)
    print(
        "ИТОГ СИНТЕТИЧЕСКОГО ЯКОРЯ: " + ("ВОСПРОИЗВЕДЁН" if ok else "НЕ ВОСПРОИЗВЕДЁН")
    )
    return 0 if ok else 2


def selftest(body="bwd", bankaudit=None, py=None, build_dir=None, verbose=True):
    py = py or os.environ.get("TEMPO_PY") or _ENV.python_vllm() or "python3"
    # Тело якоря лежит рядом с этим файлом.  ЗДЕСЬ БЫЛ ДЕФЕКТ: путь стоял как
    # "./.cachescratchpad/bankaudit.py" -- след чужой пакетной правки.  Он не всплывал, потому
    # что самопроверка падала РАНЬШЕ, на интерпретаторе из /opt/conda.  Правило проекта:
    # читать ПЕРВУЮ строку трассы -- вторая ошибка прячется за первой.
    bankaudit = bankaudit or os.environ.get(
        "TEMPO_BANKAUDIT",
        os.path.join(_HERE_DIR, "bankaudit.py"),
    )
    wenv = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "PYTHONPATH": os.environ.get(
            "PYTHONPATH", "../VLLM_fa2/solutions/fa2_sm70_cutlass_grade"
        ),
    }
    if build_dir or os.environ.get("FA2SM70_BUILD_DIR"):
        wenv["FA2SM70_BUILD_DIR"] = build_dir or os.environ["FA2SM70_BUILD_DIR"]
    r = conflicts(ANCHOR["kernel"], [py, bankaudit, "--run", body], workload_env=wenv)
    if verbose:
        print(report_text(r))
    ok = True
    print("\n" + "=" * 96)
    print(
        "ЯКОРЬ (независимый замер: bankaudit.py --run bwd, attention_kernel_backward)"
    )
    for key, got in (
        ("wavefronts", r.wavefronts),
        ("conflicts", r.conflicts),
        ("fraction", r.fraction),
    ):
        want = ANCHOR[key]
        d = abs(got - want) / want if want else float("inf")
        hit = d <= ANCHOR["tol"]
        ok &= hit
        print(
            "  %-11s ждали %-12.6g получили %-12.6g расхождение %+.3f %%   %s"
            % (
                key,
                want,
                got,
                100.0 * (got - want) / want,
                "СОШЛОСЬ" if hit else "НЕ СОШЛОСЬ",
            )
        )
    print(
        "ИТОГ ЯКОРЯ: "
        + ("ВОСПРОИЗВЕДЁН" if ok else "НЕ ВОСПРОИЗВЕДЁН -- инструмент НЕ ПРИНИМАТЬ")
    )
    print("=" * 96)
    return 0 if ok else 2


# ---------------------------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="обёртка над ncu, которая не врёт")
    ap.add_argument(
        "--which", action="store_true", help="показать всех кандидатов и причины отказа"
    )
    ap.add_argument(
        "--deep",
        action="store_true",
        help="к --which: ЖИВАЯ проверка кандидата на канарейке (нужен sudo и карта)",
    )
    ap.add_argument(
        "--selftest", action="store_true", help="прогнать якорь (bwd, 34.6 %%)"
    )
    ap.add_argument(
        "--selftest-synth",
        action="store_true",
        help="синтетический якорь: 32-way конфликт, числа из арифметики",
    )
    ap.add_argument("--body", default="bwd", help="тело для --selftest")
    ap.add_argument("--kernel", help="регулярка имени ядра (ОБЯЗАТЕЛЬНА для замера)")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument(
        "--json", help="сложить весь результат в файл JSON (для склейки с другими)"
    )
    ap.add_argument(
        "--no-source", action="store_true", help="только итог, без построчной привязки"
    )
    ap.add_argument(
        "--no-warmup",
        action="store_true",
        help="не прогонять тело вхолостую перед замером (по умолчанию прогоняется: "
        "иначе ncu профилирует ninja/nvcc досборки)",
    )
    ap.add_argument("--report", help="куда положить .ncu-rep")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- команда тела")
    a = ap.parse_args(argv)

    if a.which:
        for c in discover(a.deep):
            print(repr(c))
        try:
            print("\nВЫБРАН: " + pick_ncu())
        except NcuError as exc:
            print("\n" + str(exc))
            return 2
        return 0
    if a.selftest_synth:
        return selftest_synth()
    if a.selftest:
        return selftest(a.body)
    cmd = [x for x in a.cmd if x != "--"]
    if not a.kernel or not cmd:
        ap.error(
            "нужны --kernel РЕГУЛЯРКА и -- КОМАНДА (проверка присутствия ОБЯЗАТЕЛЬНА)"
        )
    r = conflicts(
        a.kernel,
        cmd,
        per_line=not a.no_source,
        report=a.report,
        do_warmup=not a.no_warmup,
    )
    print(report_text(r, a.top))
    if a.json:
        import json

        with open(a.json, "w") as f:
            json.dump(
                {
                    "kernel": r.kernel,
                    "launches": r.launches,
                    "total_launches": r.total_launches,
                    "wavefronts": r.wavefronts,
                    "conflicts": r.conflicts,
                    "fraction": r.fraction,
                    "src_wavefronts": r.src_wavefronts,
                    "src_excessive": r.src_excessive,
                    "src_instr": r.src_instr,
                    "src_mode": r.src_mode,
                    "rule": r.rule,
                    "rule_excessive": r.rule_excessive,
                    "rule_total": r.rule_total,
                    "per_launch": r.per_launch,
                    "rows": r.rows,
                    "by_line": r.by_line,
                    "unparsed": r.unparsed,
                    "report": r.report,
                    "cmdline": r.cmdline,
                },
                f,
                ensure_ascii=False,
                indent=1,
            )
        print("\nJSON: " + a.json)
    return 0


# ---------------------------------------------------------------------------------------------
# СЛЕПЫЕ ЗОНЫ (чего инструмент НЕ видит) -- держать в синхроне с README.
# ---------------------------------------------------------------------------------------------
# 1. ВРЕМЯ. Ни один счётчик здесь не измеряет длительность. Доля конфликтов -- верхняя оценка
#    выигрыша ПО ТРАФИКУ разделяемой памяти; во время она переходит ровно настолько, насколько
#    разделяемая память связывает. Ядро может иметь 90 % конфликтов и не ускориться ни на такт.
# 2. ФАЗЫ. Числа сложены по всему запуску ядра. Разложения по фазам (мейнлуп / эпилог / softmax)
#    ncu не даёт; для него нужен фальсификатор (сборка заведомо неверного ядра со снятой фазой).
# 3. ПРИЧИНА конфликта. Виден АДРЕС команды и её избыточные вайвфронты; ПОЧЕМУ шаг раскладки
#    кратен 32 словам -- не виден. Это работа статического линтера (tools/smem_lint.py).
# 4. CUDA-СТРОКА без -lineinfo. Отгруженные ядра собираются БЕЗ -lineinfo, поэтому привязка идёт
#    к АДРЕСУ SASS, а не к строке .cu/.h. Пересборка с FA2SM70_EXTRA_NVCC=-lineinfo включает
#    колонки файла/строки; без неё колонка "команда / строка" содержит SASS.
# 5. НЕ-shared. Конфликты банков РЕГИСТРОВОГО файла, разделение LSU с global/local, а также
#    l1tex-конфликты по пути tex здесь не считаются.
# 6. МНОЖЕСТВЕННЫЕ ЗАПУСКИ одного ядра складываются; разброс между запусками не показывается.
# 7. СОСЕД ПО КАРТЕ. Счётчики принадлежат нашему запуску и от соседа не зависят, НО сериализация
#    ncu (replay) и чужая нагрузка на ту же карту растягивают ПРОГОН по времени -- не по числам.

if __name__ == "__main__":
    sys.exit(main())
