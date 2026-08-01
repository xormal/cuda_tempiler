# -*- coding: utf-8 -*-
"""ЛИНТЕР РАСКЛАДОК РАЗДЕЛЯЕМОЙ ПАМЯТИ ДЛЯ sm_70 -- БОЕВАЯ ВЕРСИЯ.

ЗАЧЕМ ОН ЕСТЬ. Цена разделяемой памяти на Volta мерится ВАЙВФРОНТАМИ, конвейер отдаёт один
вайвфронт за такт на SM, и при ЗАМОРОЖЕННОМ объёме данных одна только раскладка двигает цену в 32
раза. `ncu` на Volta наполовину отказал (2025.x архитектуру не поддерживает; у 2024.1 CSV ломается
о локаль; посточечной привязки конфликтов не получено вовсе), а конфликт вычисляется ТОЧНО из
исходника -- ни карты, ни sudo, ни версии профилировщика для этого не нужно.

ЗАКОН (замерен на стенде, сверен ncu 17/17 -- data/mio_wavefronts.txt):

    вайвфронтов на команду = max( КОНФЛИКТНОСТЬ , ceil(уникальных_байт/128) , ширина_на_полосу/8Б , 1 )

    КОНФЛИКТНОСТЬ = максимум по банкам числа РАЗЛИЧНЫХ адресов слов, попавших в этот банк
                    (совпадающие адреса -- многоадресность, она бесплатна ПОЛНОСТЬЮ)

и ключевое различение, без которого линтер врёт:

    ПОЛ  = max( ceil(уникальных_байт/128), ширина/8Б, 1 )   -- НЕУСТРАНИМО (объём и обратный путь)
    ВОЗВРАТИМО = вайвфронты - ПОЛ                           -- вот это снимает дополнение раскладки

Именно ПОЛ делает декод чистым: там полоса пишет 16 Б подряд, 32 полосы дают 512 уникальных байт,
и четыре вайвфронта -- это минимальный ТРАФИК, а не конфликт. Линтер без этого различения указал бы
на декод (замер: 1.2 %, чисто) и тем обесценил бы все свои остальные указания.

ДВА ЗАДНИКА, ПОТОМУ ЧТО ОДНОГО НЕ ХВАТАЕТ.
  SRC     -- разбор исходника: символьная развёртка индексного выражения по полосам 0..31.
             Берёт `extern __shared__` с ручной арифметикой смещений, псевдонимы-указатели,
             шаблонные шаги вида `BK + PAD`, предикаты вида `if ((lane & 6) == 0)`.
  CUTLASS -- на cutlass-ядрах SRC СЛЕП ПО УСТРОЙСТВУ: индексного выражения в тексте нет, адрес
             считает функтор раскладки. Поэтому линтер инстанцирует НАСТОЯЩИЙ cutlass::layout::*
             в хостовой программе (tools/smem_lint_cutlass.cpp, g++, без GPU и без nvcc) и берёт
             адреса у него. Свиззлованные раскладки помечаются ОТДЕЛЬНОЙ категорией «НЕ ТРОГАТЬ»:
             они бесконфликтны по построению, и ручное дополнение их СЛОМАЕТ.

ЧЕСТНОСТЬ. Неполный инструмент даёт не «меньше данных», а ДРУГОЙ ОТВЕТ, и звучит он так же уверенно.
Поэтому раздел «НЕ РАЗОБРАНО» обязателен и печатается ВСЕГДА. Пустой список подозрений при непустом
списке неразобранного НЕ означает «чисто».

ЗАПУСК:
    python tools/smem_lint.py --selftest          # закон против замера + якорь по ядрам
    python tools/smem_lint.py <корень>            # полный отчёт по дереву
    python tools/smem_lint.py --kernel volta_fwd_ws <корень>
"""

import math
import os
import re
import subprocess
import sys

WORD = 4
BANKS = 32
LANES = 32

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPO = os.path.dirname(HERE)
DEFAULT_ROOT = "../VLLM_fa2/solutions/fa2_sm70_cutlass_grade"
CUTLASS_INC = os.path.join(DEFAULT_ROOT, "fa2_src/cutlass/include")
CUDA_INC = "/opt/conda/miniconda3/envs/cuda128/targets/x86_64-linux/include"

SIZEOF = {
    "char": 1,
    "int8_t": 1,
    "uint8_t": 1,
    "unsigned char": 1,
    "signed char": 1,
    "half": 2,
    "__half": 2,
    "cutlass::half_t": 2,
    "half_t": 2,
    "short": 2,
    "uint16_t": 2,
    "int16_t": 2,
    "__nv_bfloat16": 2,
    "float": 4,
    "int": 4,
    "unsigned": 4,
    "uint32_t": 4,
    "int32_t": 4,
    "unsigned int": 4,
    "__half2": 4,
    "half2": 4,
    "double": 8,
    "int64_t": 8,
    "uint64_t": 8,
    "float2": 8,
    "long": 8,
    "long long": 8,
    "uint2": 8,
    "float4": 16,
    "int4": 16,
    "uint4": 16,
}

# ------------------------------------------------------------------------------------------------
# 1. ЗАКОН
# ------------------------------------------------------------------------------------------------


def cost(addr_words, width_bytes):
    """Цена одного обращения. addr_words -- адрес ПЕРВОГО слова каждой активной полосы (None = не
    участвует). Возвращает (конфликтность, пол, вайвфронты, возвратимо, число активных полос).

    Совпадающие адреса схлопываются -- многоадресность на Volta бесплатна ПОЛНОСТЬЮ (замерено:
    при шаге 128 Б и DUP 1->32 цена идёт 2048->1024->...->64.8, ровно 32/DUP, насыщения нет).
    """
    active = [a for a in addr_words if a is not None]
    if not active:
        return None
    per_bank = {}
    for a in active:
        per_bank.setdefault(a % BANKS, set()).add(a)
    degree = max(len(s) for s in per_bank.values())
    uniq_bytes = len(set(active)) * width_bytes
    floor = max(math.ceil(uniq_bytes / 128.0), width_bytes / 8.0, 1.0)
    wf = max(float(degree), floor)
    return degree, floor, wf, wf - floor, len(active)


def selftest_law(path=None):
    """ЯКОРЬ №1: закон против стенда. data/mio_wavefronts.txt -- 100+ точек, из них 17 сверены ncu.

    Тело вида k8x64.st128d2: ширина 8 Б на полосу, шаг строки 128 Б, DUP 2 (две полосы на строку).
    """
    path = path or os.path.join(TEMPO, "data", "mio_wavefronts.txt")
    if not os.path.exists(path):
        return None, [f"нет файла замера {path}"]
    ok = bad = 0
    fails = []
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 10 or f[0] == "тело":
            continue
        try:
            width, dup, strb = int(f[2]), int(f[3]), int(f[4])
            model_wf = float(f[7])
            ncu = f[11] if len(f) > 11 else "-"
        except (ValueError, IndexError):
            continue
        if strb % WORD:
            continue
        addrs = [((l // dup) * strb) // WORD for l in range(LANES)]
        got = cost(addrs, width)
        if got is None:
            continue
        _, _, wf, _, _ = got
        ref = float(ncu) if ncu not in ("-", "") else model_wf
        if abs(wf - ref) < 1e-6:
            ok += 1
        else:
            bad += 1
            fails.append(
                f"{f[0]:<18} W={f[1]} шир={width} DUP={dup} шаг={strb}: "
                f"линтер {wf:g}, замер {ref:g}"
            )
    return (ok, bad), fails


# ------------------------------------------------------------------------------------------------
# 2. ЗАДНИК SRC: символьная развёртка индексного выражения по полосам
# ------------------------------------------------------------------------------------------------

LANE_SEEDS = ("threadIdx.x", "threadIdx.y", "threadIdx.z")
LANE_NAMES = ("lane", "lane_id", "laneid", "lane_idx", "tIdxLane")

C_TOKEN = re.compile(r"[A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)?")
PY_KEYWORDS = {"and", "or", "not", "if", "else", "True", "False"}

# ПЕРЕБОР по ещё не связанным шаблонным параметрам. Это НЕ факт о ядре, а перебор: линтер печатает,
# что вердикт получен перебором, и держится ли он на ВСЕХ значениях набора.
SWEEP = {
    "D": [64, 128, 256],
    "DV": [64, 128, 256],
    "DC": [64, 128, 256],
    "BK": [32, 64, 128],
    "BQ": [16, 32, 64],
    "BI": [32, 64],
    "BJ": [32, 64, 128],
    "EPT": [4, 8],
    "GF": [1, 2, 4],
    "NW": [4, 8],
    "KVB": [4, 8, 16],
    "WM": [1, 2, 4],
    "WN": [1, 2, 4, 8],
    "K": [16, 32, 64],
    "MB": [1, 2],
    "NB": [1, 2],
    "kW": [8],
    "KVFMT": [8],
    "SWZ": [0, 1],
    "KSP": [1],
    "REV": [0, 1],
    "CAUSAL": [0, 1],
    "MINB": [1],
    "PHSPLIT": [0],
    "DIAG": [0],
    "VEQK": [0, 1],
    "PAGED": [0, 1],
    "GFV": [1],
    "VQ": [0],
    "UU": [2, 4],
}
SWEEP_CAP = 48  # больше -- отчёт перестаёт быть читаемым, честнее отказать

# ФОРМА, НА КОТОРОЙ СНЯТ ЗАМЕР. Это не свойство ядра, а свойство ЗАМЕРА: bankaudit.py гонял декод
# при d=128, то есть EPT = d/32 = 4. При EPT=8 (d=256) тот же самый код КОНФЛИКТУЕТ вдвое, и линтер
# это честно показывает в общем отчёте. Сверять с якорем можно только ту форму, которую мерили.
MEASURED_BINDS = {
    "split_defer_kernel": {"EPT": [4]},
    "split_defer_mqa_kernel": {"EPT": [4]},
    "reduce_defer_out_kernel": {"EPT": [4]},
    "reduce_defer_kernel": {"EPT": [4]},
}
FORCE_BINDS = {}  # заполняется --bind или самопроверкой


def strip_comments(src):
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(" " * (j - i))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def c_to_py(expr):
    """Перевод C-выражения в python. Только арифметика/битовые/сравнения/тернарник."""
    e = expr
    # тернарник a ? b : c  ->  ((b) if (a) else (c))
    for _ in range(6):
        m = re.search(r"([^?:]+)\?([^?:]+):([^?:]+)", e)
        if not m:
            break
        e = (
            e[: m.start()]
            + f"(({m.group(2)}) if ({m.group(1)}) else ({m.group(3)}))"
            + e[m.end() :]
        )
    e = re.sub(r"\b(\d+)[uUlL]+\b", r"\1", e)
    e = re.sub(r"\(\s*(?:long|int|unsigned|size_t|long long|int64_t)\s*\)", "", e)
    e = e.replace("&&", " and ").replace("||", " or ").replace("!=", "__NE__")
    e = re.sub(r"!\s*", " not ", e)
    e = e.replace("__NE__", "!=")
    e = e.replace("/", "//")
    e = e.replace("////", "//")
    return e


class Unresolved(Exception):
    pass


LOOP_BOUND = {}


def build_env(body, consts):
    """имя -> [(позиция, выражение)]. ПОЗИЦИЯ ОБЯЗАТЕЛЬНА.

    Одно и то же имя в разных областях видимости ядра значит РАЗНОЕ (`c0` в прологе перекладки и
    `c0` в мейнлупе -- два разных выражения). Версия «первое определение выигрывает» молча брала
    чужое и выдавала уверенный НЕВЕРНЫЙ адрес. Берётся ближайшее определение ВЫШЕ обращения.
    """
    LOOP_BOUND[id(body)] = {}
    env = {k: [(-1, v)] for k, v in consts.items()}

    def add(name, expr, pos):
        env.setdefault(name, []).append((pos, expr.strip()))

    decl = re.compile(
        r"\b(?:const\s+|constexpr\s+|static\s+|volatile\s+)*"
        r"(?:int|unsigned|long|short|size_t|int64_t|uint32_t|int32_t|auto|float|bool)\s+"
        r"(\w+)\s*=\s*([^;,]+)(?=[;,])"
    )
    for m in decl.finditer(body):
        add(m.group(1), m.group(2), m.start())
    # объявления через запятую: const int a = x, b = y;   -- терминатор берём ЗАГЛЯДЫВАНИЕМ, иначе
    # первый же match съедает запятую, которая нужна следующему, и каждый второй символ теряется.
    for m in re.finditer(r",\s*(\w+)\s*=\s*([^;,]+)(?=[;,])", body):
        add(m.group(1), m.group(2), m.start())
    for m in re.finditer(
        r"for\s*\(\s*(?:int|unsigned)\s+(\w+)\s*=\s*([^;]+);\s*\1\s*<\s*([^;]+);", body
    ):
        add(m.group(1), m.group(2), m.start())  # счётчик цикла: варп-однороден
        LOOP_BOUND.setdefault(id(body), {}).setdefault(m.group(1), []).append(
            (m.start(), m.group(2).strip(), m.group(3).strip())
        )
    for m in re.finditer(r"for\s*\(\s*(?:int|unsigned)\s+(\w+)\s*=\s*([^;]+);", body):
        add(m.group(1), m.group(2), m.start())
    for k in env:
        env[k].sort()
    return env


def pick(env, name, at):
    """Ближайшее определение ВЫШЕ позиции at; если такого нет -- самое раннее."""
    lst = env.get(name)
    if not lst:
        return None
    best = None
    for pos, e in lst:
        if pos < at:
            best = e
    return best if best is not None else lst[0][1]


def resolve(expr, env, lane, at=1 << 30, depth=0, seen=None, unknown=None):
    """Численное значение выражения для данной полосы. Неизвестный символ -> Unresolved."""
    if depth > 24:
        raise Unresolved("слишком глубокая подстановка")
    seen = seen or set()
    py = c_to_py(expr)

    def sub(m):
        name = re.sub(r"\s+", "", m.group(0))
        if name in PY_KEYWORDS:
            return name
        if name in LANE_SEEDS[:1] or name in LANE_NAMES:
            return str(lane)
        if name in LANE_SEEDS[1:]:
            return "0"
        if name in seen:
            raise Unresolved(f"цикл в определении {name}")
        d = pick(env, name, at)
        if d is not None:
            v = resolve(d, env, lane, at, depth + 1, seen | {name}, unknown)
            return f"({v})"
        if unknown is not None:
            unknown.add(name)
        raise Unresolved(f"неизвестный символ {name}")

    py = C_TOKEN.sub(sub, py)
    # ПРОВЕРКА ПО ТОКЕНАМ, а не по набору букв: набор букв отвергал легальное `(0) if (1) else (2)`
    # (в нём есть i, f, e, l, s) и уводил в НЕ РАЗОБРАНО всё, где стоял тернарник -- то есть весь
    # разбор ролей варпов.
    leftover = [t for t in re.findall(r"[A-Za-z_]\w*", py) if t not in PY_KEYWORDS]
    if leftover:
        raise Unresolved(
            f"невычислимо: остались имена {', '.join(sorted(set(leftover))[:4])}"
        )
    if re.search(r"[^\d\s()+\-*/%<>=!&|^A-Za-z_]", py):
        raise Unresolved("невычислимое выражение (посторонний знак)")
    try:
        v = eval(py, {"__builtins__": {}}, {})
    except Exception as ex:
        raise Unresolved(f"ошибка вычисления: {type(ex).__name__}")
    if isinstance(v, bool):
        return 1 if v else 0
    if not isinstance(v, int):
        if float(v) != int(v):
            raise Unresolved("нецелое значение")
        v = int(v)
    return v


def symbols_of(expr, env, at=1 << 30, depth=0, seen=None, acc=None):
    """(зависит_ли_от_полосы, множество_НЕИЗВЕСТНЫХ_символов) -- транзитивно по среде.

    Второй элемент обязателен: если символ неизвестен, «не зависит от полосы» -- НЕ вывод, а
    незнание, и такое обращение обязано уйти в НЕ РАЗОБРАНО, а не в «чисто».
    """
    seen = seen or set()
    acc = acc if acc is not None else set()
    lane = False
    if depth > 24:
        return False, acc | {"<глубина>"}
    for m in C_TOKEN.finditer(c_to_py(expr)):
        name = re.sub(r"\s+", "", m.group(0))
        if name in PY_KEYWORDS:
            continue
        if name in LANE_SEEDS or name in LANE_NAMES:
            lane = True
            continue
        if name in seen:
            continue
        d = pick(env, name, at)
        if d is not None:
            l2, _ = symbols_of(d, env, at, depth + 1, seen | {name}, acc)
            lane = lane or l2
        else:
            acc.add(name)
    return lane, acc


def balanced(src, i, op="[", cl="]"):
    """Вернуть (содержимое, индекс_после) для скобки, открытой в позиции i."""
    d, j = 0, i
    while j < len(src):
        if src[j] == op:
            d += 1
        elif src[j] == cl:
            d -= 1
            if d == 0:
                return src[i + 1 : j], j + 1
        j += 1
    return None, len(src)


def find_functions(src):
    """[(имя, начало_тела, конец_тела, шаблонные_параметры)] для __global__/__device__ функций."""
    out = []
    for m in re.finditer(r"(__global__|__device__)([^;{]*?)\bvoid\b([^;{(]*?)\(", src):
        # имя -- ПОСЛЕДНИЙ идентификатор перед списком параметров. Иначе ловится __launch_bounds__,
        # и все находки уезжают в несуществующее ядро с этим именем.
        ids = re.findall(r"[A-Za-z_]\w*", m.group(3))
        if not ids:
            continue
        name = ids[-1]
        j = src.find("{", m.end())
        if j < 0:
            continue
        d, k = 0, j
        while k < len(src):
            if src[k] == "{":
                d += 1
            elif src[k] == "}":
                d -= 1
                if d == 0:
                    break
            k += 1
        # ШАБЛОННЫЕ ПАРАМЕТРЫ. Окно фиксированной длины НЕ ГОДИТСЯ: strip_comments сохраняет длину,
        # а над нашими ядрами лежат страницы комментариев, поэтому `template<` уезжает на тысячи
        # символов назад. Признак принадлежности -- между закрывающей '>' и объявлением нет ни ';',
        # ни '}', то есть ничего, что закрыло бы предыдущую сущность.
        tp = []
        for tm in reversed(list(re.finditer(r"template\s*<", src[: m.start()]))):
            body, after = balanced(src, tm.end() - 1, "<", ">")
            if body is None:
                continue
            between = src[after : m.start()]
            if ";" in between or "}" in between or "{" in between:
                break
            # СНАЧАЛА отбросить значение по умолчанию: `int DC = 64` даёт имя DC, а не «64».
            tp = [
                p.split("=")[0].strip().split()[-1].lstrip("*&")
                for p in re.split(r",(?![^<]*>)", body)
                if p.split("=")[0].strip()
            ]
            break
        out.append((name, j, k, tp))
    return out


def collect_consts(src):
    consts = {}
    for m in re.finditer(r"#define\s+(\w+)\s+([^\n\\]+)", src):
        consts.setdefault(m.group(1), m.group(2).strip())
    for m in re.finditer(
        r"(?:static\s+)?(?:constexpr|const)\s+"
        r"(?:int|unsigned|size_t|uint32_t|int32_t|long)\s+(\w+)\s*=\s*([^;,]+)",
        src,
    ):
        consts.setdefault(m.group(1), m.group(2).strip())
    for m in re.finditer(
        r"static\s+(?:int|unsigned)\s+const\s+(\w+)\s*=\s*([^;,]+)", src
    ):
        consts.setdefault(m.group(1), m.group(2).strip())
    return consts


# ОДНО ОБЪЯВЛЕНИЕ -- НЕСКОЛЬКО МАССИВОВ: `__shared__ float sB[NW], sA[NW*32*EPT], sM[NW];`.
# Версия, бравшая только первый декларатор, ТИХО не видела sA -- то есть самый горячий массив
# декода, -- и декод выходил «чистым» по причине СЛЕПОТЫ, а не чистоты.
SHARED_ARR = re.compile(
    r"__shared__\s+([A-Za-z_][\w:<>\s]*?)\s+((?:\w+\s*\[[^\];]*\]\s*,\s*)*\w+\s*\[[^\];]*\])\s*;"
)
SHARED_EXT = re.compile(
    r"extern\s+__shared__\s+([A-Za-z_][\w:<>,\s]*?)\s+(\w+)\s*\[\s*\]\s*;"
)
PTR_ALIAS = re.compile(r"\b([A-Za-z_][\w:<>]*)\s*\*\s*(\w+)\s*=\s*([^;]+);")
PTR_CAST = re.compile(
    r"\b([A-Za-z_][\w:<>]*)\s*\*\s*(\w+)\s*=\s*reinterpret_cast<\s*([\w:<>]+)\s*\*\s*>\s*\("
)


def elem_size(ty):
    t = re.sub(r"\b(const|volatile|__restrict__|restrict)\b", " ", ty).strip()
    t = re.sub(r"\s+", " ", t)
    return SIZEOF.get(t)


def collect_shared(body, fn_src_before):
    """{имя: (размер_элемента, тип, вид)} для разделяемых массивов и их псевдонимов-указателей."""
    shared = {}
    text = fn_src_before + body
    for m in SHARED_ARR.finditer(text):
        for dm in re.finditer(r"(\w+)\s*\[", m.group(2)):
            shared[dm.group(1)] = (
                elem_size(m.group(1)),
                m.group(1).strip(),
                "__shared__",
            )
    for m in SHARED_EXT.finditer(text):
        shared[m.group(2)] = (
            elem_size(m.group(1)),
            m.group(1).strip(),
            "extern __shared__",
        )
    # псевдонимы: T* p = <выражение, упоминающее уже известный разделяемый объект>;
    for _ in range(4):  # цепочки sQ -> sP -> sBuf
        grew = False
        for m in PTR_ALIAS.finditer(text):
            ty, name, rhs = m.group(1), m.group(2), m.group(3)
            if name in shared:
                continue
            base = None
            for s in shared:
                if re.search(r"\b" + re.escape(s) + r"\b", rhs):
                    base = s
                    break
            if base is None:
                continue
            cm = re.search(r"reinterpret_cast<\s*([\w:<>]+)\s*\*\s*>", rhs)
            ty2 = cm.group(1) if cm else ty
            shared[name] = (elem_size(ty2), ty2, f"псевдоним от {base}")
            grew = True
        if not grew:
            break
    return shared


VEC_BEFORE = re.compile(
    r"reinterpret_cast<\s*(?:const\s+)?([\w:]+)\s*\*\s*>\s*\(\s*&?\s*$"
)


def guards_for(body, pos):
    """Предикаты if(...), текстуально охватывающие позицию pos и упоминающие полосу."""
    out = []
    for m in re.finditer(r"\bif\s*\(", body):
        cond, after = balanced(body, m.end() - 1, "(", ")")
        if cond is None:
            continue
        rest = body[after:]
        k = 0
        while k < len(rest) and rest[k] in " \t\n":
            k += 1
        if k < len(rest) and rest[k] == "{":
            inner, end = balanced(body, after + k, "{", "}")
            span = (after + k, end)
        else:
            e = rest.find(";", k)
            span = (after, after + (e if e >= 0 else 0) + 1)
        if span[0] <= pos < span[1]:
            out.append(cond)
    return out


_INST_INDEX = {}


def inst_args_index(tree_src, names):
    """ОДИН проход по дереву: имя_шаблона -> список строк аргументов инстанцирования.

    Проход обязан быть один. Отдельный re.finditer на КАЖДУЮ функцию давал по 2 МБ сканирования на
    имя, и линтер вставал на kernel_backward.h -- инструмент, который не доходит до конца, это не
    «медленный инструмент», а ОТСУТСТВУЮЩИЙ ответ.
    """
    key = (id(tree_src), frozenset(names))
    if key in _INST_INDEX:
        return _INST_INDEX[key]
    out = {n: [] for n in names}
    if names:
        rx = re.compile(
            r"\b(" + "|".join(re.escape(n) for n in sorted(names)) + r")\s*<"
        )
        for m in rx.finditer(tree_src):
            win = tree_src[m.end() - 1 : m.end() + 500]
            args, _ = balanced(win, 0, "<", ">")
            if args is not None:
                out[m.group(1)].append(args)
    _INST_INDEX[key] = out
    return out


def split_args(args):
    parts, d, cur = [], 0, ""
    for ch in args:
        if ch == "<":
            d += 1
        elif ch == ">":
            d -= 1
        if ch == "," and d == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def harvest_instantiations(index, fname, tparams):
    """Числовые значения шаблонных параметров с МЕСТ ИНСТАНЦИИРОВАНИЯ: kern<32, BK, D, 2, 4, ...>.

    Единственный способ узнать WM/WN/BQ, не запуская компилятор. Нечисловые позиции остаются
    несвязанными -- их берёт на себя ПЕРЕБОР, и он помечается в отчёте как перебор.
    """
    binds = {}
    if not tparams:
        return binds
    for args in index.get(fname, []):
        for i, p in enumerate(split_args(args)):
            if i >= len(tparams):
                break
            p = p.strip()
            v = None
            if re.fullmatch(r"\d+", p):
                v = int(p)
            elif p == "true":
                v = 1
            elif p == "false":
                v = 0
            if v is None:
                continue
            binds.setdefault(tparams[i], set()).add(v)
    return {k: sorted(v) for k, v in binds.items()}


def eval_access(idx, esz, width, env, mask, unbound_vals, at):
    """Адреса полос при данном наборе значений несвязанных символов. Unresolved -> исключение."""
    e2 = dict(env)
    e2.update({k: [(-1, str(v))] for k, v in unbound_vals.items()})
    addrs = []
    for lane in range(LANES):
        on = True
        for g in mask:
            if not resolve(g, e2, lane, at):
                on = False
                break
        addrs.append((resolve(idx, e2, lane, at) * esz) // WORD if on else None)
    return addrs


def sweep_sets(names, binds):
    """Наборы значений для несвязанных символов. Пусто -> перебор невозможен."""
    dims = []
    for n in sorted(names):
        if n in binds:
            dims.append((n, binds[n]))
        elif n in SWEEP:
            dims.append((n, SWEEP[n]))
        else:
            return None
    total = 1
    for _, v in dims:
        total *= len(v)
    if total == 0 or total > SWEEP_CAP:
        return None
    out = [{}]
    for n, vals in dims:
        out = [dict(o, **{n: v}) for o in out for v in vals]
    return out


def scan_source(path, consts_global, tree_src=""):
    hits, unparsed, skipped = [], [], []
    try:
        raw = open(path, encoding="utf-8", errors="replace").read()
    except OSError as e:
        return hits, [(path, 0, "файл", str(e))], skipped
    src = strip_comments(raw)
    consts = dict(consts_global)
    consts.update(collect_consts(src))
    fns = find_functions(src)
    if not fns:
        return hits, unparsed, skipped
    decl_spans = [(m.start(), m.end()) for m in SHARED_ARR.finditer(src)] + [
        (m.start(), m.end()) for m in SHARED_EXT.finditer(src)
    ]
    index = inst_args_index(tree_src or src, {f[0] for f in fns if f[3]})

    for fname, b0, b1, tparams in fns:
        body = src[b0:b1]
        shared = collect_shared(body, src[:b0])
        if not shared:
            continue
        env = build_env(body, consts)
        for tp in tparams:  # шаблонные параметры символьны
            env.pop(tp, None)
        binds = harvest_instantiations(index, fname, tparams)
        binds.update(FORCE_BINDS.get(fname, {}))
        for aname, (esz, ty, kind) in sorted(shared.items()):
            for m in re.finditer(r"\b" + re.escape(aname) + r"\s*\[", body):
                apos = b0 + m.start()
                if any(a <= apos < b for a, b in decl_spans):
                    continue  # это ОБЪЯВЛЕНИЕ, а не обращение
                idx, after = balanced(body, m.end() - 1, "[", "]")
                if idx is None:
                    continue
                line = src[:apos].count("\n") + 1
                where = f"{fname}: {aname}[{idx.strip()[:70]}]"
                if esz is None:
                    unparsed.append(
                        (path, line, where, f"неизвестный размер элемента ({ty})")
                    )
                    continue
                # ШИРИНА НА ПОЛОСУ. Три источника, по убыванию надёжности:
                #   1) *reinterpret_cast<V*>(&arr[idx])           -- прямо здесь;
                #   2) T* p = &arr[idx]; ... reinterpret_cast<V*>(p)  -- через псевдоним;
                #   3) размер элемента.
                # ОГОВОРКА, которую линтер обязан держать вслух: компилятор ВПРАВЕ слить соседние
                # узкие обращения в широкое (две STS.64 подряд -> одна STS.128), и тогда ширина
                # больше, ПОЛ выше, а «возвратимое» падает. Проверять по SASS.
                vm = VEC_BEFORE.search(body[max(0, m.start() - 90) : m.start()])
                width = SIZEOF.get(vm.group(1), esz) if vm else esz
                width_src = "прямой вектор-каст" if vm else "размер элемента"
                if not vm:
                    am = re.search(
                        r"([A-Za-z_][\w:<>]*)\s*\*\s*(\w+)\s*=\s*&\s*$",
                        body[max(0, m.start() - 60) : m.start()],
                    )
                    if am:
                        nxt = body[after : after + 300]
                        cm = re.search(
                            r"reinterpret_cast<\s*(?:const\s+)?([\w:]+)\s*\*\s*>\s*\(\s*"
                            + re.escape(am.group(2)),
                            nxt,
                        )
                        if cm and cm.group(1) in SIZEOF:
                            width = SIZEOF[cm.group(1)]
                            width_src = f"вектор-каст через псевдоним {am.group(2)} ({cm.group(1)})"

                mask = []
                unk = set()
                lane_dep, u = symbols_of(idx, env, m.start())
                unk |= u
                for g in guards_for(body, m.start()):
                    gl, gu = symbols_of(g, env, m.start())
                    if gl:
                        mask.append(g)
                        unk |= gu
                if not lane_dep and not unk:
                    skipped.append(
                        (
                            path,
                            line,
                            where,
                            "индекс НЕ зависит от полосы -> все полосы на одном адресе, "
                            "многоадресность бесплатна",
                        )
                    )
                    continue

                sets = sweep_sets(unk, binds) if unk else [{}]
                if sets is None:
                    tp_hit = sorted(unk & set(tparams))
                    why = f"не связаны символы {', '.join(sorted(unk))}" + (
                        f" (шаблонные: {', '.join(tp_hit)})" if tp_hit else ""
                    )
                    why += (
                        f". УСЛОВИЕ БЕСКОНФЛИКТНОСТИ для обхода ПО СТОЛБЦУ: шаг строки в "
                        f"СЛОВАХ (выражение * {esz} / 4) обязан быть НЕЧЁТНЫМ; при чётном "
                        f"шаге кратность = min(32, gcd(шаг,32) * полос / 32)"
                    )
                    unparsed.append((path, line, where, why))
                    continue

                # КОАЛЕСЦЕНЦИЯ ВНУТРЕННЕГО ЦИКЛА -- считается ВНУТРИ перебора, потому что число
                # итераций само бывает шаблонным (EPT = d/32). `for (e=0;e<EPT;++e) arr[..+lane*EPT+e]`
                # это НЕ EPT узких обращений: адреса соседних итераций смежны, цикл развёрнут, и
                # железо видит ОДНО широкое. Без этого ПОЛ занижается, и минимальный ТРАФИК
                # объявляется конфликтом -- ровно ошибка, из-за которой чистый декод (замер 1.2 %)
                # выглядел бы грязным.
                bounds = LOOP_BOUND.get(id(body), {})
                results, fail = [], None
                for s in sets:
                    ov = {k: [(-1, str(v))] for k, v in s.items()}
                    w = width
                    wsrc = width_src
                    for v, defs in bounds.items():
                        cand = [d for d in defs if d[0] < m.start()]
                        if not cand or not re.search(r"\b" + re.escape(v) + r"\b", idx):
                            continue
                        _, init_e, bound_e = cand[-1]
                        e2 = dict(env)
                        e2.update(ov)
                        try:
                            i0 = resolve(init_e, e2, 0, m.start())
                            nn = resolve(bound_e, e2, 0, m.start())
                            base = resolve(idx, e2, 0, m.start())
                            e3 = dict(e2)
                            e3[v] = [(-1, str(i0 + 1))]
                            nxt = resolve(idx, e3, 0, m.start())
                        except Unresolved:
                            continue
                        trip = nn - i0
                        if (
                            nxt - base == 1
                            and 2 <= trip <= 8
                            and min(16, trip * esz) > w
                        ):
                            w = min(16, trip * esz)
                            wsrc = (
                                f"склейка развёрнутого цикла по {v} "
                                f"({trip} смежных элемента по {esz} Б)"
                            )
                    try:
                        a = eval_access(idx, esz, w, env, mask, s, m.start())
                    except Unresolved as ex:
                        fail = str(ex)
                        break
                    c = cost(a, w)
                    if c is not None:
                        results.append((s, a, c, w, wsrc))
                if fail or not results:
                    unparsed.append(
                        (path, line, where, fail or "ни одна полоса не активна")
                    )
                    continue

                worst = max(results, key=lambda r: r[2][3])
                s, addrs, (degree, floor, wf, rec, nact), width, width_src = worst
                if not lane_dep and rec <= 0.001:
                    skipped.append((path, line, where, "индекс не зависит от полосы"))
                    continue
                deltas = [
                    addrs[i + 1] - addrs[i]
                    for i in range(LANES - 1)
                    if addrs[i] is not None and addrs[i + 1] is not None
                ]
                # КАК ХОДЯТ ПОЛОСЫ. Подозрение без этого -- не приговор: при плотном обходе по
                # строке конфликта из шага НЕТ по построению. «Плотно» значит шаг между соседними
                # полосами РОВНО в ширину обращения; порог «не больше четырёх слов» врал -- он
                # называл строкой обход с шагом 16 Б при ширине 2 Б.
                if deltas and len(set(deltas)) == 1:
                    dl = deltas[0]
                    if dl * WORD == width:
                        walk = f"ПО СТРОКЕ ПЛОТНО (шаг полосы {dl} сл. = ширина): из шага конфликта НЕТ"
                    else:
                        walk = (
                            f"РЕГУЛЯРНО, шаг между полосами {dl} сл. = {dl * WORD} Б при ширине "
                            f"{width} Б -- {'разрежённый обход' if dl * WORD > width else 'наложение'}"
                        )
                elif deltas:
                    walk = f"ВРАЗБРОС: шаг между полосами от {min(deltas)} до {max(deltas)} слов"
                else:
                    walk = "одна активная полоса"
                allbad = all(r[2][3] > 0.001 for r in results)
                note = ""
                if len(sets) > 1 or (sets and sets[0]):
                    note = (
                        "значения "
                        + ", ".join(f"{k}={v}" for k, v in sorted(s.items()))
                        + (
                            f"; ПЕРЕБОР {len(sets)} наборов, конфликт "
                            + ("на ВСЕХ" if allbad else "НЕ на всех")
                        )
                    )
                if rec > 0.001:
                    hits.append(
                        dict(
                            path=path,
                            line=line,
                            kernel=fname,
                            what=where,
                            degree=degree,
                            floor=floor,
                            wf=wf,
                            rec=rec,
                            nact=nact,
                            width=width,
                            walk=walk,
                            note=note,
                            width_src=width_src,
                            banks=sorted({a % BANKS for a in addrs if a is not None}),
                            backend="SRC",
                        )
                    )
                else:
                    skipped.append(
                        (
                            path,
                            line,
                            where,
                            f"вайвфронтов {wf:g} = ПОЛ {floor:g} (трафик/ширина), "
                            f"конфликтность {degree} НЕ возвратима; {walk}",
                        )
                    )
    return hits, unparsed, skipped


# ------------------------------------------------------------------------------------------------
# 3. ЗАДНИК CUTLASS: настоящие функторы раскладки, скомпилированные на хосте
# ------------------------------------------------------------------------------------------------

SWIZZLED = ("TensorOpMultiplicand", "Crosswise", "Congruous")

# КАКОЕ ЯДРО КАКОЙ СЛУЧАЙ ИСПОЛЬЗУЕТ. Это ручная опись, а не вывод: она и есть граница задника.
CUTLASS_USERS = {
    "B2bGemm.accumToSmem.volta.f32accum": [
        (
            "attention_kernel_batched_impl (forward)",
            "fa2_src/fmha_kernel/kernel_forward.h: MM0::AccumulatorSharedStorage si "
            "<- B2bGemm::accumToSmem (запись P между первым и вторым умножением)",
        ),
        (
            "attention_kernel_backward_batched_impl (backward)",
            "fa2_src/fmha_kernel/kernel_backward.h: MatmulQK::AccumulatorSharedStorage "
            "attn_shared_storage <- тот же B2bGemm::accumToSmem",
        ),
    ],
    "B2bGemm.accumToSmem.volta.f16accum": [
        (
            "attention_kernel_* (обе стороны, ветка накопителя half)",
            "та же функция, ветка else",
        ),
    ],
}
CUTLASS_NOTOUCH = (
    "VoltaCongruous16.naive_col",
    "VoltaCongruous16.naive_row8",
    "VoltaCrosswise16x32.naive_col",
    "VoltaCrosswise16x32.naive_row",
)
CUTLASS_CONTROL = ("RowMajor32.col_walk", "RowMajor32.row_walk", "RowMajor64.col_walk")


def run_cutlass_probe(builddir=None):
    """Собрать и запустить хостовой зонд. Возвращает (случаи, беды)."""
    src = os.path.join(HERE, "smem_lint_cutlass.cpp")
    if not os.path.exists(src):
        return {}, [f"нет {src}"]
    builddir = builddir or os.path.join(TEMPO, "build", "lint")
    os.makedirs(builddir, exist_ok=True)
    exe = os.path.join(builddir, "smem_lint_cutlass")
    cmd = [
        "g++",
        "-std=c++17",
        "-O0",
        f"-I{CUTLASS_INC}",
        f"-I{CUDA_INC}",
        src,
        "-o",
        exe,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {}, [
            f"g++ не собрал зонд: {p.stderr.strip().splitlines()[-1][:160] if p.stderr.strip() else '?'}"
        ]
    r = subprocess.run([exe], capture_output=True, text=True)
    if r.returncode != 0:
        return {}, [f"зонд упал (rc={r.returncode})"]
    cases = {}
    for line in r.stdout.splitlines():
        f = line.split()
        if len(f) != 4 + LANES or f[0] != "CASE":
            continue
        name, esz, width = f[1], int(f[2]), int(f[3])
        elems = [int(x) for x in f[4:]]
        addrs = [None if e < 0 else (e * esz) // WORD for e in elems]
        cases[name] = (esz, width, addrs)
    return cases, []


def cutlass_findings(cases):
    hits, notouch, control = [], [], []
    for name, (esz, width, addrs) in cases.items():
        got = cost(addrs, width)
        if got is None:
            continue
        degree, floor, wf, rec, nact = got
        row = dict(
            name=name,
            degree=degree,
            floor=floor,
            wf=wf,
            rec=rec,
            width=width,
            banks=sorted({a % BANKS for a in addrs if a is not None}),
            backend="CUTLASS",
        )
        if name in CUTLASS_NOTOUCH:
            notouch.append(row)
        elif name in CUTLASS_CONTROL:
            control.append(row)
        elif rec > 0.001:
            row["users"] = CUTLASS_USERS.get(name, [])
            hits.append(row)
        else:
            control.append(row)
    return hits, notouch, control


# ------------------------------------------------------------------------------------------------
# 4. ОТЧЁТ
# ------------------------------------------------------------------------------------------------

SKIP_DIRS = {".git", "build", "_build", "__pycache__", ".mypy_cache", "cutlass"}
# ОТГРУЖЕННОЕ против ЛЕСОВ. tools/ полон теневых копий ядер и микростендов; их находки верны, но
# к боевому коду отношения не имеют и топят отчёт. По умолчанию отчёт про отгруженное, --all -- всё.
SHIPPED_DIRS = ("fa2_sm70/csrc", "fa2_src/fmha_kernel")
EXT = (".cu", ".cuh", ".h", ".hpp")


def read_tree(root):
    """Весь текст дерева одной строкой -- нужен, чтобы найти МЕСТА ИНСТАНЦИИРОВАНИЯ шаблонов."""
    buf = []
    for p in walk_sources(root):
        try:
            buf.append(
                strip_comments(open(p, encoding="utf-8", errors="replace").read())
            )
        except OSError:
            pass
    return "\n".join(buf)


def walk_sources(root, shipped_only=False):
    for dp, dn, files in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in sorted(files):
            if not f.endswith(EXT):
                continue
            full = os.path.join(dp, f)
            if shipped_only and not any(d in full for d in SHIPPED_DIRS):
                continue
            yield full


def report(root, kernel_filter=None, quiet_skipped=True, shipped_only=True):
    tree = read_tree(root)
    hits, unparsed, skipped, nfiles = [], [], [], 0
    for p in walk_sources(root, shipped_only):
        nfiles += 1
        h, u, s = scan_source(p, {}, tree)
        hits += h
        unparsed += u
        skipped += s
    if kernel_filter:
        hits = [h for h in hits if kernel_filter in h["kernel"]]

    cases, probe_err = run_cutlass_probe()
    chits, notouch, control = cutlass_findings(cases)

    print("=" * 108)
    print(
        "ПОДОЗРЕНИЯ -- ВОЗВРАТИМЫЕ вайвфронты (снимаются дополнением раскладки, ответ не меняется)"
    )
    print("=" * 108)
    if not hits and not chits:
        print("(нет)")
    for h in sorted(chits, key=lambda x: -x["rec"]):
        print(f"[CUTLASS] {h['name']}")
        print(
            f"    конфликтность {h['degree']}  ПОЛ {h['floor']:g}  вайвфронтов {h['wf']:g}"
            f"  -> ВОЗВРАТИМО {h['rec']:g} ({h['wf'] / h['floor']:.2f}x лишнего)"
        )
        print(
            f"    полоса пишет {h['width']} Б; задействовано банков {len(h['banks'])} из 32: "
            f"{h['banks'][:8]}{' ...' if len(h['banks']) > 8 else ''}"
        )
        for k, w in h.get("users", []):
            print(f"    ЯДРО: {k}\n          {w}")
    for h in sorted(hits, key=lambda x: -x["rec"]):
        print(f"[SRC] {os.path.relpath(h['path'], root)}:{h['line']}  {h['what']}")
        print(
            f"    конфликтность {h['degree']}  ПОЛ {h['floor']:g}  вайвфронтов {h['wf']:g}"
            f"  -> ВОЗВРАТИМО {h['rec']:g} ({h['wf'] / h['floor']:.2f}x лишнего)"
        )
        print(
            f"    активных полос {h['nact']}, ширина {h['width']} Б, обход: {h['walk']}"
        )
        if h.get("note"):
            print(f"    {h['note']}")
        print(
            f"    банков задействовано {len(h['banks'])} из 32: {h['banks'][:10]}"
            f"{' ...' if len(h['banks']) > 10 else ''}"
        )

    print("\n" + "=" * 108)
    print(
        "НЕ ТРОГАТЬ -- свиззл cutlass бесконфликтен ПО ПОСТРОЕНИЮ; ручное дополнение его СЛОМАЕТ"
    )
    print("=" * 108)
    for h in notouch:
        print(
            f"  {h['name']:<36} конфликтность {h['degree']}  ПОЛ {h['floor']:g}  "
            f"вайвфронтов {h['wf']:g}  возвратимо {h['rec']:g}"
        )
    print(
        "  Вердикт вынесен по ТИПУ раскладки (VoltaTensorOpMultiplicand*), а показанный обход --"
    )
    print(
        "  наивный по строке/столбцу, НЕ тот, которым ходит MmaVoltaTensorOpMultiplicandTileIterator."
    )

    print("\n" + "=" * 108)
    print("КОНТРОЛЬ -- то же измерение на телах, где ответ известен заранее")
    print("=" * 108)
    for h in control:
        print(
            f"  {h['name']:<36} конфликтность {h['degree']}  ПОЛ {h['floor']:g}  "
            f"вайвфронтов {h['wf']:g}  возвратимо {h['rec']:g}"
        )
    print(
        "  RowMajor*.col_walk -- голая раскладка без свиззла: конфликт обязан быть большим."
    )
    print("  Если он ЗДЕСЬ равен 1, зонд вырожден и весь отчёт недействителен.")

    print("\n" + "=" * 108)
    print("НЕ РАЗОБРАНО -- линтер НЕ утверждает, что здесь чисто")
    print("=" * 108)
    for e in probe_err:
        print(f"  [CUTLASS] {e}")
    print(
        f"  [CUTLASS] задник разбирает ТОЛЬКО перечисленные в CUTLASS_USERS случаи "
        f"({len(CUTLASS_USERS)} шт.). Остальные члены SharedStorage форварда/бэкварда "
        f"(mm0/mm1 Mma::SharedStorage, epilogue, BiasLoader::SmemTile, ScalingCoefs) НЕ разобраны."
    )
    seen = set()
    for path, line, what, why in unparsed:
        key = (os.path.basename(path), what.split(":")[0], why[:40])
        if key in seen:
            continue
        seen.add(key)
        print(f"  [SRC] {os.path.relpath(path, root)}:{line}  {what}\n        -- {why}")
    print(
        f"\n  Всего неразобранных обращений: {len(unparsed)} (показано {len(seen)} по одному на вид)."
    )

    if not quiet_skipped:
        print("\n" + "=" * 108)
        print("РАЗОБРАНО И ЧИСТО (для проверки, что линтер вообще смотрел)")
        print("=" * 108)
        for path, line, what, why in skipped:
            print(f"  {os.path.relpath(path, root)}:{line}  {what}\n        -- {why}")

    print("\n" + "-" * 108)
    print(
        "ОБЛАСТЬ: "
        + (
            "ТОЛЬКО ОТГРУЖЕННОЕ (" + ", ".join(SHIPPED_DIRS) + "); леса из tools/ НЕ "
            "просмотрены -- запустить с --all"
            if shipped_only
            else "ВСЁ ДЕРЕВО"
        )
    )
    print(
        f"ИТОГ: файлов {nfiles}; подозрений SRC {len(hits)} + CUTLASS {len(chits)}; "
        f"«не трогать» {len(notouch)}; разобрано-и-чисто {len(skipped)}; "
        f"НЕ РАЗОБРАНО {len(unparsed) + len(probe_err)}."
    )
    print(
        "ПУСТОЙ СПИСОК ПОДОЗРЕНИЙ ПРИ НЕПУСТОМ СПИСКЕ НЕРАЗОБРАННОГО НЕ ОЗНАЧАЕТ «КОНФЛИКТОВ НЕТ»."
    )
    return hits, chits, unparsed, skipped


# ------------------------------------------------------------------------------------------------
# 5. САМОПРОВЕРКА
# ------------------------------------------------------------------------------------------------

# ЯКОРЬ №2 -- ЗАМЕР ncu ПО ОТГРУЖЕННЫМ ЯДРАМ (docs/SM70_KERNEL_PLAYBOOK.md §53):
# конфликты/вайвфронты = 20.8 % forward d=128, 19.4 % volta_fwd_ws, 34.6 % backward, 1.2 % декод.
# Линтер обязан УКАЗАТЬ на первые три и НЕ УКАЗАТЬ на декод. Доля -- не то, что считает линтер
# (он даёт кратность на ОБРАЩЕНИЕ, а не долю трафика), поэтому сверяется НАПРАВЛЕНИЕ, а не число.
ANCHOR = [
    ("attention_kernel_batched_impl (forward)", True, 20.8),
    ("volta_fwd_ws", True, 19.4),
    ("attention_kernel_backward", True, 34.6),
    ("split_defer_mqa (декод)", False, 1.2),
]


def selftest(root):
    print("#" * 108)
    print(
        "# САМОПРОВЕРКА 1: ЗАКОН против стенда (data/mio_wavefronts.txt, из них 17 точек ncu)"
    )
    print("#" * 108)
    res, fails = selftest_law()
    if res is None:
        print("  НЕ ПРОВЕРЕНО: " + "; ".join(fails))
        law_ok = False
    else:
        ok, bad = res
        print(f"  сошлось {ok}, разошлось {bad}")
        for f in fails[:12]:
            print(f"    РАСХОЖДЕНИЕ: {f}")
        law_ok = bad == 0 and ok > 50

    print()
    print("#" * 108)
    print(
        "# САМОПРОВЕРКА 2: ЯКОРЬ по отгруженным ядрам (ncu: 20.8 / 19.4 / 34.6 / 1.2 %)"
    )
    print("#" * 108)
    tree = read_tree(root)
    FORCE_BINDS.clear()
    FORCE_BINDS.update(MEASURED_BINDS)
    print(
        "  ФОРМА ЗАМЕРА: "
        + "; ".join(f"{k} {v}" for k, v in sorted(MEASURED_BINDS.items()))
    )
    hits, unparsed, skipped = [], [], []
    for p in walk_sources(root):
        h, u, s = scan_source(p, {}, tree)
        hits += h
        unparsed += u
        skipped += s
    cases, probe_err = run_cutlass_probe()
    chits, _, control = cutlass_findings(cases)

    ctl_bad = [c for c in control if c["name"].endswith("col_walk") and c["degree"] < 4]
    if ctl_bad:
        print(
            "  КОНТРОЛЬ ПРОВАЛЕН: голая RowMajor даёт конфликтность < 4 -- зонд вырожден."
        )

    pointed = {}
    pointed["attention_kernel_batched_impl (forward)"] = [
        f"[CUTLASS] {h['name']} x{h['wf'] / h['floor']:.2f}"
        for h in chits
        for k, _ in h.get("users", [])
        if "forward" in k
    ]
    pointed["attention_kernel_backward"] = [
        f"[CUTLASS] {h['name']} x{h['wf'] / h['floor']:.2f}"
        for h in chits
        for k, _ in h.get("users", [])
        if "backward" in k
    ]
    pointed["volta_fwd_ws"] = [
        f"[SRC] {os.path.basename(h['path'])}:{h['line']} {h['what']} x{h['wf'] / h['floor']:.2f}"
        for h in hits
        if "volta_fwd_ws" in h["kernel"]
    ]
    pointed["split_defer_mqa (декод)"] = [
        f"[SRC] {os.path.basename(h['path'])}:{h['line']} {h['what']} x{h['wf'] / h['floor']:.2f}"
        for h in hits
        if "split_defer" in h["kernel"] or "reduce_defer" in h["kernel"]
    ]

    anchor_ok = True
    for name, must, meas in ANCHOR:
        got = pointed.get(name, [])
        good = bool(got) == must
        anchor_ok &= good
        verdict = "СОШЛОСЬ" if good else "РАСХОЖДЕНИЕ"
        print(
            f"  {verdict:<12} {name:<42} замер {meas:>5.1f} %  "
            f"ожидалось {'указать' if must else 'НЕ указывать'}, указаний {len(got)}"
        )
        for g in got[:4]:
            print(f"                 {g}")

    print()
    print(
        f"  НЕ РАЗОБРАНО при этом прогоне: {len(unparsed)} обращений + "
        f"{len(probe_err)} бед зонда. Это НЕ 'чисто'."
    )
    print()
    print("#" * 108)
    print(
        f"# ИТОГ САМОПРОВЕРКИ: закон {'ОК' if law_ok else 'НЕ ОК'}, "
        f"якорь {'ОК' if anchor_ok else 'НЕ ОК'}"
    )
    print("#" * 108)
    return 0 if (law_ok and anchor_ok) else 1


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    root = args[-1] if args else DEFAULT_ROOT
    if "--selftest" in argv:
        return selftest(root)
    kf = None
    if "--kernel" in argv:
        kf = argv[argv.index("--kernel") + 1]
        root = args[-1] if len(args) > 1 else DEFAULT_ROOT
    report(
        root,
        kernel_filter=kf,
        quiet_skipped="--verbose" not in argv,
        shipped_only="--all" not in argv,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
