# -*- coding: utf-8 -*-
"""ЕДИНЫЙ ВХОД В СТЕК ИЗМЕРИТЕЛЬНЫХ ИНСТРУМЕНТОВ sm_70 (tempo).

ЗАЧЕМ ОН ЕСТЬ. Шесть инструментов -- шесть разных запусков, шесть разных наборов переменных
окружения и (замерено, не предположено) ДВА РАЗНЫХ ИНТЕРПРЕТАТОРА. Забыть один из них не значит
"получить меньше данных": это значит получить ДРУГОЙ ОТВЕТ или немой отказ. Поэтому вход один.

ЧТО ЭТОТ ВХОД ДЕЛАЕТ СВЕРХ ПРОБРОСА АРГУМЕНТОВ (иначе он был бы не нужен):

  1. ВЫБИРАЕТ ИНТЕРПРЕТАТОР ПОД ИНСТРУМЕНТ И ОБЪЯСНЯЕТ ВЫБОР. Замерено на этой машине:
       * tools/residency.py НЕ РАЗБИРАЕТСЯ Python 3.11 (`SyntaxError: unterminated string literal`,
         строка 295): внутри f-строки стоит перенос строки -- это PEP 701, только Python >= 3.12.
         Системный python3 -- 3.12.10, python окружения vllm -- 3.11.11. Запуск не тем даёт отказ,
         похожий на "сломанный файл", а не на "не тот интерпретатор".
       * tools/timeit.py в БОЕВОМ режиме импортирует torch (лениво, строка 559) -- ему нужен именно
         python окружения vllm; --selftest/--precheck обходятся без torch и идут на любом.
     Правило зашито в таблицу TOOLS и печатается строкой "интерпретатор: ... потому что ...".

  2. ОТКАЗЫВАЕТ ДО ЗАПУСКА, а не посреди него: нет годного интерпретатора, нет файла инструмента,
     нет TEMPO_SUDO_PASS для ncu -- сообщение и код возврата 2, тело не стартует.

  3. ПОДСТАВЛЯЕТ ДОЛГОЖИВУЩИЕ ПУТИ. Якорь ncu.py по умолчанию ссылался на bankaudit.py в
     СЕССИОННОМ каталоге ./.cache -- он будет вычищен, и якорь перестанет
     воспроизводиться без единого признака ошибки. Копия лежит в tools/bankaudit.py, CLI
     выставляет TEMPO_BANKAUDIT на неё, если переменная не задана извне.

  4. ЗНАЕТ ПОРЯДОК. `tempo_cli.py map` печатает таблицу "какой вопрос -- каким инструментом" и
     порядок применения (сперва фазы, потом счёт; сперва корректность, потом время; сперва цена
     единицы, потом оптимизация по ней). Тот же текст -- в README.md.

  5. ГОВОРИТ, ЧЕГО НЕ ПРОВЕРИЛ. `doctor` печатает состояние среды и раздел "НЕ УСТАНОВЛЕНО".

ЧЕГО ЭТОТ ВХОД НЕ ДЕЛАЕТ: он НЕ проверяет ответы инструментов и НЕ добавляет ни одного замера.
Все якоря -- внутри самих инструментов. CLI -- это диспетчер и памятка, не измеритель.

ЗАПУСК:
    python3 tools/tempo_cli.py                 -- список подкоманд
    python3 tools/tempo_cli.py map             -- таблица "вопрос -> инструмент"
    python3 tools/tempo_cli.py doctor          -- состояние среды (GPU не трогается)
    python3 tools/tempo_cli.py selftest        -- все самопроверки, которым НЕ нужна карта
    python3 tools/tempo_cli.py <sub> --help    -- справка самого инструмента
    python3 tools/tempo_cli.py <sub> [args...] -- проброс аргументов как есть
    python3 tools/tempo_cli.py --dry <sub> ... -- показать команду и НЕ запускать
"""

import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

PY_SYS = shutil.which("python3") or sys.executable
PY_VLLM = "/opt/conda/miniconda3/envs/vllm/bin/python"
REPO = "../VLLM_fa2/solutions/fa2_sm70_cutlass_grade"
CUDA_HOME = "/opt/conda/miniconda3/envs/cuda128"
NCU_GOOD = (
    "/opt/conda/miniconda3/pkgs/nsight-compute-2024.1.1.4-0/nsight-compute/2024.1.1/ncu"
)


# ------------------------------------------------------------------------------------------------
# ТАБЛИЦА ИНСТРУМЕНТОВ.
#   min_py     -- минимальная версия интерпретатора (замерена, а не предположена)
#   needs_torch-- предикат по аргументам: нужен ли torch (=> интерпретатор окружения vllm)
#   gpu        -- нужна ли карта для БОЕВОГО режима
#   sudo       -- нужен ли sudo
# ------------------------------------------------------------------------------------------------
def _timeit_needs_torch(argv):
    dry = {"--selftest", "--precheck", "--calibrate-idle", "--help", "-h"}
    return not (set(argv) & dry)


TOOLS = {
    "phases": dict(
        script="phaseprof.py",
        min_py=(3, 8),
        needs_torch=lambda a: False,
        gpu="только режим time (сборка N+1 фальсификаторов + замер)",
        sudo=False,
        what="ДОЛЯ ФАЗЫ во времени ядра = 1 - t(фаза снята)/t(база); плюс статический счёт "
        "команд SASS по фазам; плюс разложение невязки на ПЕРЕКРЫТИЕ и НЕНАЗВАННОЕ.",
        hint=[
            "python3 tools/tempo_cli.py phases replay --times data/anchor_fwd_ws_phases.json",
            "python3 tools/tempo_cli.py phases static --spec tools/demo_phase.spec.json",
            "python3 tools/tempo_cli.py phases time   --spec <spec>.json --pairs --save t.json",
        ],
        anchor="replay по data/anchor_fwd_ws_phases.json обязан дать 35.1/19.7/15.3/5.7/2.0 %",
    ),
    "ncu": dict(
        script="ncu.py",
        min_py=(3, 8),
        needs_torch=lambda a: False,
        gpu="да (кроме --which и --selftest-synth без --deep)",
        sudo=True,
        what="Цена разделяемой памяти в ВАЙВФРОНТАХ и ДОЛЯ КОНФЛИКТОВ банков, двумя независимыми "
        "маршрутами (итог на запуск / на команду), с посточечной привязкой.",
        hint=[
            "python3 tools/tempo_cli.py ncu --which",
            "python3 tools/tempo_cli.py ncu --selftest              # якорь bwd 34.6 %",
            "python3 tools/tempo_cli.py ncu --kernel 'attention_kernel_backward' "
            "-- $PY tools/bankaudit.py --run bwd",
        ],
        anchor="bwd attention_kernel_backward: 4925440 вайвфронтов / 1703936 конфликтов = 34.6 %",
    ),
    "lint": dict(
        script="smem_lint.py",
        min_py=(3, 8),
        needs_torch=lambda a: False,
        gpu="нет (нужен g++ для cutlass-задника)",
        sudo=False,
        what="КОНФЛИКТНОСТЬ БАНКОВ и вайвфронты НА ОДНО ОБРАЩЕНИЕ из ИСХОДНИКА, с разбиением на "
        "ПОЛ (неустранимо) и ВОЗВРАТИМО (снимается дополнением раскладки).",
        hint=[
            "python3 tools/tempo_cli.py lint --selftest",
            "python3 tools/tempo_cli.py lint --kernel volta_fwd_ws",
            "python3 tools/tempo_cli.py lint --all --verbose",
        ],
        anchor="указать на forward/volta_fwd_ws/backward и НЕ указать на декод (EPT=4)",
    ),
    "residency": dict(
        script="residency.py",
        min_py=(3, 12),
        needs_torch=lambda a: False,
        gpu="нет вовсе",
        sudo=False,
        what="Волна -> окно рабочего множества KV -> L2/L1/smem -> вердикт 'переходит ли "
        "сокращение трафика во время' + граница по форме.",
        hint=[
            "python3 tools/tempo_cli.py residency --selftest",
            "python3 tools/tempo_cli.py residency --scan",
            "python3 tools/tempo_cli.py residency --d 128 --Sq 2048 --Sk 2048 --B 1 --H 8 --Hkv 8",
        ],
        anchor="97.7 % попаданий в L2 (ncu, S=4096 H=4 d=128); d=128/Sk=4096 влезает, "
        "d=512/Hkv=1/Sk>=4096 -- нет",
    ),
    "ccab": dict(
        script="cc_ab.py",
        min_py=(3, 8),
        needs_torch=lambda a: False,
        gpu="нет (нужен nvcc/ptxas/cuobjdump)",
        sudo=False,
        what="Регистры, кадр стека, разлив по отчёту, LDL/STL в теле и в циклах, разделяемая "
        "(статическая и динамическая зондом sizeof), занятость. НОЛЬ тактов GPU.",
        hint=[
            "python3 tools/tempo_cli.py ccab qtable        # якорь Q(W): 168/128/80/64",
            "python3 tools/tempo_cli.py ccab selftest      # оба якоря, ~5 мин CPU",
            "python3 tools/tempo_cli.py ccab bwd-one --tile 64,64,256 -D FMHA_BWD_R7",
        ],
        anchor="Q(12/16/24/32)=168/128/80/64; боевой backward 255 рег / 41232 Б smem / 128 потоков",
    ),
    "time": dict(
        script="timeit.py",
        min_py=(3, 8),
        needs_torch=_timeit_needs_torch,
        gpu="да, и ТОЛЬКО на пустой карте (6 гейтов)",
        sudo="для фиксации частот",
        what="ОТНОШЕНИЕ времён двух вариантов на одной форме: медиана парных пораундовых отношений "
        "+ бутстрап-ДИ 95 %. Шесть гейтов, каждый со своим ОТКАЗОМ.",
        hint=[
            "python3 tools/tempo_cli.py time --selftest        # 24 проверки, карта не трогается",
            "python3 tools/tempo_cli.py time --precheck --card 1",
            "python3 tools/tempo_cli.py time --calibrate-idle --card 1   # только на пустой карте",
        ],
        anchor="НЕ СНЯТ НА ЖЕЛЕЗЕ. Подтверждён оцениватель на синтетике (смещение +0.06 %) и "
        "24/24 гейта; опубликованное отношение префилла НЕ воспроизведено ни на одной форме",
    ),
}

ALIASES = {
    "phase": "phases",
    "phaseprof": "phases",
    "smem": "lint",
    "smem_lint": "lint",
    "res": "residency",
    "cc_ab": "ccab",
    "cc": "ccab",
    "timeit": "time",
}


# ------------------------------------------------------------------------------------------------
# ВЫБОР ИНТЕРПРЕТАТОРА
# ------------------------------------------------------------------------------------------------
_ver_cache = {}


def interp_version(path):
    if path in _ver_cache:
        return _ver_cache[path]
    v = None
    if path and os.path.exists(path):
        try:
            out = subprocess.run(
                [path, "-c", "import sys;print('%d.%d.%d' % sys.version_info[:3])"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if out.returncode == 0:
                v = tuple(int(x) for x in out.stdout.strip().split("."))
        except Exception:
            v = None
    _ver_cache[path] = v
    return v


_torch_cache = {}


def has_torch(path):
    if path in _torch_cache:
        return _torch_cache[path]
    ok = False
    if path and os.path.exists(path):
        try:
            r = subprocess.run(
                [path, "-c", "import torch"], capture_output=True, timeout=300
            )
            ok = r.returncode == 0
        except Exception:
            ok = False
    _torch_cache[path] = ok
    return ok


def pick_interp(name, argv):
    """Вернуть (путь, объяснение) или (None, причина отказа)."""
    t = TOOLS[name]
    need_torch = t["needs_torch"](argv)
    minpy = t["min_py"]
    cands = [PY_VLLM, PY_SYS] if need_torch else [PY_SYS, PY_VLLM, sys.executable]
    tried = []
    for p in cands:
        v = interp_version(p)
        if v is None:
            tried.append(f"{p}: нет или не запускается")
            continue
        if v < minpy:
            tried.append(
                f"{p}: {'.'.join(map(str, v))} < требуемых {'.'.join(map(str, minpy))}"
            )
            continue
        if need_torch and not has_torch(p):
            tried.append(f"{p}: {'.'.join(map(str, v))}, но torch не импортируется")
            continue
        why = f"{'.'.join(map(str, v))}"
        if minpy > (3, 8):
            why += f" (требуется >= {'.'.join(map(str, minpy))}: PEP 701, перенос строки внутри f-строки)"
        if need_torch:
            why += " + torch (боевой режим импортирует torch)"
        return p, why
    return None, "ни один интерпретатор не годен:\n    " + "\n    ".join(tried)


# ------------------------------------------------------------------------------------------------
# ОКРУЖЕНИЕ
# ------------------------------------------------------------------------------------------------
def build_env(name):
    e = dict(os.environ)
    e.setdefault("LC_ALL", "C")
    e.setdefault("LANG", "C")  # ncu CSV ломается о ru_RU (NBSP как разделитель тысяч)
    e.setdefault("PYTHONPATH", REPO)
    e.setdefault("CUDA_HOME", CUDA_HOME)
    e.setdefault("CC", "/usr/bin/gcc")
    e.setdefault("CXX", "/usr/bin/g++")
    e.setdefault("CUDAHOSTCXX", "/usr/bin/g++")
    e.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    e.setdefault("TEMPO_PY", PY_VLLM)
    # долгоживущая копия тела для якоря ncu (оригинал жил в сессионном /tmp и будет вычищен)
    local_bank = os.path.join(HERE, "bankaudit.py")
    if os.path.exists(local_bank):
        e.setdefault("TEMPO_BANKAUDIT", local_bank)
    if name == "ncu" and "TEMPO_SUDO_PASS" not in e and "FA2_SUDO_PASS" in e:
        e["TEMPO_SUDO_PASS"] = e["FA2_SUDO_PASS"]
    return e


def preflight(name, argv):
    """Отказы ДО запуска тела. Возвращает список строк-претензий."""
    bad = []
    t = TOOLS[name]
    script = os.path.join(HERE, t["script"])
    if not os.path.exists(script):
        bad.append(f"нет файла инструмента: {script}")
    if name == "ncu":
        offline = {"--which", "--selftest-synth", "--help", "-h"}
        live = not (set(argv) & offline) or ("--deep" in argv)
        if live and not (
            os.environ.get("TEMPO_SUDO_PASS") or os.environ.get("FA2_SUDO_PASS")
        ):
            bad.append(
                "ncu под sudo (драйвер: RmProfilingAdminOnly=1), но TEMPO_SUDO_PASS "
                "не задан -- задайте его в окружении (пароль нигде не печатается)"
            )
        if not os.path.exists(NCU_GOOD):
            bad.append(
                f"годный ncu 2024.1.1 не найден по пути {NCU_GOOD}; "
                f"проверьте `tempo_cli.py ncu --which`"
            )
    if name == "time":
        if _timeit_needs_torch(argv):
            bad.append(
                "БОЕВОЙ замер времени: убедитесь, что карта ПУСТАЯ "
                "(`tempo_cli.py time --precheck --card N`) -- иначе испортите и чужие "
                "замеры, и свои. Гейты инструмента откажут сами, но лучше до запуска"
            )
    return bad


# ------------------------------------------------------------------------------------------------
# ТЕКСТЫ
# ------------------------------------------------------------------------------------------------
MAP_TEXT = r"""
================================================================================================
ТАБЛИЦА: КАКОЙ ВОПРОС -- КАКИМ ИНСТРУМЕНТОМ (и в каком ПОРЯДКЕ)
================================================================================================

ТРИ ПРАВИЛА ПОРЯДКА (нарушение каждого уже давало уверенный неверный ответ):
  A. СПЕРВА ФАЗЫ, ПОТОМ СЧЁТ. Единица анализа -- ФАЗА, а не команда. Счёт команд ЧЕТЫРЕЖДЫ дал
     правку, которая уронила счёт и не уронила время (одна -- на 4.8 % ХУЖЕ).
  B. СПЕРВА КОРРЕКТНОСТЬ, ПОТОМ ВРЕМЯ. timeit не запускает секундомер, пока не прошла сверка;
     не оптимизируйте неверный ответ.
  C. СПЕРВА ЦЕНА ЕДИНИЦЫ, ПОТОМ ОПТИМИЗАЦИЯ ПО НЕЙ. Пока не известно, ЧЕМ мерится (вайвфронт?
     такт? байт HBM?), сокращение "штук" бессмысленно.

+---------------------------------------+-----------------------------------------------------+
| ВОПРОС                                | ПОРЯДОК ПРИМЕНЕНИЯ                                  |
+---------------------------------------+-----------------------------------------------------+
| "почему ядро медленное"               | 1) phases time --pairs  -- РАЗЛОЖИТЬ по фазам, взять|
|  (нет гипотезы вообще)                |    самую жирную; сумма столбца + невязка обязательны|
|                                       | 2) ncu на этом же ядре -- чем платит эта фаза       |
|                                       | 3) lint -- ПОЧЕМУ платит (раскладка)                |
|                                       | 4) ccab -- не разлив ли (регистры/стек/LDL-STL)     |
|                                       | НИКОГДА не начинать с 2-4: без фазы вы чините не то |
+---------------------------------------+-----------------------------------------------------+
| "во что упирается"                    | 1) phases static -- где вообще команды              |
|  (какой ресурс связывает)             | 2) residency -- HBM/L2/тензорный: есть ли ЗАПАС     |
|                                       | 3) ncu -- разделяемая: доля конфликтов              |
|                                       | 4) ccab -- занятость по регистрам/smem              |
|                                       | ОГОВОРКА: канала "задержка при низкой занятости" НЕТ|
|                                       | ни в одном; на ДЕКОДЕ связывает именно он           |
+---------------------------------------+-----------------------------------------------------+
| "стоит ли резать трафик"              | 1) residency --tmeas <мкс>  -- ЗАПАС ПО HBM в разах |
|  (сжатие KV, e4m3/int8, разреженность)|    над обязательным минимумом. Запас x1.00 = резать |
|                                       |    нечего, дальше НЕ ИДТИ                           |
|                                       | 2) только если запас есть -- ccab (цена распаковки  |
|                                       |    в регистрах) и потом time (A/B)                  |
|                                       | ЯКОРЬ-НАПОМИНАНИЕ: 13.4x трафика = x1.00 при d<=128 |
+---------------------------------------+-----------------------------------------------------+
| "влезет ли плитка / хватит ли         | 1) ccab sizeof   -- ДИНАМИЧЕСКАЯ smem зондом        |
|  регистров"                           | 2) ccab bwd-one --tile ... -- регистры, разлив,     |
|                                       |    LDL/STL В ЦИКЛАХ, вердикт ВЛЕЗАЕТ/СТЕНА-255      |
|                                       | 3) ccab regsweep -- где порог MaxLive               |
|                                       | НИ ОДНОГО ТАКТА GPU не нужно                        |
+---------------------------------------+-----------------------------------------------------+
| "правда ли стало быстрее"             | 1) СВЕРКА КОРРЕКТНОСТИ (ваш check) -- ДО секундомера|
|                                       | 2) time --precheck --card N -- можно ли вообще мерить|
|                                       | 3) time (Harness.compare) с entry_probe -- иначе    |
|                                       |    A/B, не дошедший до ядра, пройдёт незамеченным   |
|                                       | 4) если отношение в пределах ДИ -- это НЕ "быстрее" |
+---------------------------------------+-----------------------------------------------------+
| "куда девается разделяемая память"    | 1) lint (статически, без карты и sudo) -- ПОЛ и     |
|                                       |    ВОЗВРАТИМО по каждому обращению                  |
|                                       | 2) ncu --kernel ... -- подтвердить ДОЛЕЙ на железе  |
|                                       | Порядок именно такой: lint даёт кратность на одно   |
|                                       | обращение, ncu -- долю трафика; путать нельзя       |
+---------------------------------------+-----------------------------------------------------+
| "менять ли раскладку/дополнение"      | 1) lint -- ВОЗВРАТИМО > 0 ? если 0, дополнение      |
|                                       |    не даст ничего (это ПОЛ)                         |
|                                       | 2) ccab ab -- не подорожало ли в регистрах          |
|                                       | 3) time -- измерить (последним!)                    |
+---------------------------------------+-----------------------------------------------------+
| "правда ли фаза X дорогая"            | 1) phases time --pairs -- одиночные И пары          |
|                                       | 2) сверить СУММУ столбца; невязка > 20 % = у вас    |
|                                       |    есть НЕНАЗВАННАЯ фаза, ищите её                  |
|                                       | 3) phases static -- сходится ли доля команд с долей |
|                                       |    времени; резкое расхождение = фаза ждёт, а не    |
|                                       |    считает                                          |
+---------------------------------------+-----------------------------------------------------+
| "инструмент вообще годен сейчас"      | tempo_cli.py doctor  (GPU не трогается)             |
| "воспроизводятся ли якоря"            | tempo_cli.py selftest (всё, чему не нужна карта)    |
+---------------------------------------+-----------------------------------------------------+

ЧЕГО В ТАБЛИЦЕ НЕТ И ПОЧЕМУ: вопроса "какое ядро писать" -- на него не отвечает ни один
измеритель; вопроса "почему ответ неверный" -- это сверка с эталоном, а не профилировщик.
"""


def cmd_map():
    print(MAP_TEXT)


def cmd_list():
    print(__doc__.split("ЗАПУСК:")[0].rstrip())
    print("\nПОДКОМАНДЫ\n" + "=" * 96)
    for n, t in TOOLS.items():
        print(f"\n  {n:<10} -> tools/{t['script']}")
        print(f"     {t['what']}")
        print(
            f"     карта: {t['gpu']}   sudo: {t['sudo']}   python >= "
            f"{'.'.join(map(str, t['min_py']))}"
        )
        print(f"     ЯКОРЬ: {t['anchor']}")
    print("\n  map        -- таблица 'какой вопрос -- каким инструментом'")
    print("  doctor     -- состояние среды (GPU не трогается)")
    print("  selftest   -- все самопроверки, которым НЕ нужна карта")
    print(
        "\nЗАПУСК: python3 tools/tempo_cli.py <подкоманда> [аргументы инструмента как есть]"
    )
    print(
        "        python3 tools/tempo_cli.py --dry <подкоманда> ...   -- показать и не запускать"
    )


# ------------------------------------------------------------------------------------------------
# doctor
# ------------------------------------------------------------------------------------------------
def cmd_doctor():
    print("=" * 96)
    print("СОСТОЯНИЕ СРЕДЫ (ни один такт GPU не тратится, память карт не занимается)")
    print("=" * 96)

    print(
        "\n--- ИНТЕРПРЕТАТОРЫ ------------------------------------------------------------"
    )
    for p in dict.fromkeys([PY_SYS, PY_VLLM, sys.executable]):
        v = interp_version(p)
        print(f"  {p:<52} {'.'.join(map(str, v)) if v else 'НЕТ/не запускается'}")

    print(
        "\n--- ИНСТРУМЕНТ -> ИНТЕРПРЕТАТОР (боевой режим) --------------------------------"
    )
    for n, t in TOOLS.items():
        script = os.path.join(HERE, t["script"])
        exists = "есть" if os.path.exists(script) else "НЕТ ФАЙЛА"
        p, why = pick_interp(n, [])
        line = f"  {n:<10} {t['script']:<16} {exists:<10} "
        if p:
            print(line + f"{p}  [{why}]")
        else:
            print(line + f"ОТКАЗ: {why}")

    print(
        "\n--- ЧУЖИЕ ФАЙЛЫ, ОТ КОТОРЫХ ЗАВИСЯТ ЯКОРЯ -------------------------------------"
    )
    deps = [
        (os.path.join(HERE, "bankaudit.py"), "тело якоря ncu (--selftest)"),
        (os.path.join(DATA, "anchor_fwd_ws_phases.json"), "якорь phases replay"),
        (os.path.join(DATA, "mio_wavefronts.txt"), "якорь lint: 90 точек стенда"),
        (REPO, "репозиторий с отгруженными ядрами (lint, ccab, ncu-тела)"),
        (os.path.join(REPO, "docs", "VOLTA_SM70.md"), "источник якорных долей фаз"),
        (CUDA_HOME, "CUDA_HOME для nvcc/ptxas/cuobjdump"),
        (NCU_GOOD, "ЕДИНСТВЕННЫЙ годный ncu (2024.1.1)"),
    ]
    for path, what in deps:
        print(
            f"  {'есть  ' if os.path.exists(path) else 'НЕТ   '} {what}\n         {path}"
        )

    print(
        "\n--- ИНСТРУМЕНТЫ СБОРКИ --------------------------------------------------------"
    )
    for exe in ("g++", "gcc", "nvidia-smi"):
        print(f"  {exe:<12} {shutil.which(exe) or 'НЕ НАЙДЕН В PATH'}")
    for exe in ("nvcc", "ptxas", "cuobjdump"):
        p = os.path.join(CUDA_HOME, "bin", exe)
        print(f"  {exe:<12} {p if os.path.exists(p) else 'НЕТ в CUDA_HOME/bin'}")

    print(
        "\n--- ЛОКАЛЬ (ncu CSV ломается о ru_RU: NBSP как разделитель тысяч) --------------"
    )
    for k in ("LANG", "LC_ALL", "LC_NUMERIC"):
        print(f"  {k:<12} {os.environ.get(k, '(не задана)')}")
    print("  CLI выставляет LC_ALL=C LANG=C всем дочерним процессам.")

    print(
        "\n--- КАРТЫ (только опрос nvidia-smi) -------------------------------------------"
    )
    try:
        q = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,power.draw,clocks.sm,memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        procs = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for ln in q.stdout.strip().splitlines():
            print("  " + ln)
        n = len([x for x in procs.stdout.strip().splitlines() if x.strip()])
        print(f"  чужих compute-процессов на машине: {n}")
    except Exception as e:
        print(f"  nvidia-smi не опрошен: {e}")
    print(
        "  КАРТЫ 2 и 3 -- БОЕВОЙ СЕРВЕР. Не запускать, не убивать, память не занимать."
    )
    print("  НИКОГДА `pkill -f` по шаблону -- только по PID с карт 0-1.")

    print(
        "\n--- СЕКРЕТЫ (значения НЕ печатаются) ------------------------------------------"
    )
    for k in ("TEMPO_SUDO_PASS", "FA2_SUDO_PASS", "SUDO_PW"):
        print(f"  {k:<18} {'задан' if os.environ.get(k) else 'не задан'}")

    print("\n" + "=" * 96)
    print("НЕ УСТАНОВЛЕНО -- читать обязательно")
    print("=" * 96)
    print("""  * doctor проверяет НАЛИЧИЕ, а не ГОДНОСТЬ: что ncu 2024.1.1 существует -- проверено, что он
    сегодня профилирует эту карту -- НЕТ (это `ncu --which --deep`, ему нужны sudo и карта).
  * порог покоя карты (70 Вт) НЕ ЗАМЕРЕН на этих картах -- `time --calibrate-idle` на пустой.
  * что чужой процесс короче 100 мс сейчас НЕ идёт -- не проверено ничем: опрос дискретный.
  * версия torch/CUDA у интерпретатора vllm не сверялась с версией, которой собраны расширения.
  * doctor НЕ проверяет, что якоря воспроизводятся -- это `tempo_cli.py selftest`.""")
    return 0


# ------------------------------------------------------------------------------------------------
# selftest (только то, чему НЕ нужна карта)
# ------------------------------------------------------------------------------------------------
CPU_SELFTESTS = [
    ("lint", ["--selftest"], "ЗАКОН 90/90 + якорь по 4 отгруженным ядрам"),
    ("residency", ["--selftest"], "5 якорей, включая 97.7 % попаданий в L2"),
    ("time", ["--selftest"], "24 проверки гейтов и оценивателя (карта не трогается)"),
    (
        "phases",
        ["replay", "--times", os.path.join(DATA, "anchor_fwd_ws_phases.json")],
        "якорь долей фаз byte-forward 35.1/19.7/15.3/5.7/2.0 %",
    ),
    ("ccab", ["qtable"], "якорь Q(W) = 168/128/80/64"),
    ("ncu", ["--which"], "отбор годного ncu (заглушки и 2025.x отвергаются)"),
]

OK_PAT = re.compile(
    r"ЯКОРЬ ВОСПРОИЗВЕДЁН|ИТОГ САМОПРОВЕРКИ: ВСЕ ЯКОРЯ СОШЛИСЬ|"
    r"закон ОК, якорь ОК|24/24 пройдено|ЯКОРЬ 1: СОШЁЛСЯ|^ВЫБРАН:",
    re.M,
)


def cmd_selftest(dry=False):
    print("=" * 96)
    print("САМОПРОВЕРКИ, КОТОРЫМ НЕ НУЖНА КАРТА")
    print("Не покрыто здесь: ncu --selftest (карта+sudo), phases time (сборка+замер),")
    print("time на железе (нужна ПУСТАЯ карта), ccab bwd (~11 мин nvcc).")
    print("=" * 96)
    rows = []
    for name, argv, what in CPU_SELFTESTS:
        p, why = pick_interp(name, argv)
        script = os.path.join(HERE, TOOLS[name]["script"])
        if p is None:
            rows.append((name, "ОТКАЗ", why[:60]))
            continue
        cmd = [p, script] + argv
        if dry:
            print("  " + " ".join(cmd))
            rows.append((name, "dry", what))
            continue
        print(f"\n>>> {name}: {what}\n    {' '.join(cmd)}")
        r = subprocess.run(
            cmd, capture_output=True, text=True, env=build_env(name), cwd=ROOT
        )
        out = r.stdout + r.stderr
        good = (r.returncode == 0) and bool(OK_PAT.search(out))
        tail = [l for l in out.strip().splitlines() if l.strip()][-3:]
        for l in tail:
            print("      | " + l[:150])
        rows.append(
            (name, "ПРОШЛА" if good else f"НЕ ПРОШЛА (rc={r.returncode})", what)
        )
    print("\n" + "=" * 96)
    bad = 0
    for name, st, what in rows:
        print(f"  {name:<12} {st:<26} {what}")
        if st.startswith(("НЕ", "ОТКАЗ")):
            bad += 1
    print("=" * 96)
    print(f"ИТОГ: {len(rows) - bad} из {len(rows)} прошли")
    print(
        "НАПОМИНАНИЕ: пройденная самопроверка означает, что воспроизводится ЯКОРЬ, а не что"
    )
    print(
        "инструмент прав на вашем ядре. Раздел 'НЕ РАЗОБРАНО' каждого отчёта -- обязателен."
    )
    return 1 if bad else 0


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------
def main(argv):
    dry = False
    args = argv[1:]
    while args and args[0] in ("--dry", "--dry-run"):
        dry = True
        args = args[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        cmd_list()
        return 0
    sub = ALIASES.get(args[0], args[0])
    rest = args[1:]

    if sub == "map":
        cmd_map()
        return 0
    if sub == "doctor":
        return cmd_doctor()
    if sub == "selftest":
        return cmd_selftest(dry=dry)
    if sub not in TOOLS:
        print(f"неизвестная подкоманда: {args[0]}\n", file=sys.stderr)
        cmd_list()
        return 2

    t = TOOLS[sub]
    script = os.path.join(HERE, t["script"])

    if not rest:
        print(f"# {sub} -> tools/{t['script']}\n# {t['what']}\n# ЯКОРЬ: {t['anchor']}")
        print(f"# карта: {t['gpu']}   sudo: {t['sudo']}\n#\n# типовые запуски:")
        for h in t["hint"]:
            print("    " + h)
        print(
            f"#\n# справка самого инструмента: python3 tools/tempo_cli.py {sub} --help"
        )
        if sub == "lint":
            print(
                "# (у lint своей справки нет: без аргументов он печатает ПОЛНЫЙ отчёт по "
                "отгруженным ядрам)"
            )
        return 0

    bad = preflight(sub, rest)
    hard = [b for b in bad if not b.startswith("БОЕВОЙ замер")]
    for b in bad:
        print(("ОТКАЗ: " if b in hard else "ВНИМАНИЕ: ") + b, file=sys.stderr)
    if hard:
        return 2

    p, why = pick_interp(sub, rest)
    if p is None:
        print(f"ОТКАЗ: {why}", file=sys.stderr)
        return 2
    cmd = [p, script] + rest
    print(f"# интерпретатор: {p}  [{why}]", file=sys.stderr)
    print("# " + " ".join(cmd), file=sys.stderr)
    if dry:
        return 0
    return subprocess.run(cmd, env=build_env(sub), cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv))
