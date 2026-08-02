#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ГЕЙТЫ НЕПРОТЕКАНИЯ G1..G8.  Исполняемые, без карты.

Гейт -- не тест «работает ли». Гейт -- это ФАЛЬСИФИКАТОР утверждения «граница плагина
настоящая». Каждый обязан ПАДАТЬ, если утверждение неверно, и лексический гейт здесь самый
слабый: он проверяет СЛОВАРЬ. Главный -- G8: допущение «подача съедает регистры» пишется без
единого запретного слова, и текстовым гейтом невидимо.

ЗАПУСК:  python3 tests/test_gates.py        (карта не нужна)
         python3 -m pytest tests/test_gates.py -q
"""

from __future__ import annotations

import ast
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CORE = os.path.join(ROOT, "tempo", "core")
PLUGINS = os.path.join(ROOT, "tempo", "plugins")

RESULTS = []


def gate(name):
    def deco(fn):
        def wrapper():
            try:
                fn()
                RESULTS.append((name, True, ""))
            except AssertionError as e:
                RESULTS.append((name, False, str(e)))
            except Exception as e:  # гейт, упавший не своим отказом, -- тоже провал
                RESULTS.append((name, False, "%s: %s" % (type(e).__name__, e)))

        wrapper.__name__ = fn.__name__
        return wrapper

    return deco


def _py_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for f in filenames:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


# ============================================================================================
# G1 -- ЛЕКСИЧЕСКИЙ: в конвейере нет ни имён архитектур, ни вшитых чисел железа
# ============================================================================================
BANNED_WORDS = re.compile(
    r"\b(sm_7\d|sm_8\d|volta|ampere|hmma|imma|ldmatrix|mbarrier|prmt|LDS\.|LDG|STS|884|"
    r"m8n8k4|m16n8k16|l1tex__|cutlass|cublas|nvcc|ptxas|ncu)\b",
    re.IGNORECASE,
)
BANNED_CPASYNC = re.compile(r"cp\.async", re.IGNORECASE)
# Литералы железа, встреченные В АРИФМЕТИКЕ (в комментариях и строках -- законны).
HW_LITERALS = {80, 96, 255, 1530, 65536, 32, 128}


def test_g1_lexical():
    hits = []
    for path in _py_files(CORE):
        src = open(path, encoding="utf-8").read()
        # чистим строки и комментарии: гейт проверяет КОД, а не пояснения
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            raise AssertionError("%s не разбирается: %s" % (path, e))
        code_only = _strip_docs(src, tree)
        for m in BANNED_WORDS.finditer(code_only):
            hits.append("%s: имя %r" % (os.path.relpath(path, ROOT), m.group(0)))
        for m in BANNED_CPASYNC.finditer(code_only):
            hits.append("%s: имя %r" % (os.path.relpath(path, ROOT), m.group(0)))
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) or isinstance(node, ast.Compare):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, int):
                        if sub.value in HW_LITERALS:
                            hits.append(
                                "%s:%d: голый литерал железа %d в арифметике"
                                % (os.path.relpath(path, ROOT), sub.lineno, sub.value)
                            )
    assert not hits, "протечка в конвейер:\n  " + "\n  ".join(hits)


def _strip_docs(src, tree):
    """Убрать строковые литералы и комментарии -- останется код."""
    lines = src.split("\n")
    out = []
    for ln in lines:
        # грубо, но достаточно: гейт ловит ИМЕНА В КОДЕ
        ln = re.sub(r"#.*$", "", ln)
        ln = re.sub(r'"""(?:.|\n)*?"""', "", ln)
        ln = re.sub(r'"[^"]*"', '""', ln)
        ln = re.sub(r"'[^']*'", "''", ln)
        out.append(ln)
    text = "\n".join(out)
    text = re.sub(r'"""(?:.|\n)*?"""', "", text)
    return text


# ============================================================================================
# G2 -- ПУСТОЙ ПЛАГИН: структурный отказ, а не AttributeError
# ============================================================================================
def test_g2_null_plugin():
    from tempo.plugins import registry
    from tempo.plugins.base import OpSpec, PluginCapabilityError

    p = registry.load("null")
    op = OpSpec(
        "gemm", "fp16", "fp16", "fp16", "fp32", "k", "k", "n", {"M": 8, "N": 8, "K": 8}
    )
    try:
        list(p.skeletons.variants(op))
    except PluginCapabilityError:
        pass
    except Exception as e:
        raise AssertionError(
            "ожидался PluginCapabilityError, получен %s: %s" % (type(e).__name__, e)
        )
    else:
        raise AssertionError("пустой плагин не отказал вовсе")
    assert p.selftest().green, "самопроверка пустого плагина не зелена"


# ============================================================================================
# G3 -- ПОДСТАВНАЯ АРХИТЕКТУРА: конвейер проходит ЦЕЛИКОМ и принимает ДРУГИЕ решения
# ============================================================================================
def test_g3_nullarch():
    from tempo.core.model.bound import bound
    from tempo.core.search.prune import prune
    from tempo.plugins import registry
    from tempo.plugins.base import OpSpec

    p = registry.load("nullarch")
    op = OpSpec(
        "gemm", "q7", "q7", "q7", "w21", "k", "k", "n", {"M": 64, "N": 64, "K": 16}
    )
    hs = list(p.skeletons.variants(op))
    assert hs, "подставная архитектура не породила ни одной гиперформы"

    # ПЕРИОД СЧИТАН РУКОЙ (см. шапку плагина): X=18, Y=20, Z=28 -> T>=28, связывает Z
    atoms = p.skeletons.estimate_atoms(op, hs[0])
    b = bound(atoms, p.machine.channels(), warps_per_sm=2)
    assert abs(b.T - 28.0) < 1e-9, "период %r, а рукой посчитано 28" % b.T
    assert b.binding == "Z", "связывает %r, а рукой посчитано Z" % b.binding

    cands = prune(p, op, hs)
    assert any(c.kept for c in cands), "отсекатель выбросил ВСЁ"

    # РЕШЕНИЕ ОБЯЗАНО ОТЛИЧАТЬСЯ ОТ БОЕВОГО ПЛАГИНА
    real = registry.load("sm70")
    rop = OpSpec(
        "gemm",
        "fp16",
        "fp16",
        "fp16",
        "fp32",
        "k",
        "k",
        "n",
        {"M": 4096, "N": 15360, "K": 3840},
        tol_rel_l2=1e-3,
    )
    rhs = list(real.skeletons.variants(rop))[:12]
    rc = prune(real, rop, rhs)
    keys_null = {c.hyper.key for c in cands if c.kept}
    keys_real = {c.hyper.key for c in rc if c.kept}
    assert keys_null != keys_real, "план на подставной и на боевой архитектуре СОВПАЛ"
    assert p.machine.channels().keys() != real.machine.channels().keys(), (
        "каналы совпали"
    )


# ============================================================================================
# G4 -- ЗАКРЫТАЯ ТАБЛИЦА: символ вне таблицы -> отказ, а не умолчание
# ============================================================================================
def test_g4_closed_table():
    from tempo.plugins import registry
    from tempo.plugins.base import UnknownSymbol

    for name in ("sm70", "sm80", "nullarch"):
        p = registry.load(name)
        try:
            p.machine.rate("CAP.DEFINITELY_NOT_A_CHANNEL")
        except UnknownSymbol:
            continue
        raise AssertionError(
            "%s: таблица ставок НЕ закрыта -- молча вернула умолчание" % name
        )


# ============================================================================================
# G5 -- ИМПОРТЫ И ПРОЦЕССЫ: core не знает плагинов и не запускает тулчейн
# ============================================================================================
def test_g5_imports_and_processes():
    bad = []
    for path in _py_files(CORE):
        rel = os.path.relpath(path, ROOT)
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    _chk_import(a.name, rel, bad)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    _chk_import(node.module, rel, bad)
            elif isinstance(node, ast.Attribute):
                if node.attr in ("system", "popen", "spawnl", "execv"):
                    v = getattr(node.value, "id", "")
                    if v == "os":
                        bad.append("%s: os.%s в конвейере" % (rel, node.attr))
            elif isinstance(node, ast.Name) and node.id == "subprocess":
                bad.append("%s: subprocess в конвейере" % rel)
    assert not bad, "протечка:\n  " + "\n  ".join(bad)


def _chk_import(mod, rel, bad):
    if mod == "subprocess" or mod.startswith("subprocess."):
        bad.append("%s: импортирует subprocess" % rel)
    if mod.startswith("tempo.plugins"):
        if mod not in ("tempo.plugins.base",) and not mod.startswith(
            "tempo.plugins.base"
        ):
            bad.append(
                "%s: импортирует %s (можно только tempo.plugins.base)" % (rel, mod)
            )


# ============================================================================================
# G6 -- ВОЗМУЩЕНИЕ СТАВКИ: вывод сдвинулся В ПРЕДСКАЗАННУЮ СТОРОНУ и ни в какую другую
# ============================================================================================
def test_g6_perturbation():
    from tempo.core.model.bound import bound
    from tempo.plugins import registry
    from tempo.plugins.base import OpSpec
    from tempo.plugins.sm70_perturbed import FACTOR, PERTURBED_SYMBOL

    base = registry.load("sm70")
    pert = registry.load("sm70_perturbed")
    op = OpSpec(
        "gemm",
        "fp16",
        "fp16",
        "fp16",
        "fp32",
        "k",
        "k",
        "n",
        {"M": 4096, "N": 15360, "K": 3840},
        tol_rel_l2=1e-3,
    )
    h = list(base.skeletons.variants(op))[0]
    atoms = base.skeletons.estimate_atoms(op, h)

    b0 = bound(atoms, base.machine.channels(), warps_per_sm=8)
    b1 = bound(atoms, pert.machine.channels(), warps_per_sm=8)
    ch = PERTURBED_SYMBOL.split(".", 1)[1]

    assert ch in b0.per_channel, "возмущаемый канал не участвует в выводе вовсе"
    got = b1.per_channel[ch]
    want = b0.per_channel[ch] / FACTOR
    assert abs(got - want) < 1e-9, (
        "нагрузка канала %s: ожидалось %.6f (ровно /%.0f), получено %.6f"
        % (ch, want, FACTOR, got)
    )
    for name, v in b0.per_channel.items():
        if name == ch:
            continue
        assert abs(b1.per_channel[name] - v) < 1e-9, (
            "возмущение ОДНОЙ ставки сдвинуло ЧУЖОЙ канал %s: %.6f -> %.6f"
            % (name, v, b1.per_channel[name])
        )
    assert pert.selftest().green, "самопроверка возмущённого плагина не зелена"


# ============================================================================================
# G7 -- ЛИЦЕНЗИОННЫЙ: шапка на исходниках продукта, НЕ на third_party; пути окружения
# ============================================================================================
HEADER_MARK = "SPDX-License-Identifier: LicenseRef-TRL-1.0"
# Каталоги, которые продуктом НЕ являются (записи о замерах, чужой код, журнал).
G7_SKIP_DIRS = (
    "third_party",
    ".git",
    "__pycache__",
    "build",
    "data",
    "profiled",
    "pad",
)

# ЕДИНСТВЕННЫЕ ДВА ФАЙЛА, КОТОРЫМ ПУТЬ ОКРУЖЕНИЯ РАЗРЕШЁН, И У КАЖДОГО СВОЯ ПРИЧИНА.
# Список закрыт намеренно: он и есть определение правила «единственное место с путями».
G7_PATH_ALLOWED = {
    "tempo/cli/env.py": "ЭТО И ЕСТЬ то самое единственное место (правило Р8)",
    "tests/test_gates.py": "сам гейт: он обязан знать, что ищет",
}

# РАЗЛИЧЕНИЕ, БЕЗ КОТОРОГО ГЕЙТ ПРЕВРАЩАЕТСЯ ВО ВРЕДИТЕЛЯ:
#   ИСПОЛНЯЕМЫЙ путь (то, что будет запущено) -- нарушение, чинить;
#   ЗАПИСЬ О ЗАМЕРЕ или ЦИТАТА ИСТОЧНИКА (где снято / откуда взято) -- НЕ ТРОГАТЬ.
# Переписать путь в записи о замере = ПОДДЕЛАТЬ ПРОИСХОЖДЕНИЕ. Поэтому гейт смотрит только на
# КОД: строковые литералы и комментарии он вычищает ровно так же, как в G1.
RECORD_SUFFIXES = (".json", ".txt", ".md", ".ncu-rep", ".log")


def test_g7_license_and_paths():
    missing, extra, conda = [], [], []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in G7_SKIP_DIRS]
        for f in filenames:
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
            if rel.startswith("third_party"):
                if f.endswith((".py", ".cu", ".cuh", ".cpp")):
                    src = open(path, encoding="utf-8", errors="ignore").read(400)
                    if HEADER_MARK in src:
                        extra.append(rel + " -- ЧУЖОЙ код помечен НАШЕЙ шапкой")
                continue
            if not f.endswith((".py", ".cu", ".cuh", ".cpp", ".h", ".sh")):
                continue
            src = open(path, encoding="utf-8", errors="ignore").read()
            if rel.startswith(("tempo/", "tests/", "kernels/")) and f.endswith(
                (".py", ".cu", ".cuh", ".cpp")
            ):
                if HEADER_MARK not in src[:600]:
                    missing.append(rel)
            if rel in G7_PATH_ALLOWED or f.endswith(RECORD_SUFFIXES):
                continue
            code = (
                _strip_docs(src, None)
                if f.endswith(".py")
                else re.sub(r"#.*$", "", src, flags=re.M)
            )
            if "/opt/conda" in code:
                conda.append(rel)
    assert not missing, "нет лицензионной шапки:\n  " + "\n  ".join(missing)
    assert not extra, "лишняя шапка:\n  " + "\n  ".join(extra)
    assert not conda, (
        "исполняемый путь окружения вшит мимо tempo/cli/env.py:\n  "
        + "\n  ".join(conda)
    )


# ============================================================================================
# G8 -- ФАЛЬСИФИКАТОР ДОПУЩЕНИЯ (ГЛАВНЫЙ)
# ============================================================================================
def test_g8_assumption_falsifier():
    from tempo.core.search.prune import prune
    from tempo.plugins import registry
    from tempo.plugins.base import OpSpec

    a = registry.load("nullarch")
    b = registry.load("nullarch_async")

    # ДВА ПЛАГИНА РАЗЛИЧАЮТСЯ РОВНО ОДНИМ ПОЛЕМ
    ta, tb = a.sync.transactions()[0], b.sync.transactions()[0]
    assert ta.consumes_registers is True and tb.consumes_registers is False

    op = OpSpec(
        "gemm", "q7", "q7", "q7", "w21", "k", "k", "n", {"M": 64, "N": 64, "K": 16}
    )
    hs = list(a.skeletons.variants(op))
    ra = {h.key: a.skeletons.resources_of(op, h)[0] for h in hs}
    rb = {h.key: b.skeletons.resources_of(op, h)[0] for h in hs}
    deep = [k for k in ra if k.endswith("d3")]
    assert deep, "нет варианта с глубокой подачей"
    assert all(rb[k] < ra[k] for k in deep), (
        "глубина подачи стоит регистров ОДИНАКОВО у обоих -- допущение НЕ является данными"
    )

    ca = {c.hyper.key: c for c in prune(a, op, hs)}
    cb = {c.hyper.key: c for c in prune(b, op, hs)}
    занятость_a = {k: c.verdict.occ.warps_per_sm for k, c in ca.items() if c.verdict}
    занятость_b = {k: c.verdict.occ.warps_per_sm for k, c in cb.items() if c.verdict}
    assert занятость_a != занятость_b, (
        "ВЕРДИКТ НЕ СДВИНУЛСЯ: конвейер считает цену подачи САМ, а не берёт её у плагина. "
        "Это ровно та протечка, которую лексический гейт не видит."
    )
    assert b.selftest().green, "самопроверка фальсификатора не зелена"


# ============================================================================================
# РЕЕСТР ЗАКОНОВ (G9..G11).  Гейты того же рода, что G1..G8: каждый обязан ПАДАТЬ, если
# утверждение неверно.  Проверяемое утверждение: «открытие внесено в продукт, а не описано».
# ============================================================================================
# Что считается ПРОДУКТОМ при поиске маркеров и запрещённых формулировок.  Список закрыт
# намеренно -- он и есть определение слова «продукт» для этих трёх гейтов.
LAW_DIRS = (
    "tempo",
    "tools",
    "tests",
    "kernels",
    "docs",
    "bench",
    "README.md",
    "CHANGELOG.md",
)
LAW_SUFFIXES = (".py", ".cu", ".cuh", ".cpp", ".h", ".md", ".json", ".inc")


def _laws():
    from tempo.core import laws

    return laws, laws.load(ROOT)


def test_g9_law_record():
    """G9 -- ЗАКОН БЕЗ ОБЛАСТИ И БЕЗ ФАЛЬСИФИКАТОРА ЕСТЬ МИНА.

    Проверяется ровно то же, что `Rate.check()` проверяет у ЧИСЛА, только у УТВЕРЖДЕНИЯ:
    происхождение (отчёт + числа + карта), фальсификатор, область.  Замер при чужом процессе
    на карте недействителен -- у закона так же, как у ставки.
    """
    laws, all_laws = _laws()
    assert all_laws, "реестр пуст: гейт нечего проверять"
    bad = []
    for l in all_laws:
        bad += laws.check_record(l)
    bad += laws.check_links(all_laws)
    assert not bad, "записи реестра неполны:\n  " + "\n  ".join(bad)

    # ФАЛЬСИФИКАТОР САМОГО ГЕЙТА: запись без области и с чужими процессами обязана быть
    # отвергнута.  Без этой пары строк гейт зелен и на пустой проверке.
    ghost = laws.Law(
        id="L-GHOST",
        kind="ЗАКОН",
        statement="закон без области",
        anchor=laws.Anchor(
            report="EV_ghost.md",
            numbers=("1",),
            status="MEASURED",
            card=laws.Card(index=0, clock_mhz=1530, foreign_procs=2, date=""),
        ),
        falsifier=laws.Falsifier(what="", how=""),
        scope={},
    )
    probs = laws.check_record(ghost)
    assert any("область" in p for p in probs), "пустая область НЕ отвергнута"
    assert any("фальсификатор" in p for p in probs), (
        "отсутствие фальсификатора НЕ отвергнуто"
    )
    assert any("чужих процессах" in p for p in probs), (
        "чужой процесс на карте НЕ отвергнут"
    )


def test_g10_law_home():
    """G10 -- ЗАКОН БЕЗ МЕСТА.  Место обязано РАЗРЕШАТЬСЯ: путь есть, символ в файле находится.

    Ровно этот гейт поймал бы субаддитивность фаз: требование печатать оговорку записано в
    журнале наблюдателя, а инструмент её не печатал.  Объявление проверяется в ОБЕ стороны:
    закон, объявленный внесённым без исполняемого места, -- завышение; долг, у которого место
    и падающая проверка уже есть, -- занижение.
    """
    laws, all_laws = _laws()
    bad = []
    for l in all_laws:
        bad += laws.check_homes(l, ROOT)
    bad += laws.check_selftests(all_laws)
    assert not bad, "места законов не разрешаются:\n  " + "\n  ".join(bad)

    # ОПИСЬ ДОЛГОВ ОБЯЗАНА БЫТЬ НЕПУСТОЙ ИЛИ ЧЕСТНО ПУСТОЙ: молчаливое «долгов нет» при живых
    # долгах -- это и есть та ложь, ради которой заводится реестр.  Здесь проверяется лишь то,
    # что вид ДОЛГ вообще различается инструментом.
    debts = [l for l in all_laws if l.kind == "ДОЛГ"]
    wired = [l for l in all_laws if l.kind != "ДОЛГ"]
    assert wired, (
        "в реестре нет НИ ОДНОГО внесённого закона -- реестр описывает пустоту"
    )
    for l in debts:
        assert not (l.wired and l.has_selftest), (
            "%s объявлен долгом, но уже оплачен" % l.id
        )

    # docs/LAWS.md ПОРОЖДАЕТСЯ из реестра: проза, разошедшаяся с данными, учит неверному.
    path = os.path.join(ROOT, "docs", "LAWS.md")
    assert os.path.exists(path), "docs/LAWS.md не порождён"
    with open(path, encoding="utf-8") as fh:
        got = fh.read()
    want = laws.render_markdown(all_laws)
    assert got == want, (
        "docs/LAWS.md разошёлся с реестром. Пересоздать: "
        "python3 -m tempo.cli laws --md > docs/LAWS.md"
    )


def test_g11_two_way_binding():
    """G11 -- ДВУСТОРОННЯЯ ПРИВЯЗКА И ЗАПРЕТ ОПРОВЕРГНУТОГО.

    (а) каждый маркер `LAW=<id>` в дереве обязан разрешаться в запись реестра;
    (б) у каждого ВНЕСЁННОГО закона обязан быть хотя бы один ссылающийся маркер;
    (в) опровергнутой формулировки в дереве быть не должно -- кроме ЦИТАТЫ, рядом с которой
        назван идентификатор опровергающей записи.

    Пункт (в) -- единственная падающая проверка, какая у опровержения вообще возможна: у
    закона место СЧИТАЕТ, у опровержения считать нечего, оно ГОВОРИТ.  Без него опровержение
    живёт ровно до первого, кто перепишет старый вывод по памяти.
    """
    laws, all_laws = _laws()
    idx = laws.by_id(all_laws)
    found = laws.scan_markers(ROOT, LAW_DIRS, LAW_SUFFIXES)

    unknown = sorted(set(found) - set(idx))
    assert not unknown, (
        "маркеры LAW= ссылаются на записи, которых в реестре нет: %s" % unknown
    )

    unreferenced = [l.id for l in all_laws if l.kind != "ДОЛГ" and l.id not in found]
    assert not unreferenced, (
        "внесённые законы, на которые в коде НЕТ маркера LAW=: %s. Дрейф возможен в обе "
        "стороны, поэтому привязка обязана быть двусторонней." % unreferenced
    )

    bad = laws.check_forbidden(all_laws, ROOT, LAW_DIRS, LAW_SUFFIXES)
    assert not bad, "опровергнутое вернулось в дерево:\n  " + "\n  ".join(bad)

    # ФАЛЬСИФИКАТОР ГЕЙТА: проверка обязана ловить фразу, перенесённую по строке и написанную
    # другим регистром.  Первая её редакция этого не умела и пропускала три места из пяти.
    ref = [l for l in all_laws if l.forbidden]
    assert ref, "в реестре нет ни одного запрета -- пункт (в) проверяет пустоту"
    pat = ref[0].forbidden[0].pattern
    words = pat.split()
    assert len(words) > 1, (
        "запрет из одного слова не годится для проверки переноса строки"
    )
    broken = (" ".join(words[:1]) + "\n      " + " ".join(words[1:])).upper()
    assert laws.normalize(pat) in laws.normalize(broken), (
        "проверка запретов не переживает перенос строки и смену регистра -- то есть её обходят "
        "случайно, простым переформатированием"
    )


# ============================================================================================
GATES = [
    ("G1 лексический", test_g1_lexical),
    ("G2 пустой плагин", test_g2_null_plugin),
    ("G3 подставная архитектура", test_g3_nullarch),
    ("G4 закрытая таблица", test_g4_closed_table),
    ("G5 импорты и процессы", test_g5_imports_and_processes),
    ("G6 возмущение ставки", test_g6_perturbation),
    ("G7 лицензия и пути", test_g7_license_and_paths),
    ("G8 фальсификатор допущения", test_g8_assumption_falsifier),
    ("G9 запись закона", test_g9_law_record),
    ("G10 место закона", test_g10_law_home),
    ("G11 двусторонняя привязка", test_g11_two_way_binding),
]


def main():
    ok = 0
    print("ГЕЙТЫ НЕПРОТЕКАНИЯ (карта не нужна)")
    for name, fn in GATES:
        try:
            fn()
            print("  ПРОЙДЕН  %s" % name)
            ok += 1
        except AssertionError as e:
            print("  ПАДЁТ    %s\n           %s" % (name, e))
        except Exception as e:
            print("  ПАДЁТ    %s\n           %s: %s" % (name, type(e).__name__, e))
    print("ИТОГ: %d/%d" % (ok, len(GATES)))
    return 0 if ok == len(GATES) else 1


if __name__ == "__main__":
    sys.exit(main())
