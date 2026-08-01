# -*- coding: utf-8 -*-
"""ПЕРЕБИРАТЕЛЬ ДОПОЛНЕНИЙ: ищем МИНИМУМ, а не ПРИЧИНУ.

ЗАЧЕМ ЭТОТ ФАЙЛ СУЩЕСТВУЕТ. Статический линтер (tools/smem_lint.py) судит раскладку правилом
«шаг кратен 32 словам». Правило промахнулось ТРИ РАЗА ИЗ ТРЁХ на настоящих виновниках:

    volta_fwd_ws, плитка K, шаг LDK8 = 272 Б = 68 слов  -- на 32 НЕ делится, а даёт 74.5 % конфликтов;
    там же sQ, шаг 130 слов                             -- по строкам ЧИСТ, а пролог двумя STS.64 бьёт;
    cutlass-форвард, накопитель эпилога, шаг 130 слов   -- 130 mod 32 = 2 входит в резонанс с шагом
                                                           СТОЛБЦА ИТЕРАТОРА, который ровно 2 слова.

Настоящий критерий -- РЕЗОНАНС шага раскладки с шагом столбца итератора доступа, то есть свойство
ПАРЫ, а не раскладки. Вывести его из исходника трудно, и рассуждение здесь -- плохой инструмент.
Но ИСКАТЬ ЕГО НЕ НУЖНО: дополнение -- однопараметрическое семейство pad = 0..31 слов, а ворота
(побитовое равенство ответа, счётчик конфликтов, размер разделяемой) дешёвые и автоматические.

    искали ПРИЧИНУ, когда достаточно искать МИНИМУМ

ПОБОЧНЫЙ ПРОДУКТ, РАДИ КОТОРОГО СТОИТ СМОТРЕТЬ НА ВСЮ КРИВУЮ, А НЕ НА МИНИМУМ. Кривая
`pad -> конфликты` ПЕРИОДИЧНА, и её период p -- это и есть тот самый шаг столбца итератора,
который не выводится из исходника. Инструмент печатает период явно (`report`), поэтому перебор
не только чинит, но и ОБЪЯСНЯЕТ -- и объяснение получено ЗАМЕРОМ, а не рассуждением.

ПОРЯДОК РАБОТЫ (он важен: карта на этой машине общая)
=====================================================
Внутренний цикл перебора -- это N СБОРОК (только процессор) и лишь короткое чтение счётчика
(карта). Счётчик конфликтов от соседа по карте НЕ ЗАВИСИТ, время -- зависит; поэтому здесь НЕТ
и не будет замера времени. Отсюда команды разнесены:

    padsweep.py plan   --spec S --group G [--pads 0-31]   порождение наложений + диффы   (ноль GPU)
    padsweep.py build  --spec S --group G [--jobs 4]      N сборок                        (ноль GPU)
    padsweep.py run    --spec S --group G                 побитово + конфликты            (КАРТА)
    padsweep.py report --spec S --group G                 кривая, минимум, ПЕРИОД         (ноль GPU)
    padsweep.py find   --spec S                           кандидаты из ncu построчно      (КАРТА)
    padsweep.py body   --spec S --group G --pad N         один прогон -- это профилирует ncu
    любая команда с --dry                                 только СТОИМОСТЬ, ничего не делать

ТРИ ВЕРДИКТА НА ВАРИАНТ, И ПОРЯДОК ИМЕННО ТАКОЙ
===============================================
    1. ПОБИТОВО    -- ответ не изменился (дополнение обязано быть тождественным ПО ЗНАЧЕНИЮ);
    2. КОНФЛИКТЫ   -- сколько осталось (счётчик l1tex, маршрут А из tools/ncu.py);
    3. РАЗДЕЛЯЕМАЯ -- на сколько байт выросла и влезает ли (48 КБ на CTA / 96 КБ на SM).
Вариант, провалившийся по ПЕРВОМУ, дальше НЕ РАССМАТРИВАЕТСЯ: корректность идёт раньше цены.
Быстрее и с меньшим числом конфликтов, но с другим ответом -- это не результат, а брак.

ЧЕМ ЭТО НЕ ЯВЛЯЕТСЯ
===================
* Не правит боевое дерево НИ ОДНИМ БАЙТОМ. Каждый вариант -- ДВОЙНИК, порождённый tools/twin.py
  из боевого исходника плюс наложение, с воротами дрейфа (md5 + якорь ровно один раз). Наложения
  порождаются ЗДЕСЬ и тоже не хранятся руками.
* Не меряет время. Совсем. Причина -- в шапке.
* Не заменяет мышление на всём классе задач: механически переписывается ШАГ раскладки. Раскладки,
  где дополнение выражается иначе (перестановка, swizzle, XOR-адресация), инструмент честно
  выносит в раздел НЕРАЗОБРАННОЕ, а не молчит.

СЛЕПЫЕ ЗОНЫ -- в конце файла.
"""

import argparse
import collections
import datetime
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

# ЛОВУШКА ОКРУЖЕНИЯ (стоила одного ложного вывода в этом же каталоге): в tempo/tools/ лежит СВОЙ
# timeit.py. Каталог скрипта попадает в sys.path ПЕРВЫМ, и `import torch` подхватывает чужой файл.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]


def _load_mod(name, path):
    """Импорт соседа ПО ПУТИ -- каталог из sys.path мы только что выбросили (см. выше)."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


twin = _load_mod("tempo_twin", os.path.join(_HERE, "twin.py"))
ncu = _load_mod("tempo_ncu", os.path.join(_HERE, "ncu.py"))

PY = os.environ.get("TEMPO_PY", "/opt/conda/miniconda3/envs/vllm/bin/python")
CUDA_HOME = "/opt/conda/miniconda3/envs/cuda128"
CUOBJDUMP = os.path.join(CUDA_HOME, "bin", "cuobjdump")
DEFAULT_WORK = "./pad"
IDENT = r"[A-Za-z_]\w*"
# Пределы Volta. 48 КБ -- статический предел на блок без opt-in; 96 КБ -- вся разделяемая на SM.
SMEM_CTA_DEFAULT = 48 * 1024
SMEM_SM = 96 * 1024


class PadError(RuntimeError):
    pass


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def hms(sec):
    sec = int(sec)
    return "%d:%02d:%02d" % (sec // 3600, (sec % 3600) // 60, sec % 60)


# =================================================================================================
# 1. РАЗБОР ИСХОДНИКА: где массив объявлен, где он индексируется, ЧЕМ ВЫРАЖЕН ШАГ
# =================================================================================================
def balanced(text, i):
    """i указывает на '['. Вернуть (нач_содержимого, кон_содержимого) по балансу скобок."""
    assert text[i] == "["
    d, j = 0, i
    while j < len(text):
        if text[j] in "[(":
            d += 1
        elif text[j] in "])":
            d -= 1
            if d == 0:
                return i + 1, j
        j += 1
    raise PadError("скобка [ в позиции %d не закрыта" % i)


def line_of(text, pos):
    return text.count("\n", 0, pos)


def line_spans(text):
    """Смещения начала каждой строки + смещение конца файла."""
    out, p = [0], text.find("\n")
    while p != -1:
        out.append(p + 1)
        p = text.find("\n", p + 1)
    out.append(len(text))
    return out


class Site(object):
    """Одно место, где встречается имя массива."""

    def __init__(self, kind, span, whole, note=""):
        self.kind = kind  # decl | index | via   (via -- индекс приехал переменной)
        self.span = span  # (нач, кон) выражения, которое переписываем
        self.whole = whole  # (нач, кон) всего вхождения (для отчёта)
        self.note = note

    def __repr__(self):
        return "%s[%d,%d) %s" % (self.kind, self.span[0], self.span[1], self.note)


def find_sites(text, name, lookback=20):
    """Все вхождения массива `name`: объявление, прямые индексы и индексы ЧЕРЕЗ ПЕРЕМЕННУЮ.

    Возвращает (sites, unparsed). В unparsed попадает всё, чего разбор не понял -- ПЕЧАТАТЬ
    ВСЕГДА: пустой список найденного при непустом unparsed НЕ означает «чисто».
    """
    sites, unparsed, seen_via = [], [], {}
    for m in re.finditer(r"\b%s\s*\[" % re.escape(name), text):
        br = text.index("[", m.end() - 1)
        c0, c1 = balanced(text, br)
        body = text[c0:c1]
        pre = text[max(0, m.start() - 120) : m.start()]
        is_decl = re.search(r"__shared__[^;{}()]*$", pre) is not None
        if is_decl:
            sites.append(Site("decl", (c0, c1), (m.start(), c1 + 1), "объявление"))
            continue
        if re.match(r"^\s*%s\s*$" % IDENT, body):
            # Индекс приехал ПЕРЕМЕННОЙ. Ищем её определение выше по тексту -- шаг живёт ТАМ.
            var = body.strip()
            defs = [
                d
                for d in re.finditer(r"\b%s\s*=(?!=)" % re.escape(var), text)
                if d.start() < m.start()
            ]
            defs = [
                d
                for d in defs
                if line_of(text, m.start()) - line_of(text, d.start()) <= lookback
            ]
            if not defs:
                unparsed.append(
                    (
                        "%s[%s] стр.%d" % (name, var, line_of(text, m.start()) + 1),
                        "индекс приехал переменной, её определения не нашлось в %d строках "
                        "выше -- ШАГ ЛЕЖИТ НЕ ЗДЕСЬ, дополнение НЕ ПОСТРОЕНО"
                        % lookback,
                    )
                )
                continue
            d = defs[-1]
            end = text.find(";", d.end())
            if end < 0:
                unparsed.append(
                    ("%s[%s]" % (name, var), "определение переменной без ';'")
                )
                continue
            key = (d.end(), end)
            if key not in seen_via:  # одна переменная -- один переписываемый участок
                seen_via[key] = True
                sites.append(
                    Site(
                        "via",
                        (d.end(), end),
                        (d.start(), end),
                        "индекс через переменную %s (стр.%d)"
                        % (var, line_of(text, d.start()) + 1),
                    )
                )
            sites.append(
                Site(
                    "index",
                    (c0, c0),
                    (m.start(), c1 + 1),
                    "%s[%s] -- шаг взят из определения переменной" % (name, var),
                )
            )
            continue
        sites.append(Site("index", (c0, c1), (m.start(), c1 + 1), "прямой индекс"))
    if not any(s.kind == "decl" for s in sites):
        unparsed.append(
            (
                name,
                "ОБЪЯВЛЕНИЯ `__shared__ %s[...]` в файле НЕТ -- размер дополнить "
                "нечем (массив может быть extern/динамическим)" % name,
            )
        )
    return sites, unparsed


def pick_stride(text, sites, forced=None):
    """Чем выражен ШАГ. Кандидат обязан стоять И в объявлении, И в индексных выражениях.

    Возвращает (токен, таблица_обоснования). Ранжирование печатается всегда -- выбор шага это
    ЕДИНСТВЕННОЕ место, где инструмент угадывает, и угадывание должно быть видно.
    """
    decl = [s for s in sites if s.kind == "decl"]
    idxs = [s for s in sites if s.kind in ("index", "via") and s.span[1] > s.span[0]]
    if not decl:
        return forced, []
    decl_ids = set(re.findall(IDENT, text[decl[0].span[0] : decl[0].span[1]]))
    cnt = collections.Counter()
    starred = collections.Counter()
    for s in idxs:
        body = text[s.span[0] : s.span[1]]
        ids = set(re.findall(IDENT, body))
        for t in ids & decl_ids:
            cnt[t] += 1
        for t in re.findall(r"\*\s*(%s)\b" % IDENT, body):
            if t in decl_ids:
                starred[t] += 1
    table = sorted(((cnt[t], starred[t], t) for t in cnt), reverse=True)
    if forced:
        return forced, table
    if not table:
        return None, table
    return table[0][2], table


# =================================================================================================
# 2. ПОРОЖДЕНИЕ ВАРИАНТА: те же участки, шаг S -> (S + pad)
# =================================================================================================
def rewrite_text(text, spans, stride, pad):
    """Заменить ЦЕЛЫЕ СЛОВА `stride` на `(stride + pad)` ТОЛЬКО внутри указанных участков.

    Почему «только внутри»: на той же СТРОКЕ, что и обращение, обычно стоит граница цикла
        for (int w = 0; w < WN; ++w) { ... sRedM[(...) * WN + w] ... }
    и её дополнять НЕЛЬЗЯ -- иначе читаются лишние (мусорные) слова, и ответ меняется. Замена по
    строке дала бы тихо неверное ядро, прошедшее сборку.
    """
    rx = re.compile(r"\b%s\b" % re.escape(stride))
    out, pos = [], 0
    for a, b in sorted(spans):
        if a < pos:
            raise PadError(
                "участки переписывания перекрываются: [%d,%d) после %d" % (a, b, pos)
            )
        out.append(text[pos:a])
        out.append(rx.sub("(%s + %d)" % (stride, pad), text[a:b]))
        pos = b
    out.append(text[pos:])
    return "".join(out)


def spans_to_edits(text, new_text, spans, name, stride, pad):
    """Свернуть переписанные участки в правки наложения: ЯКОРЬ = целые строки, УНИКАЛЬНЫЕ в файле.

    Переписывание не добавляет и не убирает переводов строк, поэтому нумерация строк в исходном
    и новом тексте совпадает -- можно резать обе версии одними и теми же границами.
    """
    ls_a, ls_b = line_spans(text), line_spans(new_text)
    if len(ls_a) != len(ls_b):
        raise PadError(
            "переписывание изменило число строк -- якоря по строкам недействительны"
        )
    ranges = []
    for a, b in spans:
        ranges.append([line_of(text, a), line_of(text, max(a, b - 1))])
    ranges.sort()

    def merge(rs):
        out = []
        for r in rs:
            if out and r[0] <= out[-1][1]:
                out[-1][1] = max(out[-1][1], r[1])
            else:
                out.append(list(r))
        return out

    ranges = merge(ranges)
    # расширяем каждый диапазон до УНИКАЛЬНОСТИ якоря, затем сливаем те, что сошлись
    for _ in range(12):
        changed = False
        for r in ranges:
            for _try in range(10):
                anchor = text[ls_a[r[0]] : ls_a[r[1] + 1]]
                if text.count(anchor) == 1:
                    break
                if r[1] + 2 < len(ls_a):
                    r[1] += 1
                elif r[0] > 0:
                    r[0] -= 1
                else:
                    break
                changed = True
        merged = merge(sorted(ranges))
        if merged != ranges:
            ranges, changed = merged, True
        if not changed:
            break
    edits, bad = [], []
    for r in ranges:
        anchor = text[ls_a[r[0]] : ls_a[r[1] + 1]]
        repl = new_text[ls_b[r[0]] : ls_b[r[1] + 1]]
        if text.count(anchor) != 1:
            bad.append(
                (
                    "%s стр.%d-%d" % (name, r[0] + 1, r[1] + 1),
                    "якорь не удалось сделать однозначным (встречается %d раз) -- правка НЕ "
                    "построена" % text.count(anchor),
                )
            )
            continue
        if anchor == repl:
            continue
        edits.append(
            {
                "anchor": anchor,
                "replace": repl,
                "id": "%s_l%d" % (name, r[0] + 1),
                "why": "дополнение шага %s -> (%s + %d) на массиве %s"
                % (stride, stride, pad, name),
            }
        )
    return edits, bad


def build_overlay(spec, group, pad):
    """Наложение для одного значения дополнения. Ничего не пишет -- возвращает (наложение, отчёт)."""
    root = spec["prod_root"]
    tgt_rel = spec["target"]
    tgt = os.path.join(root, tgt_rel)
    text = read(tgt)
    names = spec["groups"][group]["arrays"]
    forced = spec["groups"][group].get("stride")
    all_spans, unparsed, strides, sites_all = [], [], {}, {}
    for nm in names:
        sites, un = find_sites(text, nm, spec.get("lookback", 20))
        unparsed += un
        st, table = pick_stride(text, sites, forced)
        strides[nm] = (st, table)
        sites_all[nm] = sites
        if st is None:
            unparsed.append(
                (
                    nm,
                    "ШАГ НЕ ОПОЗНАН: ни один идентификатор не стоит и в объявлении, "
                    "и в индексах. Укажите его в спецификации полем groups.<g>.stride",
                )
            )
            continue
        for s in sites:
            if s.span[1] > s.span[0] and re.search(
                r"\b%s\b" % re.escape(st), text[s.span[0] : s.span[1]]
            ):
                all_spans.append(s.span)
            elif s.span[1] > s.span[0]:
                unparsed.append(
                    (
                        "%s %s" % (nm, s.note),
                        "участок не содержит шага %s -- дополнение к нему не применено"
                        % st,
                    )
                )
    if not all_spans:
        raise PadError(
            "для группы «%s» не построено НИ ОДНОГО участка переписывания.\n"
            "  НЕРАЗОБРАННОЕ:\n    %s"
            % (group, "\n    ".join("%s: %s" % u for u in unparsed) or "(пусто)")
        )
    # шаг у группы должен быть ОДИН -- иначе это не одна раскладка, а две разные задачи
    uniq = {strides[n][0] for n in names if strides[n][0]}
    if len(uniq) > 1:
        raise PadError(
            "в группе «%s» РАЗНЫЕ шаги %s -- перебирать их одним параметром нельзя; "
            "разнесите по группам" % (group, sorted(uniq))
        )
    stride = uniq.pop()
    all_spans = sorted(set(all_spans))
    new_text = rewrite_text(text, all_spans, stride, pad)
    edits, bad = spans_to_edits(text, new_text, all_spans, "+".join(names), stride, pad)
    unparsed += bad
    ov = [{"file": tgt_rel, "md5": md5f(tgt), "edits": edits}]
    # НЕСУЩИЙ .cu. Он копируется в двойник НЕ ради правки, а ради РАЗРЕШЕНИЯ ИМЕНИ: `#include "x.cuh"`
    # ищется СНАЧАЛА рядом с включающим файлом. Компилируй мы боевой .cu -- он подтянул бы БОЕВОЙ
    # заголовок, дополнение не подействовало бы НИ НА ЧТО, и все 32 варианта собрались бы в одно
    # ядро («дополнение ничего не меняет» -- правдоподобный неверный ответ).
    car_rel = spec["carrier"]
    car = os.path.join(root, car_rel)
    inc = '#include "%s"' % os.path.basename(tgt_rel)
    ct = read(car)
    if ct.count(inc) != 1:
        raise PadError(
            "в несущем %s строка %s встречается %d раз -- якорь неоднозначен"
            % (car_rel, inc, ct.count(inc))
        )
    ov.append(
        {
            "file": car_rel,
            "md5": md5f(car),
            "edits": [
                {
                    "anchor": inc,
                    "replace": inc
                    + "   // [ДВОЙНИК padsweep %s/%s pad=%d]"
                    % (spec["name"], group, pad),
                    "id": "carrier_marker",
                    "why": "несущий .cu копируется в двойник, чтобы кавычечный #include взял "
                    "ДОПОЛНЕННЫЙ заголовок, а не боевой",
                }
            ],
        }
    )
    report = {
        "stride": stride,
        "strides": {n: strides[n][1] for n in names},
        "sites": {n: [(s.kind, s.note) for s in sites_all[n]] for n in names},
        "spans": len(all_spans),
        "edits": len(edits),
        "unparsed": unparsed,
    }
    return ov, report


# =================================================================================================
# 3. КАТАЛОГИ ВАРИАНТОВ
# =================================================================================================
def vdir(spec, group, pad):
    # pad == "li" -- ОТДЕЛЬНЫЙ вариант БАЗЫ, собранный с -lineinfo: он нужен ТОЛЬКО поиску
    # кандидатов (посточечная привязка), и держать его отдельно обязательно. Подмешай -lineinfo
    # в общие флаги -- ninja сочтёт устаревшими ВСЕ 33 варианта и перестроит их (4 часа), а
    # главное, перебор пошёл бы на ядре с другой отладочной обвязкой, чем измерялся.
    if pad == "li":
        return os.path.join(spec["work"], spec["name"], group, "prod_li")
    return os.path.join(
        spec["work"], spec["name"], group, "prod" if pad is None else "pad%02d" % pad
    )


def gen_twin(spec, group, pad, quiet=False):
    """Порождение двойника ЧЕРЕЗ tools/twin.py -- с его воротами дрейфа, а не правкой файлов."""
    d = vdir(spec, group, pad)
    ovp = os.path.join(d, "overlay.json")
    if pad is None:
        # БАЗА: боевой исходник без единой правки, кроме метки в несущем. Нужна, чтобы доказать,
        # что pad=0 (то есть `(WN + 0)`) собирается В ТЕ ЖЕ КОМАНДЫ, и разметка стоит РОВНО НОЛЬ.
        ov, rep = build_overlay(spec, group, 0)
        ov = [ov[1]]
        rep = {
            "stride": rep["stride"],
            "strides": {},
            "sites": {},
            "spans": 0,
            "edits": 0,
            "unparsed": [],
            "note": "БАЗА: правок в заголовке НЕТ",
        }
    else:
        ov, rep = build_overlay(spec, group, pad)
    write(ovp, json.dumps(ov, ensure_ascii=False, indent=2) + "\n")
    write(
        os.path.join(d, "plan.json"),
        json.dumps(rep, ensure_ascii=False, indent=2) + "\n",
    )
    cmd = [
        sys.executable,
        os.path.join(_HERE, "twin.py"),
        "gen",
        "--overlay",
        ovp,
        "--root",
        spec["prod_root"],
        "--out",
        os.path.join(d, "twin"),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise PadError(
            "ВОРОТА ДРЕЙФА twin.py отвергли вариант pad=%s:\n%s\n%s"
            % (pad, p.stdout[-1500:], p.stderr[-1500:])
        )
    if not quiet:
        print(
            "  pad=%-4s двойник порождён, правок %d, участков %d"
            % ("прод" if pad is None else pad, rep["edits"], rep["spans"])
        )
    return d, rep


# =================================================================================================
# 4. СБОРКА (ТОЛЬКО ПРОЦЕССОР)
# =================================================================================================
def variant_args(spec, group, pad):
    """ОДИН источник истины для аргументов load(): и сборка, и последующая загрузка.

    Мина, которую это закрывает: load() с ДРУГИМ набором флагов перепишет build.ninja, и «просто
    загрузить готовое» превратится в семиминутную пересборку -- посреди замера на карте.
    """
    d = vdir(spec, group, pad)
    tw = os.path.join(vdir(spec, group, None) if pad == "li" else d, "twin")
    root = spec["prod_root"]
    twin_inc = os.path.dirname(os.path.join(tw, spec["target"]))
    prod_inc = os.path.dirname(os.path.join(root, spec["target"]))
    return {
        "name": "pads_%s_%s_%s"
        % (
            spec["name"],
            group,
            "prod" if pad is None else ("prodli" if pad == "li" else "p%02d" % pad),
        ),
        "sources": [os.path.join(tw, spec["carrier"])],
        "build_directory": os.path.join(d, "ext"),
        # ДВОЙНИК ПЕРВЫМ. Кавычечный #include ищется рядом с включающим файлом, и включающий тоже
        # взят из двойника -- но -I ниже страхует остальное, и ворота порядка стоят после сборки.
        "extra_include_paths": [twin_inc]
        + [os.path.join(root, p) for p in spec["include_dirs"]],
        "extra_cuda_cflags": list(spec["nvcc"])
        + (["-lineinfo"] if pad == "li" else [])
        + [
            "-ccbin",
            "/usr/bin/g++",
            "-Wno-deprecated-gpu-targets",
            "-gencode=arch=compute_70,code=sm_70",
            "-Xptxas",
            "-v",
        ],
        "_twin_inc": twin_inc,
        "_prod_inc": prod_inc,
        "_dir": d,
    }


def _build_env():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    os.environ.setdefault("CUDA_HOME", CUDA_HOME)
    os.environ.setdefault("CC", "/usr/bin/gcc")
    os.environ.setdefault("CXX", "/usr/bin/g++")
    os.environ.setdefault("CUDAHOSTCXX", "/usr/bin/g++")


def build_one(spec, group, pad, verbose=False):
    """Собрать ОДИН вариант отдельным расширением. Карту не трогает."""
    va = variant_args(spec, group, pad)
    d, src = va["_dir"], va["sources"][0]
    if not os.path.exists(src):
        raise PadError("двойника нет: %s (сперва `plan`)" % src)
    _build_env()
    from torch.utils.cpp_extension import load

    twin_inc, prod_inc = va["_twin_inc"], va["_prod_inc"]
    name, bdir = va["name"], va["build_directory"]
    os.makedirs(bdir, exist_ok=True)
    # ЗАЛИПШИЙ ЗАМОК (происшествие 01.08.2026, 22:14). torch.utils.cpp_extension держит сборку
    # файлом `lock` (FileBaton) и ЖДЁТ ЕГО ИСЧЕЗНОВЕНИЯ БЕСКОНЕЧНО. Прерванная сборка оставляет
    # файл, и КАЖДАЯ следующая засыпает навсегда -- без единой строки в логе, без traceback, с
    # видом «долго компилируется». Шесть вариантов простояли так 53 минуты.
    # Снимать безопасно ИМЕННО ЗДЕСЬ: каталог сборки принадлежит РОВНО ОДНОМУ варианту, а варианты
    # запускаются по одному на каталог (см. _build_parallel). Чужого замка тут быть не может.
    lk = os.path.join(bdir, "lock")
    if os.path.exists(lk):
        os.remove(lk)
        print("  [ЗАЛИПШИЙ ЗАМОК] снят %s -- иначе сборка ждала бы его вечно" % lk)
    t0 = time.time()
    load(
        name=name,
        sources=va["sources"],
        build_directory=bdir,
        extra_include_paths=va["extra_include_paths"],
        extra_cuda_cflags=va["extra_cuda_cflags"],
        verbose=verbose,
    )
    dt = time.time() - t0
    # ВОРОТА ПОРЯДКА -I. Боевой каталог заголовков РАНЬШЕ двойника == собрано боевое ядро, и
    # дополнение не подействовало НИ НА ЧТО. Молча этого допускать нельзя (мина из bwd_phase_ext.py).
    nj = read(os.path.join(bdir, "build.ninja"))
    i_tw, i_pr = nj.find("-I" + twin_inc), nj.find("-I" + prod_inc)
    if i_tw < 0 or (0 <= i_pr < i_tw):
        raise PadError(
            "[ПОРЯДОК -I] боевой %s стоит РАНЬШЕ двойника -- собрано боевое ядро, "
            "дополнение не подействовало. Вариант pad=%s НЕДЕЙСТВИТЕЛЕН."
            % (prod_inc, pad)
        )
    so = os.path.join(bdir, name + ".so")
    sass = os.path.join(d, "k.sass")
    subprocess.run("%s -sass %s > %s" % (CUOBJDUMP, so, sass), shell=True)
    deps = deps_fingerprint(bdir, twin_inc)
    res = res_usage(so, spec["kernel"])
    if not res.get("n"):
        raise PadError(
            "в собранной .so НЕТ ни одного ядра по регулярке «%s» -- собрано не то, "
            "и любой замер по этому варианту недействителен." % spec["kernel"]
        )
    info = {
        "pad": pad,
        "so": so,
        "build_sec": round(dt, 1),
        "sass_md5": md5f(sass),
        "smem": res.get("smem"),
        "regs": res.get("regs"),
        "stack": res.get("stack"),
        "smem_set": res.get("smem_set"),
        "deps_md5": deps["md5"],
        "deps_n": deps["n"],
        "deps": deps["files"],
        "kernels_matched": res.get("n", 0),
        "at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    write(
        os.path.join(d, "build.json"),
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
    )
    return info


def deps_fingerprint(bdir, twin_inc):
    """ОТПЕЧАТОК ВСЕГО, ЧТО nvcc ПРОЧИТАЛ, кроме файлов самого двойника.

    ПРОИСШЕСТВИЕ, РАДИ КОТОРОГО ЭТО НАПИСАНО (01.08.2026, 22:07). Посреди перебора СОСЕД правил
    боевое дерево: `volta_fwd_ws.cuh` и `gemm/volta_warp_mma.h` уехали между волнами сборок.
    Ранние варианты собрались против одного текста, поздние -- против другого, и КРИВАЯ СРАВНИВАЛА
    БЫ РАЗНЫЕ ЯДРА, не сказав об этом ни слова: у каждого варианта свой pad, разница списывается
    на pad. Ворота дрейфа twin.py закрывают ТОЛЬКО файлы наложения; всё остальное дерево едет
    через -I и не гейтится ничем.

    Список читает ninja-шный .d (его пишет сам nvcc), поэтому покрытие ПОЛНОЕ по построению: там
    ровно те файлы, которые компилятор открыл. Файлы двойника исключены -- они и ДОЛЖНЫ отличаться.
    """
    out, n = [], 0
    for dep in sorted(glob_dfiles(bdir)):
        for path in parse_dfile(read(dep)):
            if path.startswith(twin_inc) or not os.path.exists(path):
                continue
            out.append((path, md5f(path)))
            n += 1
    out.sort()
    h = hashlib.md5(("\n".join("%s %s" % p for p in out)).encode("utf-8")).hexdigest()
    return {"md5": h, "n": n, "files": dict(out)}


def glob_dfiles(bdir):
    return [os.path.join(bdir, f) for f in os.listdir(bdir) if f.endswith(".d")]


def parse_dfile(text):
    text = text.replace("\\\n", " ")
    if ":" in text:
        text = text.split(":", 1)[1]
    return [t for t in text.split() if t not in ("\\",)]


def res_usage(so, kernel_regex):
    """СТАТИЧЕСКАЯ разделяемая (SHARED) и регистры нашего ядра ИЗ ДВОИЧНОГО ФАЙЛА.

    Зачем не парсить `ptxas -v`: его вывод уезжает в лог ninja, который при повторной сборке НЕ
    перезаписывается (ninja считает цель свежей), и число молча остаётся ОТ ПРЕДЫДУЩЕГО варианта --
    ровно тот класс ошибки, который выглядит как правдоподобный ответ.

    ИМЕННО SHARED, а не «вся разделяемая»: динамическую (extern __shared__) выделяет хост, и
    дополнение её не трогает. Значит рост SHARED -- ПРЯМОЕ доказательство, что дополнение
    подействовало на тот самый массив, а не куда-то ещё.
    """
    p = subprocess.run([CUOBJDUMP, "-res-usage", so], capture_output=True, text=True)
    txt = p.stdout or ""
    rx = re.compile(r"Function\s+([^\s:]+)")
    cur, n = None, 0
    smem = regs = stack = None
    smems = set()
    for ln in txt.splitlines():
        m = rx.search(ln)
        if m:
            cur = m.group(1)
            continue
        if cur is None or not re.search(kernel_regex, cur):
            continue
        mm = re.search(r"\bSHARED:\s*(\d+)", ln)
        if mm:
            v = int(mm.group(1))
            smem = max(smem or 0, v)
            smems.add(v)
            n += 1
        mm = re.search(r"\bREG:\s*(\d+)", ln)
        if mm:
            regs = max(regs or 0, int(mm.group(1)))
        mm = re.search(r"\bSTACK:\s*(\d+)", ln)
        if mm:
            stack = max(stack or 0, int(mm.group(1)))
    return {
        "smem": smem,
        "regs": regs,
        "stack": stack,
        "n": n,
        "smem_set": sorted(smems),
    }


# =================================================================================================
# 5. ТЕЛО (ОДИН ПРОГОН) -- именно его профилирует ncu
# =================================================================================================
def run_body(spec, group, pad, save=None):
    # ОКРУЖЕНИЕ -- ДО ПЕРВОГО ИМПОРТА. torch.utils.cpp_extension вычисляет CUDA_HOME ОДИН РАЗ, при
    # своём импорте, а его импортирует уже боевой fa2_sm70/_ext.py. Поставить переменную после --
    # значит получить «CUDA_HOME is not set» на ровном месте (поймано этим же телом).
    _build_env()
    import torch

    sys.path.insert(0, spec["prod_root"])
    import fa2_sm70 as F  # только квантователь (чистый torch)

    va = variant_args(spec, group, pad)
    so = os.path.join(va["build_directory"], va["name"] + ".so")
    if not os.path.exists(so):
        raise PadError("вариант не собран: %s (сперва `build`)" % so)
    from torch.utils.cpp_extension import load

    # ТЕ ЖЕ аргументы, что при сборке (см. variant_args) -- иначе ninja пересоберёт всё заново.
    mod = load(
        name=va["name"],
        sources=va["sources"],
        build_directory=va["build_directory"],
        extra_include_paths=va["extra_include_paths"],
        extra_cuda_cflags=va["extra_cuda_cflags"],
        verbose=False,
    )
    b = spec["body"]
    torch.manual_seed(b.get("seed", 0))
    dev = "cuda"
    B, H, Hkv, Sq, Sk, D = b["B"], b["H"], b["Hkv"], b["Sq"], b["Sk"], b["D"]
    q = torch.randn(B, Sq, H, D, dtype=torch.float16).to(dev)
    k = torch.randn(B, Hkv, Sk, D, dtype=torch.float16).to(dev)
    v = torch.randn(B, Hkv, Sk, D, dtype=torch.float16).to(dev)
    cache = F.quantize_kv_prefill(k, v, fmt=8)
    sc = cache["scales"]
    scale = 1.0 / (D**0.5)
    o, lse = mod.attn_fwd_volta_i8(
        q.contiguous(),
        cache["Kb"],
        sc,
        cache["Vb"],
        sc,
        float(scale),
        bool(b.get("causal", True)),
        8,
    )
    torch.cuda.synchronize()
    dig = hashlib.md5(o.detach().cpu().numpy().tobytes()).hexdigest()
    dl = hashlib.md5(lse.detach().cpu().numpy().tobytes()).hexdigest()
    if save:
        write(
            save,
            json.dumps(
                {
                    "o_md5": dig,
                    "lse_md5": dl,
                    "o_shape": list(o.shape),
                    "finite": bool(torch.isfinite(o).all()),
                },
                ensure_ascii=False,
            )
            + "\n",
        )
    print(
        "ОТВЕТ md5 O=%s LSE=%s  конечен=%s" % (dig, dl, bool(torch.isfinite(o).all()))
    )
    return dig, dl


# =================================================================================================
# 6. ЗАМЕР (КАРТА): побитово -> конфликты -> разделяемая
# =================================================================================================
def gpu_busy(index):
    p = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    return [l.strip() for l in (p.stdout or "").splitlines() if l.strip()]


def measure_one(spec, group, pad, base=None, per_line=False):
    base_md5 = base["o_md5"] if base else None
    d = vdir(spec, group, pad)
    cmd = [
        PY,
        os.path.abspath(__file__),
        "body",
        "--spec",
        spec["_path"],
        "--group",
        group,
        "--pad",
        ("prod" if pad is None else str(pad)),
        "--save",
        os.path.join(d, "answer.json"),
    ]
    env = dict(CUDA_VISIBLE_DEVICES=str(spec.get("device", 1)))
    # 1. ПОБИТОВО -- ПЕРВЫМ. Вариант с другим ответом дальше не рассматривается, каким бы
    #    чистым он ни был по конфликтам: это брак, а не результат.
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, **env))
    if p.returncode != 0:
        blob = (p.stdout or "") + (p.stderr or "")
        rec = {
            "pad": pad,
            "ok": False,
            "why": "прогон упал: " + (p.stderr or "")[-300:],
        }
        # ПОТОЛОК СЕМЕЙСТВА, А НЕ «ОШИБКА». Дополнение растит СТАТИЧЕСКУЮ разделяемую, а
        # ДИНАМИЧЕСКУЮ просит хост -- их сумма упирается в 96 КБ на блок, и запуск отвергается
        # «invalid argument» без единого слова про память. Это НЕ отказ инструмента: это край
        # области, где семейство вообще существует, и назвать его надо именно так.
        if "invalid argument" in blob:
            try:
                st = json.loads(read(os.path.join(d, "build.json")))["smem"]
                rec["why"] = (
                    "ЗАПУСК ОТВЕРГНУТ (invalid argument). Статическая разделяемая этого "
                    "варианта %d Б; вместе с ДИНАМИЧЕСКОЙ, которую просит хост, сумма "
                    "перешла предел %d Б на блок. Это ПОТОЛОК СЕМЕЙСТВА дополнений, а не "
                    "сбой: дальше pad просто не существует." % (st, SMEM_SM)
                )
                rec["ceiling"] = True
            except (OSError, KeyError, ValueError):
                pass
        return rec
    ans = json.loads(read(os.path.join(d, "answer.json")))
    exact = (base_md5 is None) or (ans["o_md5"] == base_md5)
    rec = {
        "pad": pad,
        "ok": True,
        "o_md5": ans["o_md5"],
        "exact": exact,
        "finite": ans["finite"],
    }
    bi = json.loads(read(os.path.join(d, "build.json")))
    rec["smem"] = bi.get("smem")
    rec["regs"] = bi.get("regs")
    rec["sass_md5"] = bi.get("sass_md5")
    rec["deps_md5"] = bi.get("deps_md5")
    # ВОРОТА ОДНОГО ДЕРЕВА: все варианты обязаны быть собраны против ОДНОГО И ТОГО ЖЕ боевого
    # текста. Иначе кривая сравнивает разные ядра и списывает разницу на pad (происшествие 22:07).
    if base is not None:
        bd = json.loads(read(os.path.join(vdir(spec, group, None), "build.json")))
        if not rec["deps_md5"] or not bd.get("deps_md5"):
            rec["deps_note"] = (
                "ОТПЕЧАТКА ЗАВИСИМОСТЕЙ НЕТ (вариант собран до появления этих "
                "ворот) -- совпадение боевого текста между вариантами НЕ ДОКАЗАНО"
            )
        elif rec["deps_md5"] != bd["deps_md5"]:
            drift = [
                f
                for f, h in (bi.get("deps") or {}).items()
                if h != (bd.get("deps") or {}).get(f)
            ]
            rec["ok"] = False
            rec["why"] = (
                "БОЕВОЕ ДЕРЕВО УЕХАЛО МЕЖДУ СБОРКАМИ: этот вариант собран против другого "
                "текста, чем база. Расходятся %d файл(ов): %s. Сравнивать нельзя."
                % (len(drift), ", ".join(os.path.basename(f) for f in drift[:4]))
            )
            return rec
    # ВОРОТА «ДОПОЛНЕНИЕ ПОДЕЙСТВОВАЛО». Собранный не тот вариант даёт РОВНО ТОТ ЖЕ счётчик и
    # читается как «дополнение не помогает» -- правдоподобный неверный ответ. Статическая
    # разделяемая обязана вырасти при pad>0 и НЕ вырасти при pad==0.
    if base is not None and rec["smem"] is not None and base.get("smem") is not None:
        d_smem = rec["smem"] - base["smem"]
        rec["d_smem"] = d_smem
        if pad and d_smem <= 0:
            rec["ok"] = False
            rec["why"] = (
                "РАЗДЕЛЯЕМАЯ НЕ ВЫРОСЛА (%+d Б) при pad=%d -- значит собрано НЕ ТО ядро "
                "либо дополнение не попало в раскладку. Замер недействителен."
                % (d_smem, pad)
            )
            return rec
        if pad == 0 and d_smem != 0:
            rec["ok"] = False
            rec["why"] = (
                "pad=0 изменил разделяемую на %+d Б -- `(шаг + 0)` обязан быть "
                "тождественным; разбор шага неверен." % d_smem
            )
            return rec
    if not exact:
        rec["why"] = (
            "ОТВЕТ ИЗМЕНИЛСЯ -- дополнение не тождественно по значению; конфликты НЕ мерим"
        )
        return rec
    # 2. КОНФЛИКТЫ
    r = ncu.conflicts(
        spec["kernel"],
        cmd,
        workload_env=env,
        per_line=per_line,
        keep_report=per_line,
        do_warmup=True,
    )
    rec.update(
        wavefronts=r.wavefronts,
        conflicts=r.conflicts,
        fraction=r.fraction,
        launches=r.launches,
        unparsed=[list(u) for u in r.unparsed],
    )
    if per_line:
        rec["by_line"] = r.by_line[:25]
    return rec


# =================================================================================================
# 7. КРИВАЯ И ЕЁ ПЕРИОД
# =================================================================================================
def period_of(pads, vals, tol=0.02):
    """Наименьший период кривой pad -> конфликты. Это и есть ШАГ СТОЛБЦА ИТЕРАТОРА.

    Проверяем ТОЛЬКО на сплошном участке (иначе период «находится» на дырках). Допуск tol --
    относительный: счётчик детерминирован, но между вариантами меняется расписание.
    """
    if len(pads) < 6 or pads != list(range(pads[0], pads[0] + len(pads))):
        return None, "кривая не сплошная -- период не проверяется"
    # ГРАНИЦА ПОИСКА -- НЕ n/2. Требование «половина длины» ловит только периоды, укладывающиеся
    # дважды, и на 13 точках ОБЪЯВЛЯЕТ ОТСУТСТВИЕ периода 8, который там виден пятью парами.
    # Достаточное свидетельство -- kMinPairs подтверждающих пар; отсюда p <= n - kMinPairs.
    kMinPairs = 3
    n = len(pads)
    for p in range(1, max(1, n - kMinPairs) + 1):
        pairs, ok = 0, True
        for i in range(n - p):
            a, b = vals[i], vals[i + p]
            if max(a, b) <= 0:
                continue
            if abs(a - b) / max(abs(a), abs(b)) > tol:
                ok = False
                break
            pairs += 1
        if ok and pairs >= kMinPairs:
            return p, "%d подтверждающих пар из %d точек, допуск %.0f %%" % (
                pairs,
                n,
                100 * tol,
            )
    return None, (
        "период <= %d не подтверждается (для периода p нужно хотя бы %d пар, то есть "
        "%d точек)" % (max(1, n - kMinPairs), kMinPairs, kMinPairs + 1)
    )


# =================================================================================================
# 8. КОМАНДЫ
# =================================================================================================
def load_spec(path):
    s = json.loads(read(path))
    s["_path"] = os.path.abspath(path)
    s.setdefault("work", DEFAULT_WORK)
    s.setdefault(
        "include_dirs",
        [
            "fa2_sm70/csrc",
            "fa2_src/cutlass/include",
            "fa2_src/cutlass/tools/util/include",
            "fa2_src/fmha_kernel",
        ],
    )
    s.setdefault(
        "nvcc",
        [
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-DHAS_PYTORCH",
        ],
    )
    s.setdefault("device", 1)
    return s


def parse_pads(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def print_unparsed(items, title="НЕРАЗОБРАННОЕ"):
    print("\n%s (%d):" % (title, len(items)))
    if not items:
        print(
            "  -- пусто. ВНИМАНИЕ: пустой список найденного при пустом неразобранном означает "
            "«чисто»; при НЕпустом -- не означает ничего."
        )
        return
    for a, b in items:
        print("  %-46s %s" % (a[:46], b))


def cost_note(spec, group, pads, sec_per_build=None):
    n = len(pads) + 1
    sec = sec_per_build or spec.get("sec_per_build", 480)
    jobs = spec.get("jobs", 4)
    print("СТОИМОСТЬ ПЕРЕБОРА (говорим ЗАРАНЕЕ, а не после):")
    print(
        "  вариантов            %d  (pad %s + БАЗА)"
        % (n, "%d..%d" % (pads[0], pads[-1]) if pads else "-")
    )
    print(
        "  сборок               %d, по ~%d с каждая -> %s в один поток, %s при --jobs %d"
        % (n, sec, hms(n * sec), hms(n * sec / max(1, jobs)), jobs)
    )
    print(
        "  прогонов на карте    %d (побитово) + %d (ncu, по 2 запуска) = %d коротких запусков"
        % (n, n, 3 * n)
    )
    print(
        "  ЗАМЕРОВ ВРЕМЕНИ      0 -- счётчик конфликтов от соседа по карте не зависит, время зависит"
    )


def cmd_plan(spec, args):
    group = args.group
    pads = parse_pads(args.pads)
    if args.dry:
        cost_note(spec, group, pads)
        return 0
    print(
        "ПОРОЖДЕНИЕ ВАРИАНТОВ: %s / группа «%s», массивы %s"
        % (spec["name"], group, ", ".join(spec["groups"][group]["arrays"]))
    )
    d, rep = gen_twin(spec, group, None)
    print("  ШАГ РАСКЛАДКИ ОПОЗНАН КАК: %s" % rep["stride"])
    _, rep0 = gen_twin(spec, group, 0, quiet=True)
    print(
        "  обоснование выбора шага (кандидат: сколько индексных выражений его содержит / "
        "сколько раз он стоит сразу за '*'):"
    )
    for nm, table in rep0["strides"].items():
        for c, s, t in table[:4]:
            print("    %-8s %-10s %d / %d" % (nm, t, c, s))
    for pad in pads:
        gen_twin(spec, group, pad, quiet=(pad != pads[0]))
    print(
        "  ... порождено %d вариантов -> %s"
        % (len(pads) + 1, os.path.join(spec["work"], spec["name"], group))
    )
    print_unparsed(rep0["unparsed"])
    cost_note(spec, group, pads)
    return 0


def cmd_build(spec, args):
    group = args.group
    pads = parse_pads(args.pads)
    todo = [None] + pads
    if args.dry:
        cost_note(spec, group, pads)
        return 0
    if args.jobs > 1:
        return _build_parallel(spec, group, todo, args)
    t0 = time.time()
    for i, pad in enumerate(todo):
        try:
            info = build_one(spec, group, pad, verbose=args.verbose)
            print(
                "  [%2d/%2d] pad=%-4s %5.0f с  smem=%s Б  рег=%s  sass_md5=%s"
                % (
                    i + 1,
                    len(todo),
                    "прод" if pad is None else pad,
                    info["build_sec"],
                    info["smem"],
                    info["regs"],
                    info["sass_md5"][:8],
                )
            )
        except Exception as ex:
            print(
                "  [%2d/%2d] pad=%-4s ОТКАЗ: %s"
                % (i + 1, len(todo), "прод" if pad is None else pad, str(ex)[:300])
            )
    print("СБОРОК %d, всего %s" % (len(todo), hms(time.time() - t0)))
    return 0


def _build_parallel(spec, group, todo, args):
    """Сборки независимы -- гоняем их пачками СВОИМИ процессами. Карта не участвует."""
    t0, running, done = time.time(), [], []
    queue = list(todo)
    while queue or running:
        while queue and len(running) < args.jobs:
            pad = queue.pop(0)
            cmd = [
                PY,
                os.path.abspath(__file__),
                "build",
                "--spec",
                spec["_path"],
                "--group",
                group,
                "--pads",
                ("" if pad is None else str(pad)),
                "--jobs",
                "1",
                "--only-one",
                ("prod" if pad is None else str(pad)),
            ]
            running.append(
                (
                    pad,
                    subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                    ),
                )
            )
        time.sleep(2)
        for pad, p in list(running):
            if p.poll() is not None:
                out = p.stdout.read()
                running.remove((pad, p))
                done.append(pad)
                tag = "прод" if pad is None else pad
                line = [l for l in out.splitlines() if "pad=" in l]
                print(
                    "  [%2d/%2d] %s"
                    % (
                        len(done),
                        len(todo),
                        line[-1].strip()
                        if line
                        else "pad=%s rc=%d %s" % (tag, p.returncode, out[-200:]),
                    )
                )
    print(
        "СБОРОК %d, всего %s (в %d потоков)"
        % (len(todo), hms(time.time() - t0), args.jobs)
    )
    return 0


def cmd_run(spec, args):
    group = args.group
    pads = parse_pads(args.pads)
    if args.dry:
        cost_note(spec, group, pads)
        return 0
    dev = spec.get("device", 1)
    busy = gpu_busy(dev)
    if busy and not args.force:
        print(
            "КАРТА %d ЗАНЯТА (pid %s). Замер НЕ НАЧАТ: чужой процесс на карте не портит счётчик\n"
            "конфликтов, но портит ЧУЖОЙ замер -- ncu сериализует запуски. Ждите или --force."
            % (dev, ", ".join(busy))
        )
        return 4
    base = measure_one(spec, group, None, base=None, per_line=args.per_line)
    if not base.get("ok"):
        print("БАЗА НЕ ИЗМЕРЕНА: %s" % base.get("why"))
        return 3
    print(
        "БАЗА (боевой исходник): конфл %.6g / вайвфр %.6g = %.2f %%, smem %s Б, ответ md5 %s"
        % (
            base["conflicts"],
            base["wavefronts"],
            100 * base["fraction"],
            base["smem"],
            base["o_md5"][:12],
        )
    )
    # ДОБИРАЕМ, А НЕ ЗАТИРАЕМ. Перебор идёт кусками (карта общая, её отдают и забирают), и запуск
    # `run --pads 13-31` не должен стирать уже снятые 0..12: переснять их стоит час.
    swp = os.path.join(spec["work"], spec["name"], group, "sweep.json")
    rows = [base]
    if os.path.exists(swp):
        for r in json.loads(read(swp)):
            if r.get("pad") is not None and r["pad"] not in pads:
                rows.append(r)
    best, since = base["fraction"], 0
    for pad in pads:
        r = measure_one(spec, group, pad, base=base, per_line=args.per_line)
        rows.append(r)
        if not r.get("ok"):
            print("  pad=%-3d ОТКАЗ: %s" % (pad, r.get("why", "")[:160]))
        elif not r["exact"]:
            print("  pad=%-3d ПОБИТОВО: НЕТ  -- ОТБРОШЕН (%s)" % (pad, r["why"]))
        else:
            gain = r["smem"] - base["smem"] if (r["smem"] and base["smem"]) else None
            print(
                "  pad=%-3d ПОБИТОВО: да   конфл %10.6g (%5.2f %%)  smem %s Б (%+d)  %s"
                % (
                    pad,
                    r["conflicts"],
                    100 * r["fraction"],
                    r["smem"],
                    gain or 0,
                    "SASS==база" if r["sass_md5"] == base["sass_md5"] else "",
                )
            )
            if r["fraction"] < best - 1e-6:
                best, since = r["fraction"], 0
            else:
                since += 1
        rows[1:] = sorted(rows[1:], key=lambda r: r["pad"])
        write(swp, json.dumps(rows, ensure_ascii=False, indent=1) + "\n")
        if args.patience and since >= args.patience:
            print(
                "\nРАННЯЯ ОСТАНОВКА: %d точек подряд без улучшения минимума.\n"
                "  ПЕРЕБОР НЕПОЛОН -- пройдено %d из %d значений; кривая ОБОРВАНА, и её ПЕРИОД\n"
                "  (шаг столбца итератора) по обрывку определять НЕЛЬЗЯ. Полный проход: --patience 0."
                % (args.patience, len(rows) - 1, len(pads))
            )
            break
    return cmd_report(spec, args)


def cmd_report(spec, args):
    group = args.group
    path = os.path.join(spec["work"], spec["name"], group, "sweep.json")
    if not os.path.exists(path):
        print("замеров нет: %s" % path)
        return 2
    rows = json.loads(read(path))
    base = rows[0]
    ok = [r for r in rows[1:] if r.get("ok") and r.get("exact")]
    print(
        "\nКРИВАЯ pad -> конфликты (%s / %s), массивы %s, шаг %s"
        % (
            spec["name"],
            group,
            ", ".join(spec["groups"][group]["arrays"]),
            json.loads(
                read(
                    os.path.join(
                        spec["work"], spec["name"], group, "pad00", "plan.json"
                    )
                )
            )["stride"]
            if os.path.exists(
                os.path.join(spec["work"], spec["name"], group, "pad00", "plan.json")
            )
            else "?",
        )
    )
    print(
        "  %-5s %12s %8s %10s %8s %s"
        % ("pad", "конфликты", "доля", "вайвфронты", "smem Б", "побитово")
    )
    print("  " + "-" * 62)
    print(
        "  %-5s %12.6g %7.2f%% %10.6g %8s %s"
        % (
            "база",
            base["conflicts"],
            100 * base["fraction"],
            base["wavefronts"],
            base["smem"],
            "эталон",
        )
    )
    for r in rows[1:]:
        if not r.get("ok"):
            print("  %-5d %12s" % (r["pad"], "ОТКАЗ"))
        elif not r["exact"]:
            print(
                "  %-5d %12s %8s %10s %8s %s"
                % (r["pad"], "-", "-", "-", r.get("smem"), "НЕТ -- отброшен")
            )
        else:
            print(
                "  %-5d %12.6g %7.2f%% %10.6g %8s %s"
                % (
                    r["pad"],
                    r["conflicts"],
                    100 * r["fraction"],
                    r["wavefronts"],
                    r["smem"],
                    "да",
                )
            )
    if not ok:
        print("\nНИ ОДНОГО побитово точного варианта -- сообщать нечего.")
        return 0
    b = min(ok, key=lambda r: r["fraction"])
    print(
        "\nМИНИМУМ: pad=%d -> %.2f %% (база %.2f %%), это x%.2f по трафику конфликтов;"
        % (
            b["pad"],
            100 * b["fraction"],
            100 * base["fraction"],
            base["conflicts"] / b["conflicts"] if b["conflicts"] else float("inf"),
        )
    )
    print(
        "  разделяемая %s -> %s Б (%+d, предел на CTA %d, вся на SM %d)"
        % (
            base["smem"],
            b["smem"],
            (b["smem"] or 0) - (base["smem"] or 0),
            SMEM_CTA_DEFAULT,
            SMEM_SM,
        )
    )
    # МИНИМУМ И ВЫБОР -- РАЗНЫЕ ВЕЩИ. Кривая периодична, поэтому у минимума обычно есть
    # НЕОТЛИЧИМЫЕ по счётчику братья, и среди них надо брать САМЫЙ ДЕШЁВЫЙ по разделяемой:
    # лишние байты режут число блоков на SM, а этого счётчик конфликтов не видит вовсе.
    tol = 0.01 * b["fraction"]
    tie = min(
        (r for r in ok if r["fraction"] <= b["fraction"] + tol),
        key=lambda r: (r["smem"], r["pad"]),
    )
    if tie["pad"] != b["pad"]:
        print(
            "  БРАТЬ НАДО pad=%d: %.2f %% (от минимума в пределах 1 %% относительных), но "
            "разделяемая\n  всего %+d Б вместо %+d -- у периодичной кривой минимум и ВЫБОР это "
            "разные вещи."
            % (
                tie["pad"],
                100 * tie["fraction"],
                tie["smem"] - base["smem"],
                (b["smem"] or 0) - (base["smem"] or 0),
            )
        )
    ceil_rows = [r for r in rows[1:] if r.get("ceiling")]
    if ceil_rows:
        first = min(r["pad"] for r in ceil_rows)
        last_ok = max((r["pad"] for r in ok), default=None)
        dyn = SMEM_SM - (max((r["smem"] for r in ok if r["pad"] == last_ok), default=0))
        print(
            "\nПОТОЛОК СЕМЕЙСТВА: pad <= %s. С pad=%d запуск отвергается -- статическая\n"
            "  разделяемая плюс динамическая перебирают %d Б на блок. Отсюда ВЫВОДИТСЯ (а не\n"
            "  берётся из исходника) динамическая часть: не больше %d Б, то есть свободного\n"
            "  места на дополнение было %d слов, и перебор их ИСЧЕРПАЛ."
            % (last_ok, first, SMEM_SM, dyn, last_ok)
        )
    pads = [r["pad"] for r in ok]
    vals = [r["conflicts"] for r in ok]
    p, why = period_of(pads, vals)
    if p:
        print(
            "\nПЕРИОД КРИВОЙ = %d слов(а) (%s).\n"
            "  ЭТО И ЕСТЬ ШАГ СТОЛБЦА ИТЕРАТОРА ДОСТУПА -- та величина, которую из исходника не\n"
            "  вывести. Резонанс шага раскладки с ней и порождал конфликты; перебор её ИЗМЕРИЛ."
            % (p, why)
        )
    else:
        print(
            "\nПЕРИОД НЕ ОПРЕДЕЛЁН (%s). Это не значит «его нет»: кривая могла быть оборвана\n"
            "  ранней остановкой либо смешивать два обращения с разными шагами столбца."
            % why
        )
    un = []
    for r in rows:
        for u in r.get("unparsed", []):
            un.append(tuple(u))
    print_unparsed(sorted(set(un)))
    return 0


def cmd_find(spec, args):
    """Шаг 1: кандидаты. ncu построчно -> ранжирование по ИЗБЫТОЧНЫМ вайвфронтам -> раскладки."""
    if args.dry:
        print(
            "СТОИМОСТЬ: 1 сборка (если варианта «прод» ещё нет) + 3 коротких запуска на карте."
        )
        return 0
    dev = spec.get("device", 1)
    if gpu_busy(dev) and not args.force:
        print(
            "КАРТА %d ЗАНЯТА -- поиск кандидатов требует карты. Ждите или --force."
            % dev
        )
        return 4
    group = args.group or sorted(spec["groups"])[0]
    if not os.path.exists(os.path.join(vdir(spec, group, None), "build.json")):
        gen_twin(spec, group, None)
    # ПОСТРОЧНАЯ ПРИВЯЗКА ТРЕБУЕТ -lineinfo. Без него ncu отдаёт адреса SASS, наш разбор честно
    # говорит «не разобрано», и список кандидатов выходит ПУСТЫМ -- а пустой список читается как
    # «конфликтов нет». Поэтому поиск собирает СВОЙ вариант базы, не трогая перебор.
    d = vdir(spec, group, "li")
    if not os.path.exists(os.path.join(d, "build.json")):
        print("собираю отдельную БАЗУ С -lineinfo (перебор не трогается) ...")
        build_one(spec, group, "li")
    cmd = [
        PY,
        os.path.abspath(__file__),
        "body",
        "--spec",
        spec["_path"],
        "--group",
        group,
        "--pad",
        "li",
        "--save",
        os.path.join(d, "answer.json"),
    ]
    r = ncu.conflicts(
        spec["kernel"],
        cmd,
        workload_env={"CUDA_VISIBLE_DEVICES": str(dev)},
        per_line=True,
    )
    print(ncu.report_text(r, top=args.top))
    text = read(os.path.join(spec["prod_root"], spec["target"]))
    lines = text.splitlines()
    decls = {}
    for m in re.finditer(r"__shared__\s+\w+\s+(%s)\s*\[" % IDENT, text):
        decls[m.group(1)] = line_of(text, m.start()) + 1
    # каждое имя массива -> строки, где оно индексируется
    by_name = collections.defaultdict(float)
    indirect = {}
    unknown = []
    base = os.path.basename(spec["target"])
    for row in r.by_line:
        key = row["key"]
        if not key.startswith(base):
            unknown.append(
                (
                    key,
                    "строка не из %s: %.6g избыточных вайвфронтов" % (base, row["exc"]),
                )
            )
            continue
        try:
            ln = int(key.split(":")[1])
        except (IndexError, ValueError):
            unknown.append((key, "не разобрал номер строки"))
            continue
        src = lines[ln - 1] if 0 < ln <= len(lines) else ""
        hit = [n for n in decls if re.search(r"\b%s\s*\[" % re.escape(n), src)]
        near = 0
        if not hit:
            # ПРИВЯЗКА ПО СОСЕДСТВУ. -lineinfo указывает на строку, куда ПЛАНИРОВЩИК поставил
            # команду, а не на ту, где обращение написано: замерено, что 9216 избыточных
            # вайвфронтов sRedS приезжают на строку с __shfl_xor_sync. Поэтому если на самой
            # строке раскладки нет -- смотрим окно +-kNear строк и ПОМЕЧАЕМ такую привязку как
            # косвенную (она слабее прямой и в отчёте идёт со звёздочкой).
            kNear = args.near
            for off in range(1, kNear + 1):
                for j in (ln - 1 - off, ln - 1 + off):
                    if 0 <= j < len(lines):
                        hit = [
                            n
                            for n in decls
                            if re.search(r"\b%s\s*\[" % re.escape(n), lines[j])
                        ]
                        if hit:
                            near = off
                            break
                if hit:
                    break
        if not hit:
            unknown.append(
                (
                    "%s стр.%d" % (base, ln),
                    "%.6g избыточных вайвфронтов, но обращения к ОБЪЯВЛЕННОЙ раскладке нет "
                    "ни в строке, ни в +-%d строк: %s"
                    % (row["exc"], args.near, src.strip()[:60]),
                )
            )
            continue
        for n in hit:
            by_name[n] += row["exc"] / len(hit)
            if near:
                indirect[n] = indirect.get(n, 0.0) + row["exc"] / len(hit)
    print(
        "\nКАНДИДАТЫ (раскладка -> ИЗБЫТОЧНЫЕ вайвфронты, то есть что снимает дополнение):"
    )
    tot = sum(by_name.values()) or 1.0
    for n, v in sorted(by_name.items(), key=lambda x: -x[1]):
        ind = indirect.get(n, 0.0)
        print(
            "  %-12s стр.%-6d %12.6g  %5.1f %% всего избыточного%s"
            % (
                n,
                decls[n],
                v,
                100 * v / tot,
                "   (* %.0f %% из них привязаны ПО СОСЕДСТВУ, а не по строке)"
                % (100 * ind / v)
                if ind
                else "",
            )
        )
    print(
        "  (*) косвенная привязка: -lineinfo указывает строку, куда команду поставил ПЛАНИРОВЩИК."
    )
    print_unparsed(unknown + [tuple(u) for u in r.unparsed])
    return 0


SELFTEST_SRC = """\
__global__ void k() {
  __shared__ float sR[BQ * WN];
  __shared__ float sT[BQ * WN * 2];
  for (int w = 0; w < WN; ++w) { float y = sR[(row) * WN + w]; use(y); }
  sR[(row) * WN + wn] = x;
  const int idx = (row) * WN + wn;
  sT[idx] = x; sT[BQ * WN + idx] = y;
  const int idx = (row) * WN + w;
  ss += sT[idx]; vv += sT[BQ * WN + idx];
  float other = sQ[r * LDQ + c];
}
"""


def selftest():
    """ЛОВУШКИ, на которых механическая замена даёт ТИХО НЕВЕРНОЕ ядро. Ноль GPU, ноль сборок."""
    bad = 0
    sites, un = find_sites(SELFTEST_SRC, "sR")
    st, table = pick_stride(SELFTEST_SRC, sites)
    print(
        "1. шаг опознан как %r (ожидалось 'WN'), кандидаты %s"
        % (st, [t for _, _, t in table])
    )
    bad += st != "WN"
    spans = [s.span for s in sites if s.span[1] > s.span[0]]
    new = rewrite_text(SELFTEST_SRC, sorted(set(spans)), "WN", 3)
    # ЛОВУШКА 1: граница цикла на ТОЙ ЖЕ СТРОКЕ, что и обращение. Дополнить её -- читать мусор.
    ok1 = "for (int w = 0; w < WN; ++w)" in new
    print(
        "2. граница цикла `w < WN` НЕ дополнена: %s"
        % ("да" if ok1 else "НЕТ -- ЯДРО БЫЛО БЫ НЕВЕРНЫМ")
    )
    bad += not ok1
    # ЛОВУШКА 2: чужая раскладка со своим шагом не тронута
    ok2 = "sQ[r * LDQ + c]" in new
    print(
        "3. чужая раскладка sQ[r * LDQ + c] не тронута: %s" % ("да" if ok2 else "НЕТ")
    )
    bad += not ok2
    # ЛОВУШКА 3: индекс, приехавший ПЕРЕМЕННОЙ -- шаг живёт в её определении, а не в скобках
    sT_sites, un2 = find_sites(SELFTEST_SRC, "sT")
    spans2 = [s.span for s in sT_sites if s.span[1] > s.span[0]]
    new2 = rewrite_text(SELFTEST_SRC, sorted(set(spans2)), "WN", 3)
    ok3 = (
        new2.count("const int idx = (row) * (WN + 3)") == 2
        and new2.count("sT[BQ * (WN + 3) + idx]") == 2
    )
    print(
        "4. индекс через переменную: оба определения idx дополнены и обе половины сдвинуты: %s"
        % ("да" if ok3 else "НЕТ")
    )
    bad += not ok3
    # ЛОВУШКА 4: половинное смещение массива обязано вырасти ВМЕСТЕ с шагом
    ok4 = "__shared__ float sT[BQ * (WN + 3) * 2];" in new2
    print("5. размер двухполовинного массива вырос: %s" % ("да" if ok4 else "НЕТ"))
    bad += not ok4
    # ЛОВУШКА 5: якорь-дубликат. Строка `ss += sT[idx]...` уникальна, а вот две строки `const int
    # idx = ...` различаются только хвостом -- проверяем, что якоря вышли ОДНОЗНАЧНЫМИ.
    edits, badspans = spans_to_edits(
        SELFTEST_SRC, new2, sorted(set(spans2)), "sT", "WN", 3
    )
    dup = [e for e in edits if SELFTEST_SRC.count(e["anchor"]) != 1]
    print(
        "6. все якоря однозначны: %s (правок %d, непостроенных %d)"
        % ("да" if not dup else "НЕТ", len(edits), len(badspans))
    )
    bad += bool(dup)
    print_unparsed([tuple(u) for u in un + un2])
    print("\nСАМОПРОВЕРКА: %s" % ("ПРОЙДЕНА" if not bad else "ПРОВАЛЕНА (%d)" % bad))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(
        description="перебиратель дополнений раскладки (ищем МИНИМУМ)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c, fn in (
        ("plan", cmd_plan),
        ("build", cmd_build),
        ("run", cmd_run),
        ("report", cmd_report),
        ("find", cmd_find),
    ):
        p = sub.add_parser(c)
        p.add_argument("--spec", required=True)
        p.add_argument("--group", default=None if c == "find" else "")
        p.add_argument("--pads", default="0-31")
        p.add_argument(
            "--dry", action="store_true", help="только СТОИМОСТЬ, ничего не делать"
        )
        p.add_argument("--jobs", type=int, default=4)
        p.add_argument(
            "--only-one", default=None, help="служебное: собрать РОВНО этот вариант"
        )
        p.add_argument(
            "--patience",
            type=int,
            default=12,
            help="ранняя остановка после N точек без улучшения; 0 = полный проход "
            "(для ПЕРИОДА кривой нужен полный)",
        )
        p.add_argument("--per-line", action="store_true")
        p.add_argument(
            "--force", action="store_true", help="мерить даже на занятой карте"
        )
        p.add_argument("--verbose", action="store_true")
        p.add_argument("--top", type=int, default=25)
        p.add_argument(
            "--near",
            type=int,
            default=3,
            help="окно строк для КОСВЕННОЙ привязки обращения к раскладке",
        )
        p.set_defaults(fn=fn)
    p = sub.add_parser("body", help="ОДИН прогон варианта -- это и профилирует ncu")
    p.add_argument("--spec", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--pad", required=True)
    p.add_argument("--save", default=None)
    p.set_defaults(fn=None)
    sub.add_parser("selftest", help="ловушки разбора раскладки: ноль GPU, ноль сборок")

    args = ap.parse_args()
    if args.cmd == "selftest":
        return selftest()
    spec = load_spec(args.spec)
    if args.cmd == "body":
        pad = (
            None
            if args.pad == "prod"
            else ("li" if args.pad == "li" else int(args.pad))
        )
        run_body(spec, args.group, pad, args.save)
        return 0
    if args.cmd == "build" and args.only_one is not None:
        pad = None if args.only_one == "prod" else int(args.only_one)
        try:
            info = build_one(spec, args.group, pad, verbose=args.verbose)
            print(
                "  pad=%-4s %5.0f с  smem=%s Б  рег=%s  sass_md5=%s"
                % (
                    "прод" if pad is None else pad,
                    info["build_sec"],
                    info["smem"],
                    info["regs"],
                    info["sass_md5"][:8],
                )
            )
            return 0
        except Exception as ex:
            print(
                "  pad=%-4s ОТКАЗ: %s" % ("прод" if pad is None else pad, str(ex)[:400])
            )
            return 1
    if not args.group and args.cmd != "find":
        ap.error("--group обязателен (группы: %s)" % ", ".join(sorted(spec["groups"])))
    try:
        return args.fn(spec, args)
    except PadError as ex:
        print("\n" + str(ex) + "\n", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())


# =================================================================================================
# СЛЕПЫЕ ЗОНЫ (то, чего инструмент НЕ ВИДИТ -- читать до того, как поверить его вердикту)
# =================================================================================================
# 1. МЕРИТ ТРАФИК, А НЕ ВРЕМЯ. Доля конфликтов -- ВЕРХНЯЯ оценка выигрыша по трафику разделяемой
#    памяти. Во время она переходит настолько, насколько разделяемая связывает. Ядро на 61 % своего
#    счётного потолка может не сдвинуться вовсе. Время меряет владелец на свободной карте.
# 2. ДОПОЛНЕНИЕ -- НЕ ЕДИНСТВЕННОЕ СЕМЕЙСТВО. Перестановка (амплитудой 64 Б на volta_fwd_ws дала
#    5.87 % против 15.59 % у дополнения) в этот перебор НЕ ВХОДИТ: она не однопараметрична.
#    Минимум по pad -- минимум ПО СВОЕМУ СЕМЕЙСТВУ, а не глобальный.
# 3. ПОБИТОВОЕ РАВЕНСТВО ПРОВЕРЯЕТСЯ НА ОДНОЙ ФОРМЕ. Дополнение шага не может изменить значение
#    ни на какой форме, поэтому одной хватает как СТРАЖА; но страж этот -- симметричный (оба плеча
#    читают один и тот же путь). Чужой путь (torch/SDPA) сюда не подмешан НАМЕРЕННО: он дал бы
#    другое значение и по причине, не связанной с раскладкой.
# 4. ЗАНЯТОСТЬ. Рост разделяемой памяти может срезать число блоков на SM. Инструмент печатает
#    smem и пределы, но НЕ считает занятость -- она зависит ещё от регистров и от размера блока.
# 5. РАССЛОЕНИЕ ПО ЗАПУСКАМ. Если под регулярку ядра попадает несколько запусков, счётчики
#    СЛОЖЕНЫ (так делает ncu.py, и он об этом говорит в r.unparsed).
# 6. ШАГ, ВЫРАЖЕННЫЙ НЕ ИДЕНТИФИКАТОРОМ. Раскладка с числовым шагом (`[i * 130 + j]`) или со
#    swizzle/XOR-адресацией сюда не попадает -- она уйдёт в НЕРАЗОБРАННОЕ, и это НЕ значит «чисто».
