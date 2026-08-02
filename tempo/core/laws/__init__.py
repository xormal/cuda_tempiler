#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""РЕЕСТР ЗАКОНОВ -- загрузчик и проверки.  Архитектурно СЛЕПОЙ.

ЗАЧЕМ ОН ЕСТЬ.  Открытие, записанное только в отчёте, протухает за неделю: его невозможно ни
найти в точке применения, ни опровергнуть, ни заметить, что оно уже опровергнуто.  Реестр --
обобщение уже работающей дисциплины `Rate` (число + единица + статус + происхождение) с ЧИСЛА
на УТВЕРЖДЕНИЕ.  Ровно те же правила: отказ загрузчиком, а не предупреждение.

ЕДИНИЦА РЕЕСТРА -- ЗАПИСЬ ЗАКОНА:

    id ........... устойчивый идентификатор, по нему ссылается КОД (маркер `LAW=<id>`)
    kind ......... ЗАКОН | ОПРОВЕРЖЕНИЕ | ДОЛГ
    statement .... формулировка одной фразой
    anchor ....... ЧЕМ доказано: отчёт, раздел, ЧИСЛА, прибор, карта, статус, СМЕЩЕНИЕ замера
    falsifier .... ЧТО обязано случиться, чтобы закон пал, КАК это проверить и во что обойдётся
    scope ........ ОБЛАСТЬ ДЕЙСТВИЯ, включая явное `not`
    home ......... ГДЕ он живёт в продукте: путь + символ.  Только .md => это ДОЛГ, а не закон
    supersedes / superseded_by ... вытеснение вместо удаления

ДВА ПОЛЯ, КОТОРЫХ НЕТ В ОБЫЧНОМ «СПИСКЕ ВЫВОДОВ», И ОБА НЕСУЩИЕ:
  * `anchor.bias` -- В КАКУЮ СТОРОНУ ОШИБАЕТСЯ ЗАМЕР.  Без него фальсификатор нечестен:
    доля полосы, посчитанная при ОДНОКРАТНОМ чтении весов, занижена, и ошибка идёт в сторону
    вывода «чтение не связывает» -- значит найденное чтение реально, а не артефакт.
  * `superseded_by` -- опровергнутая запись НЕ УДАЛЯЕТСЯ.  Отрицательное знание теряется
    первым, поэтому хранится с наибольшей защитой: удалить «int8 = x1.00» значит потерять и
    его ОБЛАСТЬ (счётный угол), и урок (перенос вердикта за пределы области).

РАЗМЕЩЕНИЕ.  Законы МАШИНЫ -- в каталоге плагина (`<плагин>/data/laws/`), законы ИНСТРУМЕНТА
и МЕТОДА -- в корневом `data/laws/`.  Иначе имя архитектуры протечёт в конвейер и лексический
гейт упадёт справедливо.  Этот модуль имён архитектур не знает: он берёт их из ИМЕНИ КАТАЛОГА.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

SCHEMA = "tempo/law/1"
KINDS = ("ЗАКОН", "ОПРОВЕРЖЕНИЕ", "ДОЛГ")
# СТАТУС ЯКОРЯ.  Шире, чем у `Rate`, и намеренно: у закона бывает доказательство, которого у
# числа не бывает, и бывает точный счёт по машинному коду, для которого карта не нужна вовсе.
#   MEASURED    -- секундомер; КАРТА ОБЯЗАТЕЛЬНА (и в ней 0 чужих процессов);
#   SASS        -- точный счёт по машинному коду: воспроизводим без карты, спорить не о чем;
#   PROVEN      -- математическое доказательство; фальсификатор КОНСТРУКТИВНЫЙ, а не замерный;
#   MODEL       -- следствие модели, не наблюдение;
#   UPPER_BOUND -- оценка сверху (в выборе связывающего канала участвовать НЕ вправе);
#   NOT_MEASURED-- заявлено и не проверено.  Законом быть не может, только ДОЛГОМ.
LAW_STATUS = ("MEASURED", "SASS", "PROVEN", "MODEL", "UPPER_BOUND", "NOT_MEASURED")
LAWS_DIRNAME = "laws"
DATA_DIRNAME = "data"
PLUGINS_REL = ("tempo", "plugins")
# Расширения, которые ДОКУМЕНТИРУЮТ, но не ИСПОЛНЯЮТ.  Закон, у которого все места такие,
# автоматически становится ДОЛГОМ: пожелание в прозе не падает, когда закон нарушен.
DOC_SUFFIXES = (".md", ".txt", ".rst")


class LawError(Exception):
    """Запись реестра нарушила схему.  ОТКАЗ на загрузке, а не предупреждение."""


@dataclass(frozen=True)
class Card:
    index: int
    clock_mhz: int
    foreign_procs: int
    date: str


@dataclass(frozen=True)
class Anchor:
    report: str
    section: str = ""
    numbers: tuple = ()
    tool: str = ""
    status: str = "MEASURED"
    bias: str = ""
    card: Optional[Card] = None


@dataclass(frozen=True)
class Falsifier:
    what: str
    how: str = ""
    cost: str = ""


@dataclass(frozen=True)
class Home:
    kind: str
    path: str
    symbol: str = ""

    @property
    def documentary(self) -> bool:
        return self.path.endswith(DOC_SUFFIXES)


@dataclass(frozen=True)
class Forbidden:
    """ЗАПРЕЩЁННАЯ ФОРМУЛИРОВКА -- падающая проверка ОПРОВЕРЖЕНИЯ.

    У закона место -- символ, который считает.  У опровержения места нет и быть не может:
    опровергнутое утверждение НЕ СЧИТАЕТ, оно ГОВОРИТ.  Поэтому его проверка -- обратная:
    фразы не должно остаться в дереве.  Без неё опровержение живёт ровно до первого, кто
    перепишет старый вывод из памяти, -- а в этом проекте так уже было четырежды.
    """

    pattern: str
    why: str = ""
    allow: tuple = ()  # где фраза законна: разбор самого опровержения, запись о замере


@dataclass(frozen=True)
class Law:
    id: str
    kind: str
    statement: str
    anchor: Anchor
    falsifier: Falsifier
    scope: Mapping[str, Any]
    home: tuple = ()
    forbidden: tuple = ()
    supersedes: tuple = ()
    superseded_by: str = ""
    tasks: tuple = ()
    owner: str = ""  # "" -- корень (инструмент/метод); иначе имя каталога плагина
    source: str = ""  # относительный путь файла реестра

    @property
    def wired(self) -> bool:
        """Есть ли МЕСТО, которое исполняется (а не только описывает)."""
        return any(not h.documentary for h in self.home) or bool(self.forbidden)

    @property
    def has_selftest(self) -> bool:
        """Падающая проверка: случай самопроверки ЛИБО запрет формулировки."""
        return any(h.kind == "случай_самопроверки" for h in self.home) or bool(
            self.forbidden
        )


# ================================================================================================
# 1. ЧТЕНИЕ
# ================================================================================================
def _need(d: Mapping[str, Any], key: str, where: str):
    if key not in d or d[key] in (None, "", [], {}):
        raise LawError("%s: поле %r пусто или отсутствует" % (where, key))
    return d[key]


def _card(d) -> Optional[Card]:
    if not d:
        return None
    return Card(
        index=int(d["index"]),
        clock_mhz=int(d["clock_mhz"]),
        foreign_procs=int(d["foreign_procs"]),
        date=str(d.get("date", "")),
    )


def _law_from(d: Mapping[str, Any], owner: str, source: str) -> Law:
    where = "%s:%s" % (source, d.get("id", "<без id>"))
    if d.get("schema", SCHEMA) != SCHEMA:
        raise LawError("%s: схема %r, ожидается %r" % (where, d.get("schema"), SCHEMA))
    kind = _need(d, "kind", where)
    if kind not in KINDS:
        raise LawError("%s: kind %r вне %r" % (where, kind, KINDS))
    a = _need(d, "anchor", where)
    f = _need(d, "falsifier", where)
    return Law(
        id=str(_need(d, "id", where)),
        kind=kind,
        statement=str(_need(d, "statement", where)),
        anchor=Anchor(
            report=str(_need(a, "report", where + ".anchor")),
            section=str(a.get("section", "")),
            numbers=tuple(str(x) for x in a.get("numbers", ())),
            tool=str(a.get("tool", "")),
            status=str(a.get("status", "MEASURED")),
            bias=str(a.get("bias", "")),
            card=_card(a.get("card")),
        ),
        falsifier=Falsifier(
            what=str(_need(f, "what", where + ".falsifier")),
            how=str(f.get("how", "")),
            cost=str(f.get("cost", "")),
        ),
        scope=dict(_need(d, "scope", where)),
        home=tuple(
            Home(
                kind=str(h.get("kind", "")),
                path=str(h["path"]),
                symbol=str(h.get("symbol", "")),
            )
            for h in d.get("home", ())
        ),
        forbidden=tuple(
            Forbidden(
                pattern=str(_need(x, "pattern", where + ".forbidden")),
                why=str(x.get("why", "")),
                allow=tuple(str(p) for p in x.get("allow", ())),
            )
            for x in d.get("forbidden", ())
        ),
        supersedes=tuple(str(x) for x in d.get("supersedes", ())),
        superseded_by=str(d.get("superseded_by", "")),
        tasks=tuple(int(x) for x in d.get("tasks", ())),
        owner=owner,
        source=source,
    )


def load_file(path: str, owner: str, root: str) -> list:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    rel = os.path.relpath(path, root).replace(os.sep, "/")
    items = raw.get("laws") if isinstance(raw, dict) and "laws" in raw else [raw]
    if isinstance(raw, dict) and "laws" in raw and raw.get("schema", SCHEMA) != SCHEMA:
        raise LawError(
            "%s: схема файла %r, ожидается %r" % (rel, raw.get("schema"), SCHEMA)
        )
    out = []
    for it in items:
        d = dict(it)
        d.setdefault("schema", SCHEMA)
        out.append(_law_from(d, owner, rel))
    return out


def _dirs(root: str):
    """Где вообще лежат реестры.  Владелец -- ИМЯ КАТАЛОГА, а не литерал в коде."""
    yield "", os.path.join(root, DATA_DIRNAME, LAWS_DIRNAME)
    plug = os.path.join(root, *PLUGINS_REL)
    if os.path.isdir(plug):
        for name in sorted(os.listdir(plug)):
            d = os.path.join(plug, name, DATA_DIRNAME, LAWS_DIRNAME)
            if os.path.isdir(d):
                yield name, d


def load(root: str) -> list:
    """Весь реестр.  Порядок устойчив: сперва корень, дальше плагины по алфавиту."""
    out = []
    seen = {}
    for owner, d in _dirs(root):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            for law in load_file(os.path.join(d, fn), owner, root):
                if law.id in seen:
                    raise LawError(
                        "идентификатор %s объявлен дважды: %s и %s"
                        % (law.id, seen[law.id], law.source)
                    )
                seen[law.id] = law.source
                out.append(law)
    return out


def by_id(laws: Sequence) -> dict:
    return {l.id: l for l in laws}


# ================================================================================================
# 2. ПРОВЕРКИ.  Каждая возвращает СПИСОК ПРЕТЕНЗИЙ; пустой список = пройдено.
# ================================================================================================
def check_record(law: Law) -> list:
    """G9: закон без ОБЛАСТИ и без ФАЛЬСИФИКАТОРА -- мина, а не знание."""
    bad = []
    if not law.statement.strip():
        bad.append("%s: пустая формулировка" % law.id)
    if not law.anchor.report:
        bad.append("%s: у якоря не назван отчёт" % law.id)
    if not law.anchor.numbers:
        bad.append("%s: у якоря нет НИ ОДНОГО числа -- нечего сверять" % law.id)
    if not law.falsifier.what.strip():
        bad.append("%s: нет фальсификатора" % law.id)
    if not law.falsifier.how.strip():
        bad.append("%s: у фальсификатора не сказано КАК проверять" % law.id)
    if not law.scope:
        bad.append("%s: пустая область действия" % law.id)
    if law.anchor.status not in LAW_STATUS:
        bad.append(
            "%s: статус якоря %r вне %r" % (law.id, law.anchor.status, LAW_STATUS)
        )
    if law.anchor.status == "NOT_MEASURED" and law.kind != "ДОЛГ":
        bad.append("%s: NOT_MEASURED не может быть законом -- только ДОЛГОМ" % law.id)
    if law.anchor.status == "MEASURED":
        c = law.anchor.card
        if c is None:
            bad.append(
                "%s: MEASURED без карты -- заявка на точность, которой у неё нет"
                % law.id
            )
        elif c.foreign_procs != 0:
            bad.append(
                "%s: MEASURED снят при %d чужих процессах -- замер недействителен"
                % (law.id, c.foreign_procs)
            )
    # Опровержение обязано НАЗВАТЬ, что именно оно опровергает: либо ссылкой на запись
    # реестра, либо дословной запрещённой формулировкой.  Иначе это просто мнение.
    if law.kind == "ОПРОВЕРЖЕНИЕ" and not (
        law.superseded_by or law.supersedes or law.forbidden
    ):
        bad.append(
            "%s: ОПРОВЕРЖЕНИЕ не называет опровергнутое -- нужна ссылка (supersedes/"
            "superseded_by) либо запрещённая формулировка" % law.id
        )
    return bad


def check_homes(law: Law, root: str) -> list:
    """G10: место обязано РАЗРЕШАТЬСЯ -- путь есть, символ в файле находится."""
    bad = []
    for h in law.home:
        p = os.path.join(root, h.path)
        if not os.path.exists(p):
            bad.append("%s: место %s не существует" % (law.id, h.path))
            continue
        if not h.symbol:
            continue
        try:
            with open(p, encoding="utf-8", errors="ignore") as fh:
                txt = fh.read()
        except OSError as e:
            bad.append("%s: место %s не читается (%s)" % (law.id, h.path, e))
            continue
        if h.symbol not in txt:
            bad.append("%s: в %s нет символа %r" % (law.id, h.path, h.symbol))
    # ОБЪЯВЛЕНИЕ ПРОВЕРЯЕТСЯ В ОБЕ СТОРОНЫ, иначе реестр разойдётся с деревом за неделю.
    if law.kind != "ДОЛГ" and not law.wired:
        bad.append(
            "%s: объявлен внесённым, но все места ДОКУМЕНТАРНЫ (.md) -- это ДОЛГ. "
            "Пометьте kind: ДОЛГ или назовите исполняемое место." % law.id
        )
    if law.kind == "ДОЛГ" and law.wired and law.has_selftest:
        bad.append(
            "%s: объявлен ДОЛГОМ, но у него есть и исполняемое место, и падающая проверка -- "
            "долг оплачен, повысьте вид до ЗАКОНа/ОПРОВЕРЖЕНИЯ" % law.id
        )
    return bad


def check_links(laws: Sequence) -> list:
    """Вытеснение обязано быть ДВУСТОРОННИМ, иначе цепочка рвётся при чтении с любого конца."""
    idx = by_id(laws)
    bad = []
    for l in laws:
        for s in l.supersedes:
            if s not in idx:
                bad.append("%s: вытесняет несуществующий %s" % (l.id, s))
            elif idx[s].superseded_by != l.id:
                bad.append("%s вытесняет %s, а тот не ссылается обратно" % (l.id, s))
        if l.superseded_by:
            if l.superseded_by not in idx:
                bad.append("%s: вытеснен несуществующим %s" % (l.id, l.superseded_by))
            elif l.id not in idx[l.superseded_by].supersedes:
                bad.append(
                    "%s вытеснен %s, а тот его не перечисляет" % (l.id, l.superseded_by)
                )
            elif l.kind != "ОПРОВЕРЖЕНИЕ":
                bad.append("%s вытеснен, но не помечен ОПРОВЕРЖЕНИЕм" % l.id)
    return bad


def check_selftests(laws: Sequence) -> list:
    """Правило приёмки открытия, пункт (в): закон без ПАДАЮЩЕЙ проверки -- комментарий."""
    bad = []
    for l in laws:
        if l.kind == "ДОЛГ":
            continue
        if not l.has_selftest:
            bad.append(
                "%s: внесён, но у него нет ни места вида 'случай_самопроверки', ни запрещённой "
                "формулировки -- закон без падающей проверки это комментарий" % l.id
            )
    return bad


# ------------------------------------------------------------------------------------------------
# ЗАПРЕЩЁННЫЕ ФОРМУЛИРОВКИ.  Проверка ОПРОВЕРЖЕНИЯ -- обратная: фразы в дереве быть не должно.
# ------------------------------------------------------------------------------------------------
# Реестр цитирует опровергнутое дословно, и порождённая из него опись -- тоже.  Искать запрет
# там значило бы запретить самому себе называть то, что запрещено.
SELF_EXEMPT = (DATA_DIRNAME + "/" + LAWS_DIRNAME + "/", "docs/LAWS.md")


def normalize(text: str) -> str:
    """Свернуть пробелы и регистр.

    ПОЧЕМУ ЭТО НЕ КОСМЕТИКА.  Опровергнутая фраза живёт в ПРОЗЕ, а проза переносится по
    строкам и пишется то заглавными, то строчными.  Первая редакция этой проверки искала
    точную подстроку и НЕ НАШЛА три места из пяти: в одном фраза была разорвана переводом
    строки, в двух отличался регистр.  Проверка, которую так легко обойти случайно,
    не проверяет ничего.
    """
    return " ".join(text.split()).casefold()


def check_forbidden(
    laws: Sequence, root: str, subdirs: Sequence, suffixes: Sequence
) -> list:
    bad = []
    files = []
    for sub in subdirs:
        base = os.path.join(root, sub)
        if os.path.isfile(base):
            files.append(base)
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in filenames:
                if fn.endswith(tuple(suffixes)):
                    files.append(os.path.join(dirpath, fn))
    texts = {}
    for p in files:
        rel = os.path.relpath(p, root).replace(os.sep, "/")
        if any(x in rel for x in SELF_EXEMPT):
            continue
        try:
            with open(p, encoding="utf-8", errors="ignore") as fh:
                texts[rel] = normalize(fh.read())
        except OSError:
            continue
    for l in laws:
        key = normalize(l.id)
        for f in l.forbidden:
            pat = normalize(f.pattern)
            for rel, txt in sorted(texts.items()):
                if any(a in rel for a in f.allow):
                    continue
                for pos in _occurrences(txt, pat):
                    if (
                        key
                        in txt[max(0, pos - CITE_WINDOW) : pos + len(pat) + CITE_WINDOW]
                    ):
                        continue  # это ЦИТАТА с указанием опровергающей записи -- законно
                    bad.append(
                        "%s: в %s осталась ОПРОВЕРГНУТАЯ формулировка %r (%s)"
                        % (l.id, rel, f.pattern, f.why)
                    )
                    break
    return bad


# ОКНО ЦИТИРОВАНИЯ.  Опровергнутую фразу МОЖНО написать -- но только рядом с идентификатором
# записи, которая её опровергает.  Правило механическое и оттого не имеет дыр: списка
# «файлов-исключений», в который со временем попадает всё, здесь нет.  Цена соблюдения --
# одна ссылка; цена несоблюдения -- вывод, который живёт после своего опровержения.
CITE_WINDOW = 600


def _occurrences(text: str, pat: str):
    i = 0
    while True:
        i = text.find(pat, i)
        if i < 0:
            return
        yield i
        i += len(pat)


# ================================================================================================
# 3. МАРКЕРЫ В КОДЕ (двусторонняя привязка).  Маркер -- `LAW=<id>` в любом файле продукта.
# ================================================================================================
MARKER = "LAW="


def markers_in(text: str) -> set:
    """Все идентификаторы, на которые ссылается ЭТОТ текст."""
    out = set()
    i = 0
    while True:
        i = text.find(MARKER, i)
        if i < 0:
            return out
        j = i + len(MARKER)
        k = j
        while k < len(text) and (text[k].isalnum() or text[k] in "-_"):
            k += 1
        if k > j:
            out.add(text[j:k])
        i = k


def scan_markers(root: str, subdirs: Sequence, suffixes: Sequence) -> dict:
    """{идентификатор: [файлы]} по дереву продукта."""
    found = {}
    for sub in subdirs:
        base = os.path.join(root, sub)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in filenames:
                if not fn.endswith(tuple(suffixes)):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    with open(p, encoding="utf-8", errors="ignore") as fh:
                        txt = fh.read()
                except OSError:
                    continue
                for lid in markers_in(txt):
                    found.setdefault(lid, []).append(
                        os.path.relpath(p, root).replace(os.sep, "/")
                    )
    return found


# ================================================================================================
# 4. ВЫБОРКА: «какие законы действуют В ЭТОЙ ТОЧКЕ»
# ================================================================================================
def _in_range(spec: str, value: float) -> Optional[bool]:
    """'8..4096' -> отрезок; '<=64' / '>=128' -> луч; иначе None (ось не ограничена числом)."""
    s = str(spec).strip()
    try:
        if ".." in s:
            lo, hi = s.split("..", 1)
            return float(lo) <= value <= float(hi)
        if s.startswith("<="):
            return value <= float(s[2:])
        if s.startswith(">="):
            return value >= float(s[2:])
        if s.startswith("<"):
            return value < float(s[1:])
        if s.startswith(">"):
            return value > float(s[1:])
        return abs(float(s) - value) < 1e-12
    except ValueError:
        return None


def acts_at(law: Law, axis: str, value: str):
    """(действует?, пояснение).  None означает «ось не названа» -- это НЕ «действует»."""
    if axis not in law.scope:
        return None, "ось %s в области не названа" % axis
    spec = law.scope[axis]
    if isinstance(spec, (list, tuple)):
        return (value in [str(x) for x in spec]), "перечень %s" % list(spec)
    try:
        v = float(value)
    except ValueError:
        return (str(spec) == value), "значение %r" % spec
    r = _in_range(spec, v)
    if r is None:
        return None, "область по %s задана словами: %r" % (axis, spec)
    return r, "область %s = %s" % (axis, spec)


def select(laws: Sequence, arch=None, kind=None, unwired=False, scope=None) -> list:
    out = []
    for l in laws:
        if arch is not None and l.scope.get("arch") != arch and l.owner != arch:
            continue
        if kind is not None and l.kind != kind:
            continue
        if unwired and l.wired:
            continue
        if scope:
            axis, _, value = scope.partition("=")
            ok, _why = acts_at(l, axis.strip(), value.strip())
            if ok is False:
                continue
        out.append(l)
    return out


# ================================================================================================
# 5. ПЕЧАТЬ
# ================================================================================================
def render_text(laws: Sequence, root: str = "") -> str:
    lines = []
    for l in laws:
        mark = {"ЗАКОН": "З", "ОПРОВЕРЖЕНИЕ": "О", "ДОЛГ": "Д"}[l.kind]
        where = "вшит" if l.wired else "ДОЛГ (только проза)"
        lines.append("[%s] %-22s %s" % (mark, l.id, l.statement))
        lines.append(
            "      якорь ....... %s %s | %s"
            % (l.anchor.report, l.anchor.section, ", ".join(l.anchor.numbers))
        )
        if l.anchor.bias:
            lines.append("      смещение .... %s" % l.anchor.bias)
        lines.append("      фальсиф. .... %s" % l.falsifier.what)
        lines.append("      область ..... %s" % _scope_line(l.scope))
        lines.append(
            "      место ....... %s"
            % (
                "; ".join(
                    "%s %s%s" % (h.kind, h.path, (":" + h.symbol) if h.symbol else "")
                    for h in l.home
                )
                or "НЕТ (%s)" % where
            )
        )
        for f in l.forbidden:
            lines.append("      ЗАПРЕТ ...... %r" % f.pattern)
        if l.superseded_by:
            lines.append("      вытеснен .... %s" % l.superseded_by)
        lines.append("")
    return "\n".join(lines)


def _scope_line(scope: Mapping[str, Any]) -> str:
    parts = []
    for k in sorted(scope):
        if k == "not":
            continue
        v = scope[k]
        parts.append(
            "%s=%s"
            % (k, ",".join(str(x) for x in v) if isinstance(v, (list, tuple)) else v)
        )
    if scope.get("not"):
        parts.append("НЕ: %s" % "; ".join(str(x) for x in scope["not"]))
    return "  ".join(parts)


def render_markdown(laws: Sequence) -> str:
    """docs/LAWS.md порождается ОТСЮДА.  Проза, разошедшаяся с реестром, ловится гейтом."""
    из_кода = [l for l in laws if l.kind == "ЗАКОН"]
    опров = [l for l in laws if l.kind == "ОПРОВЕРЖЕНИЕ"]
    долги = [l for l in laws if l.kind == "ДОЛГ"]
    out = []
    out.append("<!-- SPDX-License-Identifier: LicenseRef-TRL-1.0")
    out.append("     Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>")
    out.append(
        "     Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement. -->"
    )
    out.append("")
    out.append("<!-- ПОРОЖДЁН: python3 -m tempo.cli laws --md > docs/LAWS.md")
    out.append(
        "     РУКАМИ НЕ ПРАВИТЬ. Источник -- data/laws/*.json и <плагин>/data/laws/*.json."
    )
    out.append("     Расхождение этого файла с реестром роняет гейт G10. -->")
    out.append("")
    out.append("# РЕЕСТР ЗАКОНОВ")
    out.append("")
    out.append(
        "Запись закона -- обобщение дисциплины `Rate` (число + единица + статус + происхождение) "
        "с ЧИСЛА на УТВЕРЖДЕНИЕ. Шесть обязательных полей: **формулировка**, **якорь** (отчёт, "
        "числа, прибор, карта), **фальсификатор**, **область действия**, **место в продукте**, "
        "**вид**. Открытие без места и без падающей проверки -- пожелание, а не закон "
        "(`docs/CONTRACT.md`, правило приёмки открытия)."
    )
    out.append("")
    out.append(
        "| вид | сколько |\n|---|--:|\n| ЗАКОН | %d |\n| ОПРОВЕРЖЕНИЕ | %d |\n| ДОЛГ | %d |"
        % (len(из_кода), len(опров), len(долги))
    )
    out.append("")
    for title, group, note in (
        ("## ЗАКОНЫ", из_кода, ""),
        (
            "## ОПРОВЕРЖЕНИЯ",
            опров,
            "Опровергнутая запись НЕ УДАЛЯЕТСЯ: отрицательное знание теряется первым, поэтому "
            "хранится с наибольшей защитой. Удалить его -- значит потерять и ОБЛАСТЬ, в которой "
            "исходное утверждение верно, и урок о том, как его вынесли за эту область.",
        ),
        (
            "## ДОЛГИ (открыто, в продукт НЕ внесено)",
            долги,
            "Место названо только прозой. Это честная картина, а не провал: гейт G10 печатает "
            "такие записи отдельно, чтобы их нельзя было принять за внесённые.",
        ),
    ):
        if not group:
            continue
        out.append(title)
        out.append("")
        if note:
            out.append(note)
            out.append("")
        for l in group:
            out.extend(_md_law(l))
    return "\n".join(out).rstrip() + "\n"


def _md_law(l: Law) -> list:
    o = ["### %s" % l.id, ""]
    o.append("**Формулировка.** %s" % l.statement)
    o.append("")
    a = l.anchor
    who = (
        ""
        if a.card is None
        else (
            " Карта %d, %d МГц, чужих процессов %d, %s."
            % (a.card.index, a.card.clock_mhz, a.card.foreign_procs, a.card.date)
        )
    )
    o.append(
        "**Якорь** (`%s`). %s%s Статус: `%s`.%s%s"
        % (
            a.report,
            (a.section + ": ") if a.section else "",
            "; ".join(a.numbers),
            a.status,
            who,
            (" Прибор: `%s`." % a.tool) if a.tool else "",
        )
    )
    if a.bias:
        o.append("")
        o.append("**Смещение замера.** %s" % a.bias)
    o.append("")
    f = l.falsifier
    o.append(
        "**Фальсификатор.** %s%s%s"
        % (
            f.what,
            (" _Как:_ " + f.how) if f.how else "",
            (" _Цена:_ " + f.cost) if f.cost else "",
        )
    )
    o.append("")
    o.append("**Область действия.** %s" % _scope_line(l.scope))
    o.append("")
    if l.home:
        o.append("**Где живёт в продукте.**")
        o.append("")
        for h in l.home:
            o.append(
                "* `%s` -- %s%s"
                % (h.path, h.kind, (", символ `%s`" % h.symbol) if h.symbol else "")
            )
        o.append("")
    else:
        o.append("**Где живёт в продукте.** НИГДЕ -- это долг.")
        o.append("")
    if l.forbidden:
        o.append(
            "**Запрещённые формулировки** (гейт G11 роняет сборку, если они вернулись):"
        )
        o.append("")
        for f in l.forbidden:
            o.append("* `%s` -- %s" % (f.pattern, f.why))
        o.append("")
    tail = []
    if l.supersedes:
        tail.append("вытесняет: %s" % ", ".join(l.supersedes))
    if l.superseded_by:
        tail.append("вытеснен: %s" % l.superseded_by)
    if l.tasks:
        tail.append("задачи: %s" % ", ".join(str(t) for t in l.tasks))
    if tail:
        o.append("_%s_" % ("; ".join(tail)))
        o.append("")
    return o
