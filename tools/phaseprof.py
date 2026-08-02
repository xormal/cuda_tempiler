# -*- coding: utf-8 -*-
"""ФАЗОВЫЙ ПРОФИЛИРОВЩИК ЯДЕР sm_70: разложение времени ядра по НАЗВАННЫМ ФАЗАМ.

ЗАЧЕМ. Единица анализа в этом проекте -- ФАЗА, а не команда. Но фазового профилировщика не
существует: ncu 2025.x на Volta не работает вовсе, у рабочего 2024.1 CSV ломается о локаль, а
привязки "во что уходит время ВНУТРИ ядра" он не даёт ни в какой версии. Разложение форварда
(35.1 / 19.7 / 15.2 / 5.7 / 2.0 %) получено РУКАМИ: программист писал #ifdef, собирал заведомо
неверное ядро со снятой фазой, мерил время, считал долю. Человеко-часы на фазу -> раз в полгода.

ЧТО ДЕЛАЕТ ЭТОТ ИНСТРУМЕНТ
  1. находит фазы, размеченные макросом FMHA_PHASE(имя, id) (см. fmha_phase.h);
  2. собирает базу + по варианту на каждую снятую фазу (+ пары + "снять все");
  3. СТАТИЧЕСКИ (без карты!) раскладывает SASS по фазам через привязку строк (-lineinfo) и
     ловит СБИТЫЙ ХВОСТ -- случай, когда снятие фазы X выбило команды фазы Y;
  4. по замеренным временам печатает таблицу долей, СУММУ СТОЛБЦА и НЕВЯЗКУ -- крупно;
  5. разделяет невязку на ПЕРЕКРЫТИЕ и НЕНАЗВАННОЕ (через вариант "снять все" и пары).

ПОЧЕМУ СУММА НЕ ОБЯЗАНА ДАВАТЬ 100 % (и почему её всё равно НАДО печатать).
Снятие фазы меняет ПЕРЕКРЫТИЕ. Фаза, идеально спрятанная за другой, при снятии даст ноль, хотя
занята полностью. Поэтому доли -- это не разбиение, и сумма -- не 100 %. Но именно из-за этого
столбец обязан быть сложен: у нашей таблицы форварда он даёт 77.7 %, то есть ПЯТАЯ ЧАСТЬ времени
префилла лежит в фазах, которые никто не назвал, -- и это простояло в журнале незамеченным.

ТОЧНОЕ РАЗЛОЖЕНИЕ НЕВЯЗКИ (нужен один лишний вариант -- "снять ВСЕ фазы"):

    1  =  SUM(s_i)          +   ( s_all - SUM(s_i) )   +   ( 1 - s_all )
           |                        |                         |
           названо одиночно      ПЕРЕКРЫТИЕ                НЕНАЗВАННОЕ
                                (фазы прячут друг друга;   (пролог, эпилог, запуск,
                                 одиночные снятия          квантование по волнам,
                                 НЕДОсчитывают)            обвязка -- ничьё)

Тождество тривиально; содержательно то, что КАЖДОЕ слагаемое измеримо отдельно. Знак среднего
члена и есть ответ на вопрос "невязка -- это перекрытие или ненайденная фаза".
Пара фаз даёт то же самое поточечно:  e_ij = s_ij - (s_i + s_j);  e_ij > 0 -- прячут друг друга.

ЗАПУСК
    python3 phaseprof.py scan   --spec spec.json
    python3 phaseprof.py static --spec spec.json          # БЕЗ карты, только nvcc/nvdisasm
    python3 phaseprof.py time   --spec spec.json [--pairs]
    python3 phaseprof.py replay --times times.json        # разбор уже записанных времён

ФОРМАТ spec.json -- см. demo_phase.spec.json рядом.
ФОРМАТ times.json: {"base": <t0>, "phases": {"имя": <t_i>}, "all": <t_all>,
                    "pairs": {"a+b": <t_ij>}, "units": "ms", "note": "..."}
                   Вместо времён можно дать ускорения: {"speedup": {"имя": 1.541}}.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys

# ---------------------------------------------------------------------------------------------
# 1. РАЗБОР РАЗМЕТКИ
# ---------------------------------------------------------------------------------------------

PHASE_RE = re.compile(r"\bFMHA_PHASE\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)")
ELSE_RE = re.compile(r"\bFMHA_PHASE_ELSE\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)")
# Пломбу ловим по ЛЮБОМУ имени, содержащему SEAL: в боевом коде она почти всегда обёрнута
# в свой макрос (у нас -- DSEAL/DSEAL_ARR), и поиск строго по FMHA_SEAL дал бы "БЕЗ ПЛОМБЫ" на
# полностью опломбированном ядре, то есть ЛОЖНУЮ ТРЕВОГУ вместо помощи.
SEAL_RE = re.compile(r"\b\w*SEAL\w*\s*\(")
ON_RE = re.compile(r"\bFMHA_PHASE_ON\s*\(\s*(\w+)\s*\)")


def strip_comments(src):
    """Убирает комментарии и строковые литералы, СОХРАНЯЯ длину и переводы строк.

    Нужно, чтобы счётчик скобок не спотыкался о '{' в комментарии, а номера строк не поехали.
    """
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            out[i] = out[i + 1] = " "
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                if src[i] != "\n":
                    out[i] = " "
                i += 1
            if i + 1 < n:
                out[i] = out[i + 1] = " "
                i += 2
        elif c in "\"'":
            q = c
            out[i] = " "
            i += 1
            while i < n and src[i] != q:
                if src[i] == "\\":
                    out[i] = " "
                    i += 1
                if i < n and src[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
                i += 1
        else:
            i += 1
    return "".join(out)


def match_block(src, start):
    """От позиции start ищет первую '{' и её пару. Возвращает (открыв, закрыв) или None."""
    i = src.find("{", start)
    if i < 0:
        return None
    # между макросом и '{' не должно быть ничего, кроме пробелов -- иначе это не тело фазы
    if src[start:i].strip():
        return None
    depth, j = 0, i
    while j < len(src):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return (i, j)
        j += 1
    return None


class Phase(object):
    def __init__(self, name, pid, path, l0, l1):
        self.name = name
        self.id = pid
        self.path = path
        self.l0 = l0
        self.l1 = l1
        self.has_else = False
        self.sealed = False

    @property
    def bit(self):
        return 1 << self.id

    def __repr__(self):
        return "%s(id=%d, %s:%d-%d)" % (
            self.name,
            self.id,
            os.path.basename(self.path),
            self.l0,
            self.l1,
        )


def scan_sources(paths):
    """Возвращает (phases, unparsed). unparsed -- список строк для раздела НЕ РАЗОБРАНО."""
    phases, unparsed = [], []
    for path in paths:
        if not os.path.exists(path):
            unparsed.append("файл не найден: %s" % path)
            continue
        raw = open(path, encoding="utf-8", errors="replace").read()
        src = strip_comments(raw)
        found_any = False
        for m in PHASE_RE.finditer(src):
            found_any = True
            name, sid = m.group(1), m.group(2)
            line = src[: m.start()].count("\n") + 1
            try:
                pid = int(sid, 0)
            except ValueError:
                unparsed.append(
                    "%s:%d  FMHA_PHASE(%s, %s): id -- не число, фаза пропущена"
                    % (os.path.basename(path), line, name, sid)
                )
                continue
            blk = match_block(src, m.end())
            if blk is None:
                unparsed.append(
                    "%s:%d  FMHA_PHASE(%s, %d): не нашёл парную '}' -- границы фазы "
                    "неизвестны, из статического разбора ИСКЛЮЧЕНА"
                    % (os.path.basename(path), line, name, pid)
                )
                continue
            l0 = src[: blk[0]].count("\n") + 1
            l1 = src[: blk[1]].count("\n") + 1
            ph = Phase(name, pid, path, l0, l1)
            # пломба: ищем в пределах 40 строк после конца фазы (обычно сразу за ней)
            tail_beg = blk[1]
            tail_end = blk[1]
            nl = 0
            while tail_end < len(src) and nl < 40:
                if src[tail_end] == "\n":
                    nl += 1
                tail_end += 1
            ph.sealed = bool(SEAL_RE.search(src[tail_beg:tail_end]))
            after = src[blk[1] + 1 : blk[1] + 400]
            me = ELSE_RE.match(after.lstrip()) or ELSE_RE.search(after[:120])
            if me:
                ph.has_else = True
                # FMHA_PHASE_ELSE раскрывается в голое `else` и АРГУМЕНТЫ ИГНОРИРУЕТ -- значит
                # опечатка в имени/id не даст ни ошибки компиляции, ни неверного кода, но
                # прочитается человеком как подстановка ДРУГОЙ фазы. Ловим здесь.
                try:
                    eid = int(me.group(2), 0)
                except ValueError:
                    eid = None
                if me.group(1) != name or eid != pid:
                    unparsed.append(
                        "%s:%d  подстановка FMHA_PHASE_ELSE(%s, %s) стоит за фазой "
                        "FMHA_PHASE(%s, %d): аргументы ELSE НЕ ПРОВЕРЯЮТСЯ компилятором, и "
                        "расхождение имён читается как подстановка другой фазы"
                        % (
                            os.path.basename(path),
                            line,
                            me.group(1),
                            me.group(2),
                            name,
                            pid,
                        )
                    )
            elif re.match(r"\s*else\b", after):
                ph.has_else = True
                unparsed.append(
                    "%s:%d  за фазой %s стоит ГОЛОЕ `else` вместо "
                    "FMHA_PHASE_ELSE(%s, %d) -- работает, но при переносе фазы "
                    "молча оторвётся" % (os.path.basename(path), line, name, name, pid)
                )
            phases.append(ph)
        for m in ON_RE.finditer(src):
            line = src[: m.start()].count("\n") + 1
            unparsed.append(
                "%s:%d  FMHA_PHASE_ON(%s) -- условие вне блочной разметки; границ у "
                "него нет, статический разбор его НЕ ВИДИТ"
                % (os.path.basename(path), line, m.group(1))
            )
        if not found_any:
            unparsed.append(
                "в %s разметки FMHA_PHASE НЕТ (файл в spec, но не размечен)"
                % os.path.basename(path)
            )

    # --- проверки целостности множества фаз
    by_id = {}
    for p in phases:
        by_id.setdefault(p.id, []).append(p)
    for pid, group in sorted(by_id.items()):
        names = set(x.name for x in group)
        if len(names) > 1:
            unparsed.append(
                "id=%d носят РАЗНЫЕ фазы %s -- маска снимет их ВМЕСТЕ, доли "
                "перепутаны" % (pid, sorted(names))
            )
    if by_id:
        want = set(range(max(by_id) + 1))
        gaps = sorted(want - set(by_id))
        if gaps:
            unparsed.append(
                "пропуски в id: %s (не ошибка, но пары строятся по битам -- "
                "проверьте, что фаза не забыта)" % gaps
            )
        if max(by_id) > 31:
            unparsed.append("id=%d > 31: не влезает в 32-битную маску" % max(by_id))
    # вложенность
    for a in phases:
        for b in phases:
            if a is b or a.path != b.path:
                continue
            if b.l0 > a.l0 and b.l1 < a.l1:
                unparsed.append(
                    "фаза %s вложена в %s -- команды отнесены ВНУТРЕННЕЙ, доля "
                    "внешней занижена" % (b.name, a.name)
                )
    for p in phases:
        if not p.sealed and not p.has_else:
            unparsed.append(
                "фаза %s БЕЗ ПЛОМБЫ и без подстановки: при снятии компилятор вправе "
                "выбросить зависимый код ДРУГИХ фаз, и доля %s будет ЗАВЫШЕНА"
                % (p.name, p.name)
            )
    return phases, unparsed


# ---------------------------------------------------------------------------------------------
# 2. СБОРКА И СТАТИЧЕСКИЙ РАЗБОР SASS
# ---------------------------------------------------------------------------------------------

INSTR_RE = re.compile(r"^\s+/\*[0-9a-f]{4,}\*/\s+(.*?);")
LINE_RE = re.compile(r'^\s*//## File "([^"]+)", line (\d+)')
FUNC_RE = re.compile(r"^\s*\.text\.(\S+):")


def run(cmd, cwd=None):
    p = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return p.returncode, p.stdout


def build_variant(spec, mask, tag, extra=""):
    out = spec["out"].format(mask=mask, tag=tag)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cmd = spec["build"].format(mask=mask, tag=tag, out=out, extra=extra)
    rc, log = run(cmd)
    return rc, out, log, cmd


def disasm(cubin, nvdisasm):
    rc, txt = run("%s -g -c %s" % (shlex.quote(nvdisasm), shlex.quote(cubin)))
    if rc != 0:
        return None, txt
    return txt, ""


def attribute(sass_text, phases, kernel_filter=None):
    """SASS -> {имя фазы: число команд}, плюс 'НЕ ОТНЕСЕНО' и 'БЕЗ ПРИВЯЗКИ'.

    Привязка: -lineinfo даёт (файл, строка) для КАЖДОГО блока команд. Строку сопоставляем
    диапазонам фаз. Вложенные -- по САМОМУ УЗКОМУ диапазону.
    """
    ranges = []
    for p in phases:
        ranges.append((os.path.basename(p.path), p.l0, p.l1, p.name))
    counts = {p.name: 0 for p in phases}
    counts["НЕ ОТНЕСЕНО"] = 0
    counts["БЕЗ ПРИВЯЗКИ"] = 0
    per_line = {}
    cur = None
    cur_fn = None
    for ln in sass_text.splitlines():
        mf = FUNC_RE.match(ln)
        if mf:
            cur_fn = mf.group(1)
            cur = None
            continue
        m = LINE_RE.match(ln)
        if m:
            cur = (os.path.basename(m.group(1)), int(m.group(2)))
            continue
        if not INSTR_RE.match(ln):
            continue
        if kernel_filter and cur_fn and kernel_filter not in cur_fn:
            continue
        if cur is None:
            counts["БЕЗ ПРИВЯЗКИ"] += 1
            continue
        per_line[cur] = per_line.get(cur, 0) + 1
        best = None
        for f, l0, l1, name in ranges:
            if f == cur[0] and l0 <= cur[1] <= l1:
                if best is None or (l1 - l0) < best[0]:
                    best = (l1 - l0, name)
        counts[best[1] if best else "НЕ ОТНЕСЕНО"] += 1
    return counts, per_line


def ptxas_stats(log):
    """Регистры/разлив из вывода `-Xptxas -v`. Нужны, чтобы поймать ПОДМЕНУ РАСПИСАНИЯ:
    если у стрип-варианта резко другое число регистров, сравнение времён меряет уже не фазу."""
    regs = re.findall(r"Used (\d+) registers", log)
    spill_s = re.findall(r"(\d+) bytes spill stores", log)
    spill_l = re.findall(r"(\d+) bytes spill loads", log)
    stack = re.findall(r"(\d+) bytes stack frame", log)
    f = lambda a: max(int(x) for x in a) if a else None
    return {
        "regs": f(regs),
        "spill_st": f(spill_s),
        "spill_ld": f(spill_l),
        "stack": f(stack),
    }


# ---------------------------------------------------------------------------------------------
# 3. АРИФМЕТИКА ДОЛЕЙ
# ---------------------------------------------------------------------------------------------


def share(t0, ti):
    """Доля фазы = 1 - t_снято/t_база.  Для ускорения x: 1 - 1/x."""
    return 1.0 - (ti / t0)


def load_times(obj):
    """Нормализует times.json: времена ИЛИ ускорения -> (t0, {имя: t_i}, t_all, {пара: t})."""
    if "speedup" in obj:
        t0 = float(obj.get("base", 1.0))
        ph = {k: t0 / float(v) for k, v in obj["speedup"].items()}
        allt = t0 / float(obj["speedup_all"]) if "speedup_all" in obj else None
        pairs = {k: t0 / float(v) for k, v in obj.get("speedup_pairs", {}).items()}
    else:
        t0 = float(obj["base"])
        ph = {k: float(v) for k, v in obj.get("phases", {}).items()}
        allt = float(obj["all"]) if obj.get("all") is not None else None
        pairs = {k: float(v) for k, v in obj.get("pairs", {}).items()}
    return t0, ph, allt, pairs


# ==============================================================================================
# СУБАДДИТИВНОСТЬ ФАЗ -- ОБЯЗАТЕЛЬНАЯ ОГОВОРКА ПРИ ЛЮБОМ НАБОРЕ ДОЛЕЙ (LAW=L-PHASE-SUBADDITIVE)
# ==============================================================================================
# ЗАМЕРЕНО (EV_additivity.md, карта 0, 1530 МГц, чередование ВСЕХ ЧЕТЫРЁХ вариантов внутри
# раунда, медиана по 9 раундам): снятие двух умножений по отдельности даёт в сумме 49-53 %,
# снятие ОБОИХ сразу -- 43-48 %.  Суммирование фаз ЗАВЫШАЕТ на 8-13 % относительно, и завышение
# УБЫВАЕТ с ростом формы (13.2 -> 9.9 -> 8.2 %).
#
# ПОЧЕМУ ЭТО ПЕЧАТАЕТСЯ ВСЕГДА, А НЕ ХРАНИТСЯ В ПАМЯТИ.  Требование «печатать оговорку рядом с
# числами» было записано в журнале наблюдателя как свойство инструмента -- и НЕ ИСПОЛНЕНО:
# инструмент раскладывал невязку на перекрытие и неназванное (это есть и это хорошо), а слов
# «верхняя оценка» и числа 8-13 % в нём не было.  Требование, записанное и не исполненное, --
# ровно тот случай, ради которого заводится реестр законов с проверкой «место существует».
SUBADD_LO, SUBADD_HI = 8.0, 13.0
SUBADD_NOTE = (
    "СУММА ДОЛЕЙ -- ВЕРХНЯЯ ОЦЕНКА, А НЕ РАЗБИЕНИЕ (LAW=L-PHASE-SUBADDITIVE).\n"
    "  Фазы СУБаддитивны: сняв одну, вторая укладывается плотнее, поэтому одиночные снятия в\n"
    "  сумме ЗАВЫШАЮТ. Замерено: завышение %.0f-%.0f %% относительно (3.8-5.7 п.п.), и оно\n"
    "  УБЫВАЕТ с ростом формы (13.2 -> 9.9 -> 8.2 %%). Читать всякую сумму ниже как «НЕ БОЛЕЕ».\n"
    "  ОБЛАСТЬ: проверено на ОДНОМ теле и ДВУХ фазах из пяти. Аддитивность есть свойство\n"
    "  РЕЖИМА, а не закона: для новой пары проверять заново тем же приёмом -- фальсификатором,\n"
    "  снимающим ОБЕ фазы сразу." % (SUBADD_LO, SUBADD_HI)
)


def subadditivity_note(tot=None):
    """Оговорка + (если дана сумма) её поправленный коридор.  Печатается при ЛЮБОМ наборе долей."""
    out = [SUBADD_NOTE]
    if tot is not None and tot > 0:
        out.append(
            "  ПОПРАВКА К НАПЕЧАТАННОЙ СУММЕ %.1f %%: истинная доля названных фаз лежит в\n"
            "  %.1f-%.1f %% (делением на 1+%.2f и 1+%.2f). Это НЕ уточнение суммы, а её ЧЕСТНЫЙ\n"
            "  интервал: точное значение даёт только вариант, снимающий ВСЕ фазы сразу."
            % (
                100 * tot,
                100 * tot / (1 + SUBADD_HI / 100),
                100 * tot / (1 + SUBADD_LO / 100),
                SUBADD_HI / 100,
                SUBADD_LO / 100,
            )
        )
    return "\n".join(out)


def report_shares(t0, ph, allt, pairs, units="ms", note="", anchor=None):
    W = 78
    print("=" * W)
    print("РАЗЛОЖЕНИЕ ВРЕМЕНИ ПО ФАЗАМ")
    if note:
        print("  " + note)
    print("=" * W)
    print("  база: %.6f %s" % (t0, units))
    print()
    print("  %-26s %12s %10s %10s" % ("фаза снята", "время", "ускорение", "ДОЛЯ"))
    print("  " + "-" * (W - 4))
    tot = 0.0
    negative = []
    rows = sorted(ph.items(), key=lambda kv: -share(t0, kv[1]))
    for name, ti in rows:
        s = share(t0, ti)
        tot += s
        print(
            "  %-26s %12.6f %9.3fx %9.1f%%%s"
            % (
                name,
                ti,
                t0 / ti if ti else float("inf"),
                100 * s,
                "   <-- ОТРИЦАТЕЛЬНАЯ" if s < -0.002 else "",
            )
        )
        if s < -0.002:
            negative.append(name)
    print("  " + "-" * (W - 4))
    print(
        "  %-26s %12s %10s %9.1f%%   <-- СУММА СТОЛБЦА"
        % ("ИТОГО названо", "", "", 100 * tot)
    )

    # ОГОВОРКА ИДЁТ ВПЛОТНУЮ К СУММЕ, а не в конце отчёта: число, прочитанное без неё, --
    # это и есть та ошибка, ради которой она заведена.
    print()
    print("  " + subadditivity_note(tot).replace("\n", "\n  "))

    if negative:
        print()
        print(
            "  ОТРИЦАТЕЛЬНЫЕ ДОЛИ у %s: снятие фазы сделало ядро МЕДЛЕННЕЕ."
            % ", ".join(negative)
        )
        print(
            "  Это НЕ шум и не ошибка знака. Три причины, дающие ровно такую картину:"
        )
        print(
            "    * ПОДМЕНА РАСПИСАНИЯ -- ptxas ушёл на другое число регистров и испортил"
        )
        print("      порядок (замерено в §3b: -96 команд дали БОЛЬШЕЕ время);")
        print(
            "    * снятая фаза ПРЯТАЛА чужую задержку (была прикрытием, а не работой);"
        )
        print(
            "    * фальсификатор поменял поток управления (иная ветвь, иной обход памяти)."
        )
        print(
            "  Пока причина не названа, доля ОСТАЛЬНЫХ фаз тоже под вопросом: база та же,"
        )
        print(
            "  а вот сопоставимость вариантов уже не доказана. Смотрите `static` (регистры)."
        )

    resid = 1.0 - tot
    print()
    print("  #" * (W // 2))
    print(
        "  ##  НЕВЯЗКА = %.1f %%  --  время, НЕ покрытое ни одной названной фазой"
        % (100 * resid)
    )
    print("  #" * (W // 2))

    if allt is None:
        print()
        print(
            "  Разделить невязку НЕЧЕМ: нет варианта 'снять ВСЕ фазы' (ключ \"all\")."
        )
        print("  Пока он не собран, невязка -- это СМЕСЬ двух разных вещей:")
        print(
            "    * ПЕРЕКРЫТИЕ  -- фазы прячут друг друга, одиночные снятия недосчитывают;"
        )
        print(
            "    * НЕНАЗВАННОЕ -- пролог, эпилог, запуск, квантование по волнам, обвязка."
        )
        print("  Соберите вариант с маской ВСЕХ битов -- разложение станет точным.")
    else:
        s_all = share(t0, allt)
        overlap = s_all - tot
        unnamed = 1.0 - s_all
        print()
        print(
            "  РАЗЛОЖЕНИЕ НЕВЯЗКИ (вариант 'снято ВСЁ': %.6f %s, доля %.1f %%):"
            % (allt, units, 100 * s_all)
        )
        print("    названо одиночными снятиями .... %6.1f %%" % (100 * tot))
        print(
            "    ПЕРЕКРЫТИЕ (s_all - сумма) ..... %6.1f %%   %s"
            % (
                100 * overlap,
                "фазы прячут друг друга"
                if overlap > 0.005
                else (
                    "общий узел/пол: снятия мешают друг другу"
                    if overlap < -0.005
                    else "~нет"
                ),
            )
        )
        print(
            "    НЕНАЗВАННОЕ (1 - s_all) ........ %6.1f %%   пролог/эпилог/запуск/волны/обвязка"
            % (100 * unnamed)
        )
        print("    " + "-" * 46)
        print(
            "    сумма ......................... %6.1f %%"
            % (100 * (tot + overlap + unnamed))
        )
        if unnamed > 0.05:
            print()
            print(
                "    ВНИМАНИЕ: %.1f %% времени НЕ ПРИНАДЛЕЖИТ НИ ОДНОЙ ФАЗЕ."
                % (100 * unnamed)
            )
            print(
                "    Это не погрешность, это НЕНАЗВАННАЯ РАБОТА. Кандидаты по нашим ядрам:"
            )
            print(
                "      пересчёт накопителя онлайн-софтмакса; эпилог (запись O и LSE);"
            )
            print(
                "      пролог (заполнение конвейера); квантование по волнам и перекос по SM;"
            )
            print("      обвязка на ВХОДЕ. Пролог и волны меряются БЕЗ правки кода.")

    if pairs:
        print()
        print("  ПЕРЕКРЫТИЕ ПАРАМИ:  e = s(пара) - s(a) - s(b)")
        print(
            "  %-26s %10s %10s %10s %10s"
            % ("пара", "s(пара)", "s(a)+s(b)", "e", "чтение")
        )
        print("  " + "-" * (W - 4))
        for key, tij in sorted(pairs.items()):
            a, b = key.split("+")
            if a not in ph or b not in ph:
                print("  %-26s   -- одна из фаз пары не мерена, пропуск" % key)
                continue
            sij = share(t0, tij)
            ssum = share(t0, ph[a]) + share(t0, ph[b])
            e = sij - ssum
            tag = (
                "прячут друг друга"
                if e > 0.005
                else ("общий узел" if e < -0.005 else "независимы")
            )
            print(
                "  %-26s %9.1f%% %9.1f%% %9.1f%% %10s"
                % (key, 100 * sij, 100 * ssum, 100 * e, tag)
            )

    if anchor:
        print()
        print("  СВЕРКА С ЯКОРЕМ (известные числа §3b VOLTA_SM70.md):")
        ok = True
        for name, want in anchor.items():
            if name not in ph:
                print("    %-22s ЯКОРЬ %5.1f%%   -- фаза не мерена" % (name, want))
                ok = False
                continue
            got = 100 * share(t0, ph[name])
            d = got - want
            mark = "СОШЛОСЬ" if abs(d) <= 0.5 else ("расхождение %+.1f п.п." % d)
            if abs(d) > 0.5:
                ok = False
            print(
                "    %-22s ЯКОРЬ %5.1f%%   получено %5.1f%%   %s"
                % (name, want, got, mark)
            )
        print(
            "    ИТОГ СВЕРКИ: %s"
            % ("ЯКОРЬ ВОСПРОИЗВЕДЁН" if ok else "ЯКОРЬ НЕ ВОСПРОИЗВЕДЁН")
        )
    return tot, resid


# ---------------------------------------------------------------------------------------------
# 4. КОМАНДЫ
# ---------------------------------------------------------------------------------------------


def print_unparsed(unparsed, extra=()):
    W = 78
    print()
    print("=" * W)
    print("НЕ РАЗОБРАНО  --  читать ОБЯЗАТЕЛЬНО")
    print("=" * W)
    items = list(unparsed) + list(extra)
    if not items:
        print("  (пусто)")
    else:
        for s in items:
            print("  * " + s)
    print()
    print("  Пустой список подозрений при НЕПУСТОМ списке выше НЕ означает 'чисто':")
    print(
        "  неполный инструмент даёт не меньше данных, а ДРУГОЙ ОТВЕТ с той же уверенностью."
    )


def cmd_scan(spec, args):
    phases, unparsed = scan_sources(spec["sources"])
    print("ФАЗЫ, НАЙДЕННЫЕ В РАЗМЕТКЕ (%d):" % len(phases))
    print("  %-14s %4s %6s  %s" % ("имя", "id", "маска", "где"))
    for p in sorted(phases, key=lambda x: x.id):
        print(
            "  %-14s %4d %6d  %s:%d-%d  %s%s"
            % (
                p.name,
                p.id,
                p.bit,
                os.path.basename(p.path),
                p.l0,
                p.l1,
                "пломба" if p.sealed else "БЕЗ ПЛОМБЫ",
                ", подстановка" if p.has_else else "",
            )
        )
    print_unparsed(unparsed)
    return phases, unparsed


def cmd_static(spec, args):
    nvdisasm = spec.get("nvdisasm", "nvdisasm")
    phases, unparsed = scan_sources(spec["sources"])
    phases = sorted(phases, key=lambda x: x.id)
    if not phases:
        print("фаз нет -- нечего разбирать")
        print_unparsed(unparsed)
        return
    uniq = {}
    for p in phases:
        uniq.setdefault(p.name, p)
    names = [p.name for p in sorted(uniq.values(), key=lambda x: x.id)]
    kf = spec.get("kernel")

    print("=" * 78)
    print("СТАТИЧЕСКИЙ РАЗБОР SASS ПО ФАЗАМ  (карта НЕ нужна)")
    print("=" * 78)
    print(
        "  Привязка через -lineinfo: у каждой команды SASS есть (файл, строка) исходника."
    )
    print(
        "  Это счёт КОМАНД, а не времени: витки циклов не взвешены, задержки не учтены."
    )
    print()

    rc, cub, log, cmd = build_variant(spec, 0, "base")
    if rc != 0:
        print("СБОРКА БАЗЫ НЕ УДАЛАСЬ:\n%s\n%s" % (cmd, log[-3000:]))
        return
    base_stats = ptxas_stats(log)
    sass, err = disasm(cub, nvdisasm)
    if sass is None:
        print("nvdisasm не смог: %s" % err[-500:])
        return
    base_counts, _ = attribute(sass, phases, kf)
    total = sum(base_counts.values())

    print(
        "  БАЗА: %d команд SASS, регистров %s, разлив %s Б"
        % (total, base_stats["regs"], base_stats["spill_st"])
    )
    print()
    print("  %-16s %10s %8s" % ("фаза", "команд", "доля"))
    print("  " + "-" * 40)
    named = 0
    for n in names:
        c = base_counts.get(n, 0)
        named += c
        print("  %-16s %10d %7.1f%%" % (n, c, 100.0 * c / max(total, 1)))
    print("  " + "-" * 40)
    print(
        "  %-16s %10d %7.1f%%" % ("ИТОГО названо", named, 100.0 * named / max(total, 1))
    )
    print(
        "  %-16s %10d %7.1f%%   <-- команды, не попавшие ни в одну фазу"
        % (
            "НЕ ОТНЕСЕНО",
            base_counts["НЕ ОТНЕСЕНО"],
            100.0 * base_counts["НЕ ОТНЕСЕНО"] / max(total, 1),
        )
    )
    print(
        "  %-16s %10d %7.1f%%   <-- у команды нет строки (пролог/эпилог компилятора)"
        % (
            "БЕЗ ПРИВЯЗКИ",
            base_counts["БЕЗ ПРИВЯЗКИ"],
            100.0 * base_counts["БЕЗ ПРИВЯЗКИ"] / max(total, 1),
        )
    )

    # ---- матрица СБИТОГО ХВОСТА
    print()
    print("=" * 78)
    print("ПРОВЕРКА ФАЛЬСИФИКАТОРОВ: не выбило ли снятие фазы X команды фазы Y")
    print("=" * 78)
    print("  Строка = что снято. Столбец = сколько команд у фазы ОСТАЛОСЬ.")
    print(
        "  Диагональ обязана упасть в ~0. ЛЮБОЙ заметный провал ВНЕ диагонали значит, что"
    )
    print(
        "  компилятор выбросил чужую работу -> доля снятой фазы ЗАВЫШЕНА, замер недействителен."
    )
    print()
    hdr = (
        "  %-14s" % "снято \\ фаза"
        + "".join("%9s" % n[:8] for n in names)
        + "%9s%8s" % ("всего", "рег")
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    print(
        "  %-14s" % "(база)"
        + "".join("%9d" % base_counts.get(n, 0) for n in names)
        + "%9d%8s" % (total, base_stats["regs"])
    )
    cascades = []
    var_stats = {}
    for p in sorted(uniq.values(), key=lambda x: x.id):
        rc, cub, log, cmd = build_variant(spec, p.bit, "strip_%s" % p.name)
        if rc != 0:
            cascades.append(
                "вариант со снятой %s НЕ СОБРАЛСЯ -- фаза выпала из разбора" % p.name
            )
            print("  %-14s  СБОРКА НЕ УДАЛАСЬ" % p.name)
            continue
        st = ptxas_stats(log)
        var_stats[p.name] = st
        s2, _ = disasm(cub, nvdisasm)
        if s2 is None:
            cascades.append("nvdisasm не разобрал вариант %s" % p.name)
            continue
        c2, _ = attribute(s2, phases, kf)
        t2 = sum(c2.values())
        row = (
            "  %-14s" % p.name
            + "".join("%9d" % c2.get(n, 0) for n in names)
            + "%9d%8s" % (t2, st["regs"])
        )
        print(row)
        for n in names:
            if n == p.name:
                continue
            b, a = base_counts.get(n, 0), c2.get(n, 0)
            if b >= 8 and a < 0.8 * b:
                cascades.append(
                    "СБИТЫЙ ХВОСТ: снятие '%s' убрало %.0f%% команд фазы '%s' "
                    "(%d -> %d) -- доля '%s' будет ЗАВЫШЕНА; поставьте пломбу "
                    "FMHA_SEAL на выходы '%s' и РАЗЛИЧНУЮ подстановку "
                    "FMHA_PHASE_ELSE (одинаковые значения склеит CSE)"
                    % (p.name, 100.0 * (b - a) / b, n, b, a, p.name, p.name)
                )
        if (
            base_stats["regs"]
            and st["regs"]
            and abs(st["regs"] - base_stats["regs"]) > 0.15 * base_stats["regs"]
        ):
            cascades.append(
                "ПОДМЕНА РАСПИСАНИЯ: у варианта '%s' регистров %d против %d у базы "
                "(>15%%). Сравнение времён меряет уже не фазу, а другое распределение "
                "регистров -- см. §3b 'падение числа регистров при росте времени'"
                % (p.name, st["regs"], base_stats["regs"])
            )
        if st["spill_st"] and not base_stats["spill_st"]:
            cascades.append(
                "у варианта '%s' появился РАЗЛИВ (%d Б), у базы его нет"
                % (p.name, st["spill_st"])
            )

    # ---- цена пломбы
    seal_note = []
    if spec.get("seal_off"):
        rc, cub, log, _ = build_variant(spec, 0, "base_noseal", extra=spec["seal_off"])
        if rc == 0:
            s3, _ = disasm(cub, nvdisasm)
            if s3 is not None:
                c3, _ = attribute(s3, phases, kf)
                t3 = sum(c3.values())
                seal_note.append(
                    "ЦЕНА ПЛОМБЫ: %d -> %d команд (%+.2f %%). Пломба стоит по одной "
                    "команде PRMT на слово и присутствует В ОБОИХ вариантах, поэтому "
                    "из отношения времён она в первом порядке сокращается."
                    % (t3, total, 100.0 * (total - t3) / max(t3, 1))
                )
        else:
            seal_note.append(
                "сборка 'база без пломбы' не удалась -- цена пломбы НЕ ИЗМЕРЕНА"
            )
    else:
        seal_note.append("в spec нет ключа seal_off -- цена пломбы НЕ ИЗМЕРЕНА")
    print()
    for s in seal_note:
        print("  " + s)

    print_unparsed(
        unparsed,
        cascades
        + [
            "статический счёт НЕ взвешивает витки циклов: команда в теле главного цикла и команда в "
            "прологе весят здесь ОДИНАКОВО. Доли по времени берутся только из режима `time`.",
            "команды, поднятые планировщиком через границу фазы, привязка относит к ИСХОДНОЙ строке, "
            "а стоят они в чужой фазе (замерено: LOP3 уезжает на ~16 позиций).",
        ],
    )


def cmd_time(spec, args):
    phases, unparsed = scan_sources(spec["sources"])
    uniq = {}
    for p in phases:
        uniq.setdefault(p.name, p)
    plist = sorted(uniq.values(), key=lambda x: x.id)
    if not plist:
        print("фаз нет")
        print_unparsed(unparsed)
        return

    def measure(mask, tag):
        rc, out, log, cmd = build_variant(spec, mask, tag)
        if rc != 0:
            return None, "сборка не удалась: %s" % log[-800:]
        rc, txt = run(spec["run"].format(out=out, mask=mask, tag=tag))
        if rc != 0:
            return None, "запуск не удался: %s" % txt[-800:]
        m = re.findall(r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)", txt.strip().splitlines()[-1])
        if not m:
            return None, "не нашёл числа в выводе: %r" % txt[-200:]
        return float(m[-1]), ""

    fails = []
    t0, err = measure(0, "base")
    if t0 is None:
        print("БАЗА НЕ ИЗМЕРЕНА: " + err)
        return
    ph = {}
    for p in plist:
        v, err = measure(p.bit, "strip_%s" % p.name)
        if v is None:
            fails.append("фаза %s НЕ ИЗМЕРЕНА: %s" % (p.name, err))
        else:
            ph[p.name] = v
    allmask = 0
    for p in plist:
        allmask |= p.bit
    allt, err = measure(allmask, "strip_all")
    if allt is None:
        fails.append(
            "вариант 'снять всё' НЕ ИЗМЕРЕН: %s -- невязку разделить нечем" % err
        )
    pairs = {}
    if args.pairs:
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                a, b = plist[i], plist[j]
                v, err = measure(a.bit | b.bit, "strip_%s_%s" % (a.name, b.name))
                if v is None:
                    fails.append("пара %s+%s НЕ ИЗМЕРЕНА: %s" % (a.name, b.name, err))
                else:
                    pairs["%s+%s" % (a.name, b.name)] = v

    report_shares(
        t0,
        ph,
        allt,
        pairs,
        spec.get("units", "ms"),
        note=spec.get("name", ""),
        anchor=spec.get("anchor"),
    )
    if args.save:
        json.dump(
            {
                "base": t0,
                "phases": ph,
                "all": allt,
                "pairs": pairs,
                "units": spec.get("units", "ms"),
                "note": spec.get("name", ""),
            },
            open(args.save, "w"),
            ensure_ascii=False,
            indent=1,
        )
        print("\n  записано: %s" % args.save)
    print_unparsed(unparsed, fails)


def cmd_replay(args):
    obj = json.load(open(args.times, encoding="utf-8"))
    t0, ph, allt, pairs = load_times(obj)
    report_shares(
        t0,
        ph,
        allt,
        pairs,
        obj.get("units", "ms"),
        obj.get("note", ""),
        anchor=obj.get("anchor"),
    )
    print_unparsed(
        obj.get("unparsed", []),
        [
            "режим replay НИЧЕГО не собирает и не проверяет: он разбирает ЧУЖИЕ числа. "
            "Проверки фальсификаторов (сбитый хвост, подмена расписания) НЕ выполнялись."
        ],
    )


# ==============================================================================================
# САМОПРОВЕРКА.  Карта не нужна: проверяется АРИФМЕТИКА разложения и ОБЯЗАТЕЛЬНОСТЬ оговорки.
# ==============================================================================================
def selftest(out=print):
    ok = bad = 0

    def chk(label, cond, detail=""):
        nonlocal ok, bad
        out(
            "  %-9s %s%s"
            % ("ПРОЙДЕН" if cond else "ПАДЁТ", label, ("  " + detail) if detail else "")
        )
        if cond:
            ok += 1
        else:
            bad += 1

    out("САМОПРОВЕРКА ФАЗОВОГО ПРОФИЛИРОВЩИКА (карта не нужна)")

    # ЯКОРЬ 1. Арифметика доли: доля = 1 - снято/база, и ускорение 2x = доля 50 %.
    chk("1. доля фазы = 1 - снято/база", abs(share(2.0, 1.0) - 0.5) < 1e-12)

    # ЯКОРЬ 2. ЗАМЕРЕННАЯ ТРОЙКА EV_additivity.md ВОСПРОИЗВОДИТСЯ разложением.
    # 32.07 + 17.02 = 49.09 по отдельности, 43.36 вместе -> завышение +13.2 %.
    a, b, both = 0.3207, 0.1702, 0.4336
    over = 100.0 * ((a + b) / both - 1.0)
    chk(
        "2. замеренная тройка даёт завышение +13.2 %",
        abs(over - 13.2) < 0.1,
        "%.1f %%" % over,
    )
    chk(
        "2-бис. коридор оговорки НАКРЫВАЕТ замеренное завышение",
        SUBADD_LO <= over <= SUBADD_HI + 0.3,
        "%.1f %% против объявленных %.0f-%.0f %%" % (over, SUBADD_LO, SUBADD_HI),
    )

    # ЯКОРЬ 3. ГЛАВНЫЙ (LAW=L-PHASE-SUBADDITIVE): набор долей НЕ ПЕЧАТАЕТСЯ БЕЗ ОГОВОРКИ.
    # Требование было записано в журнале наблюдателя и НЕ ИСПОЛНЕНО -- эта проверка падает,
    # если оговорку убрать, и потому она и есть закон, а не пожелание.
    import io as _io
    import contextlib as _cl

    buf = _io.StringIO()
    with _cl.redirect_stdout(buf):
        report_shares(1.0, {"альфа": 0.68, "бета": 0.83}, None, {}, units="мс")
    txt = buf.getvalue()
    chk(
        "3. набор долей НЕ печатается без оговорки о верхней оценке",
        "ВЕРХНЯЯ ОЦЕНКА" in txt and "СУБаддитивны" in txt and "8-13 %" in txt,
        "оговорка на месте" if "ВЕРХНЯЯ ОЦЕНКА" in txt else "ОГОВОРКИ НЕТ",
    )
    chk(
        "3-бис. рядом с суммой напечатан её ЧЕСТНЫЙ интервал",
        "ПОПРАВКА К НАПЕЧАТАННОЙ СУММЕ" in txt,
    )
    chk(
        "3-в. названа ОБЛАСТЬ (одно тело, две фазы из пяти)",
        "ОДНОМ теле" in txt and "ДВУХ фазах" in txt,
    )

    # ЯКОРЬ 4. Тождество разложения невязки: названо + перекрытие + неназванное = 1.
    t0, ph, allt = 1.0, {"a": 0.7, "b": 0.85}, 0.55
    tot = sum(share(t0, v) for v in ph.values())
    s_all = share(t0, allt)
    chk(
        "4. названо + перекрытие + неназванное = 1",
        abs(tot + (s_all - tot) + (1 - s_all) - 1.0) < 1e-12,
    )

    out("ИТОГ САМОПРОВЕРКИ: пройдено %d, упало %d" % (ok, bad))
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description="фазовый профилировщик ядер sm_70")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("scan", "static", "time"):
        p = sub.add_parser(c)
        p.add_argument("--spec", required=True)
        if c == "time":
            p.add_argument(
                "--pairs", action="store_true", help="мерить и ПАРЫ (перекрытие)"
            )
            p.add_argument("--save", default=None)
    p = sub.add_parser("replay")
    p.add_argument("--times", required=True)
    sub.add_parser("selftest")
    args = ap.parse_args()

    if args.cmd == "selftest":
        sys.exit(selftest())
    if args.cmd == "replay":
        cmd_replay(args)
        return
    spec = json.load(open(args.spec, encoding="utf-8"))
    base = os.path.dirname(os.path.abspath(args.spec))
    spec["sources"] = [
        s if os.path.isabs(s) else os.path.join(base, s) for s in spec["sources"]
    ]
    {"scan": cmd_scan, "static": cmd_static, "time": cmd_time}[args.cmd](spec, args)


if __name__ == "__main__":
    main()
