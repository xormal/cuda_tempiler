# -*- coding: utf-8 -*-
"""КАЛИБРОВКА МОДЕЛИ: замеренные СТАВКИ вместо зашитых, и ОТЧЁТ О НЕВЯЗКЕ.

ЗАЧЕМ ЭТО КОМПИЛЯТОРУ, А НЕ ОТЧЁТУ
==================================
Компилятор минимального временного отпечатка укладывает операции в решётку (канал, такт). Чтобы
укладывать, ему нужна ЦЕНА ЗАНЯТИЯ КАНАЛА. Она НЕ КОНСТАНТА: замерено (ncu, 17/17), что при
неизменных 128 уникальных байтах один только шаг раскладки двигает цену в 32 РАЗА
(2048 -> 64.8 такта). Значит у модели обязана быть КАЛИБРОВОЧНАЯ ПОВЕРХНОСТЬ -- цена канала как
функция (раскладка, карта доступа), снятая ЗАМЕРОМ. Перебор дополнений порождает эту поверхность
по одной кривой на массив; фазовое разложение даёт независимую проверку -- ДОЛЮ канала на фазу,
то есть цель, к которой укладка обязана сойтись.

Отсюда три вещи, которые делает этот модуль:
    1. ФОРМАТ  data/calib/*.json -- одна схема для кривых дополнения и для разложений по фазам;
    2. ЗАГРУЗЧИК -- tempo.py берёт ставки отсюда, а не из своих констант, и ПЕЧАТАЕТ, какие
       ставки замерены, а какие остались модельными;
    3. ОТЧЁТ О НЕВЯЗКЕ -- где модель расходится с калибровкой и на сколько. Это главная ценность:
       расхождение модели с замером есть указание, ЧЕГО МОДЕЛЬ НЕ ЗНАЕТ.

ЧЕГО ЗДЕСЬ НАМЕРЕННО НЕТ
========================
В data/calib/ не кладутся ПРЕДСКАЗАНИЯ модели -- только замеры. Иначе модель калибруется сама
собой, невязка обнуляется по построению и инструмент начинает подтверждать что угодно. Модельная
кривая считается на лету (predict_curve) и живёт только в отчёте о невязке, рядом с замеренной.

СТАТУС НАБЛЮДАЕМОСТИ -- ОБЯЗАТЕЛЕН У КАЖДОЙ ЗАПИСИ (DESIGN_OBSERVER.md §3)
=========================================================================
    без_возмущения -- счётчик, не требующий правки кода (вайвфронты, конфликты, регистры);
    только_парами  -- величина, для которой надо СОБРАТЬ вариант (доли фаз): зонд входит в
                      расписание, поэтому годится лишь пара одинаковой структуры;
    неразделимо    -- зонд неотделим от предмета; величина по отдельности НЕ ОПРЕДЕЛЕНА.
Оценка без статуса -- это заявка на точность, которой у неё нет, поэтому поле обязательное, и
для kind="phases" статус "без_возмущения" ОТВЕРГАЕТСЯ воротами.

ФОРМАТ ЗАПИСИ (schema = "tempo/calib/1")
========================================
    {
      "schema": "tempo/calib/1",
      "id": "<совпадает с именем файла без .json>",
      "kind": "padcurve|phases|law_points|wf_per_inst|conflict_share|rate",
      "quantity": "<ЧТО откалибровано -- объект, а не 'скорость'>",
      "units": "<в чём число>",
      "taken_with": {"tool": "...", "counter": "...", "cmd": "...", "version": "..."},
      "shape": {"kernel": "...", "B":1,"H":2,"S":512,"D":128, ...},
      "card": {"index": 1|null, "foreign_procs": 0, "note": "..."},
      "observability": "без_возмущения|только_парами|неразделимо",
      "observability_why": "...",
      "provenance": {"source": ["data/...", "README.md:383"], "date": "...", "who": "..."},
      "binds": [ {"symbol": "MIO_BYTES_PER_CYCLE", "value": 128.0, "note": "..."} ],
      "payload": { <по виду записи> }
    }

payload по видам:
  padcurve      {"array","stride_words_base","elem_bytes","access":{...},
                 "curve":[{"pad","wavefronts","conflicts","wf_per_inst","smem_bytes"}...],
                 "stop":{"reason","pads_tried"}, "winner_pad": int}
  phases        {"base","shares":{имя:доля},"s_all","overlap","unnamed","pairs":{"a+b":e_ij}}
  law_points    {"points":[{"body","warps","width_bytes","dup","stride_bytes","ncu_wf"}...]}
  wf_per_inst   {"rows":[{"kernel","mix","model_nominal","model_safe","measured"}...]}
  conflict_share{"rows":[{"kernel","wavefronts","conflicts","fraction"}...]}
  rate          {"note": "..."}   -- вся суть в binds

КОМАНДЫ
-------
    calib.py validate            ворота формата, код возврата != 0 при ошибке
    calib.py list                что откалибровано и каким статусом
    calib.py ingest [--force]    завести записи из уже снятых замеров (ноль тактов GPU)
    calib.py residual            ГЛАВНОЕ: где модель расходится с калибровкой и на сколько
    calib.py predict ...         модельная кривая дополнения (без карты)
    calib.py selftest            самопроверка инструмента (ворота обязаны срабатывать)
"""

import argparse
import datetime
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPO = os.path.dirname(HERE)
CALIB_DIR = os.path.join(TEMPO, "data", "calib")

SCHEMA = "tempo/calib/1"
KINDS = ("padcurve", "phases", "law_points", "wf_per_inst", "conflict_share", "rate")
OBSERVABILITY = ("без_возмущения", "только_парами", "неразделимо")

BANKS = 32
WORD = 4
LANES = 32

# СИМВОЛЫ МОДЕЛИ, которые калибровка имеет право подменять. Список закрытый НАМЕРЕННО: запись,
# привязанная к неизвестному символу, -- это ставка, которая никуда не попадёт, и молчаливое
# "калибровка применена" при неприменённой ставке хуже отсутствия калибровки.
KNOWN_SYMBOLS = (
    "MIO_BYTES_PER_CYCLE",
    "MIO_WAVEFRONT_BYTES",
    "SCHEDULERS",
    "REG_ISA_LIMIT",
    "REG_OVERHEAD",
)
SYMBOL_PREFIXES = (
    "CAP_PER_SCHED.",
    "LATENCY.",
    "MIO_WF_PER_INST.",
    "MIO_CONFLICT_SHARE.",
)


# ================================================================================================
# 1. ЗАКОН (тот же, что у smem_lint.cost -- намеренно ОДНА реализация на два инструмента)
# ================================================================================================
def _load_cost():
    """Берём cost() из smem_lint, чтобы закон не разъехался в двух местах."""
    sys.path.insert(0, HERE)
    try:
        import smem_lint

        return smem_lint.cost
    except Exception:
        return None


def cost_fallback(addr_words, width_bytes):
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


COST = _load_cost() or cost_fallback


def predict_curve(
    stride_words,
    width_bytes,
    lanes_per_row=1,
    lane_step_words=None,
    pads=range(32),
    nlanes=LANES,
):
    """МОДЕЛЬНАЯ кривая дополнения: pad -> вайвфронты. Ноль тактов GPU.

    Карта доступа задана двумя числами: сколько полос обслуживают одну строку (lanes_per_row) и
    с каким шагом они идут вдоль строки (lane_step_words). Крайние случаи:
        lanes_per_row = 1  -- обход ПО СТОЛБЦУ: каждая полоса на своей строке; дополнение решает всё;
        lanes_per_row = 32 -- обход ПО СТРОКЕ: шаг строки не участвует, дополнение бессильно.
    """
    if lane_step_words is None:
        lane_step_words = max(1, int(width_bytes) // WORD)
    out = []
    for pad in pads:
        st = stride_words + pad
        addrs = [
            (l // lanes_per_row) * st + (l % lanes_per_row) * lane_step_words
            for l in range(nlanes)
        ]
        c = COST(addrs, width_bytes)
        if c is None:
            continue
        degree, floor, wf, rec, nact = c
        out.append(
            {"pad": pad, "degree": degree, "floor": floor, "wf": wf, "recoverable": rec}
        )
    return out


def curve_argmin(curve, key="wf"):
    if not curve:
        return None
    best = min(curve, key=lambda p: (p[key], p["pad"]))
    return best


# ================================================================================================
# 2. ВОРОТА ФОРМАТА
# ================================================================================================
def validate(rec, fname=None):
    """-> (ошибки, предупреждения). Ошибка = запись НЕГОДНА, ставки из неё не берутся."""
    E, W = [], []

    def need(k, typ=None):
        if k not in rec or rec[k] is None:
            E.append("нет обязательного поля %r" % k)
            return False
        if typ is not None and not isinstance(rec[k], typ):
            E.append("поле %r не того вида: %s" % (k, type(rec[k]).__name__))
            return False
        return True

    if rec.get("schema") != SCHEMA:
        E.append("schema %r, ожидалось %r" % (rec.get("schema"), SCHEMA))
    for k in ("id", "quantity", "units", "observability_why"):
        need(k, str)
    if need("kind", str) and rec["kind"] not in KINDS:
        E.append("kind %r неизвестен (знаем %s)" % (rec["kind"], ", ".join(KINDS)))
    if fname and rec.get("id") and rec["id"] != fname:
        E.append("id %r не совпадает с именем файла %r" % (rec["id"], fname))

    # ЧЕМ СНЯТО
    if need("taken_with", dict):
        if not rec["taken_with"].get("tool"):
            E.append("taken_with.tool пуст: 'чем снято' -- обязательное поле")
    # НА КАКОЙ ФОРМЕ
    if need("shape", dict):
        if not rec["shape"]:
            E.append("shape пуст: форма замера -- обязательное поле")
    # СКОЛЬКО ЧУЖИХ ПРОЦЕССОВ БЫЛО НА КАРТЕ
    if need("card", dict):
        fp = rec["card"].get("foreign_procs", "ОТСУТСТВУЕТ")
        if fp == "ОТСУТСТВУЕТ":
            E.append("card.foreign_procs отсутствует: без него замер невоспроизводим")
        elif isinstance(fp, str):
            if fp != "не применимо":
                E.append(
                    "card.foreign_procs -- строка %r; допустимо только 'не применимо'"
                    % fp
                )
        elif not isinstance(fp, int) or fp < 0:
            E.append("card.foreign_procs = %r -- нужно целое >= 0" % (fp,))
        elif fp > 0:
            W.append(
                "на карте было %d чужих процессов: счётчики устойчивы, ВРЕМЯ -- нет"
                % fp
            )
    # СТАТУС НАБЛЮДАЕМОСТИ
    obs = rec.get("observability")
    if obs not in OBSERVABILITY:
        E.append("observability %r не из набора %s" % (obs, ", ".join(OBSERVABILITY)))
    if rec.get("kind") == "phases" and obs == "без_возмущения":
        E.append(
            "kind=phases со статусом 'без_возмущения': доли фаз ТРЕБУЮТ сборки варианта, "
            "то есть зонд входит в расписание (DESIGN_OBSERVER.md §3)"
        )
    if need("provenance", dict):
        if not rec["provenance"].get("source"):
            E.append("provenance.source пуст: неоткуда перепроверить")

    # ПРИВЯЗКИ
    for b in rec.get("binds") or []:
        if not isinstance(b, dict) or "symbol" not in b or "value" not in b:
            E.append("запись binds без symbol/value: %r" % (b,))
            continue
        s = b["symbol"]
        if s not in KNOWN_SYMBOLS and not any(s.startswith(p) for p in SYMBOL_PREFIXES):
            E.append(
                "binds: символ %r модели неизвестен -- ставка НИКУДА не попадёт" % s
            )
        if not isinstance(b["value"], (int, float)):
            E.append("binds[%s].value не число: %r" % (s, b["value"]))

    # ТЕЛО ПО ВИДАМ
    p = rec.get("payload")
    if not isinstance(p, dict):
        E.append("payload отсутствует или не словарь")
        return E, W
    k = rec.get("kind")
    if k == "padcurve":
        E2, W2 = _validate_padcurve(p)
        E += E2
        W += W2
    elif k == "phases":
        E2, W2 = _validate_phases(p)
        E += E2
        W += W2
    elif k == "law_points":
        if not p.get("points"):
            E.append("law_points без points")
    elif k == "wf_per_inst":
        for r in p.get("rows") or []:
            if r.get("measured") is None:
                E.append("wf_per_inst: строка %r без measured" % r.get("kernel"))
    elif k == "conflict_share":
        for r in p.get("rows") or []:
            wf, cf = r.get("wavefronts"), r.get("conflicts")
            if wf is None or cf is None:
                # Доля БЕЗ абсолютных счётчиков законна (так записан исходный замер), но она
                # НЕПРОВЕРЯЕМА арифметикой -- отношение нельзя пересчитать. Это предупреждение,
                # а не отказ, и оно обязано быть видно.
                if r.get("fraction") is None:
                    E.append(
                        "conflict_share: у строки %r нет ни счётчиков, ни доли"
                        % r.get("kernel")
                    )
                else:
                    W.append(
                        "%r: только доля %.3f, абсолютных счётчиков нет -- отношение "
                        "арифметически НЕ ПРОВЕРЯЕМО" % (r.get("kernel"), r["fraction"])
                    )
            elif not (0 <= cf <= wf):
                E.append(
                    "conflict_share: %r конфликтов %g при вайвфронтах %g"
                    % (r.get("kernel"), cf, wf)
                )
    return E, W


def _validate_padcurve(p):
    E, W = [], []
    for k in ("array", "stride_words_base", "curve"):
        if p.get(k) is None:
            E.append("padcurve без %r" % k)
    cur = p.get("curve") or []
    if not cur:
        E.append("padcurve с пустой кривой")
        return E, W
    pads = [c.get("pad") for c in cur]
    if 0 not in pads:
        E.append(
            "в кривой НЕТ точки pad=0: без отгруженной раскладки кривой не с чем сравнивать"
        )
    if len(set(pads)) != len(pads):
        E.append("в кривой повторяются значения pad: %r" % pads)
    for c in cur:
        if not isinstance(c.get("pad"), int) or not (0 <= c["pad"] <= 31):
            E.append("pad вне 0..31: %r" % (c.get("pad"),))
        wf, cf = c.get("wavefronts"), c.get("conflicts")
        if wf is None:
            E.append("точка pad=%r без wavefronts" % c.get("pad"))
        elif cf is not None and not (0 <= cf <= wf + 1e-9):
            E.append(
                "pad=%r: конфликтов %g при вайвфронтах %g" % (c.get("pad"), cf, wf)
            )
        if c.get("smem_bytes") is None:
            W.append(
                "pad=%r без smem_bytes: ворота занятости по этой точке НЕ проверены"
                % c.get("pad")
            )
    win = p.get("winner_pad")
    if win is not None:
        best = min(cur, key=lambda c: (c.get("wavefronts", float("inf")), c["pad"]))
        if best["pad"] != win:
            E.append(
                "winner_pad=%r, а минимум кривой на pad=%r: победитель ОБЪЯВЛЕН, а не найден"
                % (win, best["pad"])
            )
    st = p.get("stop") or {}
    if not st.get("reason"):
        W.append("нет stop.reason: неизвестно, перебор исчерпан или оборван")
    return E, W


def _validate_phases(p):
    E, W = [], []
    sh = p.get("shares")
    if not isinstance(sh, dict) or not sh:
        E.append("phases без shares")
        return E, W
    ssum = sum(sh.values())
    p_sum = p.get("sum_named")
    if p_sum is not None and abs(p_sum - ssum) > 1e-6:
        E.append("sum_named=%g не равен сумме долей %g" % (p_sum, ssum))
    if p.get("s_all") is None:
        W.append(
            "нет варианта 'снять ВСЕ фазы': невязка %.1f%% НЕ РАЗДЕЛЕНА на перекрытие и "
            "неназванное -- доли суть НИЖНИЕ оценки" % (100 * (1 - ssum))
        )
    else:
        ov = p["s_all"] - ssum
        if p.get("overlap") is not None and abs(p["overlap"] - ov) > 1e-6:
            E.append("overlap=%g, а s_all-SUM=%g" % (p["overlap"], ov))
    if not p.get("pairs"):
        W.append("пар нет: поточечное перекрытие e_ij не мерено")
    return E, W


# ================================================================================================
# 3. ЗАГРУЗКА
# ================================================================================================
def load_dir(path=None, strict=True):
    """-> (годные записи, {id: ошибки}, {id: предупреждения})."""
    path = path or CALIB_DIR
    recs, errs, warns = [], {}, {}
    if not os.path.isdir(path):
        return recs, {"<каталог>": ["нет каталога %s" % path]}, warns
    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        full = os.path.join(path, fn)
        try:
            rec = json.load(open(full, encoding="utf-8"))
        except Exception as ex:
            errs[fn] = ["не разобран JSON: %s" % ex]
            continue
        E, W = validate(rec, fname=fn[:-5])
        if W:
            warns[rec.get("id", fn)] = W
        if E:
            errs[rec.get("id", fn)] = E
            if strict:
                continue
        rec["_file"] = full
        recs.append(rec)
    return recs, errs, warns


def rates(recs):
    """-> {символ: {"value","from","observability","note"}}. Конфликт двух записей -- ОШИБКА."""
    out, clash = {}, []
    for r in recs:
        for b in r.get("binds") or []:
            s = b["symbol"]
            if s in out and abs(out[s]["value"] - b["value"]) > 1e-12:
                clash.append(
                    "%s: %g (из %s) против %g (из %s)"
                    % (s, out[s]["value"], out[s]["from"], b["value"], r["id"])
                )
                continue
            out[s] = {
                "value": float(b["value"]),
                "from": r["id"],
                "observability": r.get("observability"),
                "note": b.get("note", ""),
            }
    return out, clash


# ================================================================================================
# 4. ПИСАТЕЛЬСКИЙ ИНТЕРФЕЙС (его зовёт padsweep.py / phaseprof.py -- формат по построению)
# ================================================================================================
def _skeleton(
    rid,
    kind,
    quantity,
    units,
    taken_with,
    shape,
    card,
    observability,
    observability_why,
    provenance,
    binds=None,
    payload=None,
):
    return {
        "schema": SCHEMA,
        "id": rid,
        "kind": kind,
        "quantity": quantity,
        "units": units,
        "taken_with": taken_with,
        "shape": shape,
        "card": card,
        "observability": observability,
        "observability_why": observability_why,
        "provenance": provenance,
        "binds": binds or [],
        "payload": payload or {},
    }


def write(rec, out_dir=None, force=False):
    out_dir = out_dir or CALIB_DIR
    os.makedirs(out_dir, exist_ok=True)
    E, W = validate(rec, fname=rec.get("id"))
    if E:
        raise ValueError("запись не проходит ворота:\n  " + "\n  ".join(E))
    path = os.path.join(out_dir, rec["id"] + ".json")
    if os.path.exists(path) and not force:
        raise FileExistsError("уже есть %s (перезапись только с force=True)" % path)
    json.dump(rec, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return path, W


def emit_padcurve(
    rid,
    kernel,
    array,
    stride_words_base,
    curve,
    *,
    taken_with,
    shape,
    card,
    provenance,
    elem_bytes=None,
    access=None,
    stop=None,
    out_dir=None,
    force=False,
    note="",
    binds=None,
):
    """КРИВАЯ ДОПОЛНЕНИЯ -- главный вход для padsweep.py.

    curve -- список точек {"pad", "wavefronts", "conflicts", "wf_per_inst", "smem_bytes"}.
    winner_pad НЕ передаётся: он ВЫЧИСЛЯЕТСЯ как минимум кривой (объявленный победитель, не
    совпавший с минимумом, -- отдельный класс дефекта, см. ворота).
    """
    cur = sorted(curve, key=lambda c: c["pad"])
    win = min(cur, key=lambda c: (c["wavefronts"], c["pad"]))["pad"] if cur else None
    rec = _skeleton(
        rid,
        "padcurve",
        "цена занятия канала MIO массивом %s ядра %s как функция дополнения раскладки"
        % (array, kernel),
        "вайвфронтов (сумма по запуску) и вайвфронтов на команду",
        taken_with,
        shape,
        card,
        "без_возмущения",
        "счётчик вайвфронтов/конфликтов не требует правки кода; правится РАСКЛАДКА, а она и есть "
        "предмет измерения, а не зонд",
        provenance,
        binds=binds or [],
        payload={
            "array": array,
            "kernel": kernel,
            "stride_words_base": stride_words_base,
            "elem_bytes": elem_bytes,
            "access": access or {},
            "curve": cur,
            "winner_pad": win,
            "stop": stop or {},
            "note": note,
        },
    )
    return write(rec, out_dir, force)


def emit_phases(
    rid,
    kernel,
    shares,
    *,
    taken_with,
    shape,
    card,
    provenance,
    s_all=None,
    pairs=None,
    base=1.0,
    out_dir=None,
    force=False,
    note="",
    units="доля времени ядра",
):
    """РАЗЛОЖЕНИЕ ПО ФАЗАМ -- вход для phaseprof.py. Невязка раскладывается ЗДЕСЬ, а не у читателя."""
    ssum = sum(shares.values())
    pl = {
        "base": base,
        "shares": shares,
        "sum_named": ssum,
        "s_all": s_all,
        "pairs": pairs or {},
        "note": note,
    }
    if s_all is not None:
        pl["overlap"] = s_all - ssum
        pl["unnamed"] = 1.0 - s_all
    else:
        pl["overlap"] = None
        pl["unnamed"] = None
        pl["residual_undivided"] = 1.0 - ssum
    rec = _skeleton(
        rid,
        "phases",
        "доля канала, занятая каждой названной фазой ядра %s" % kernel,
        units,
        taken_with,
        shape,
        card,
        "только_парами",
        "доля фазы получается СБОРКОЙ варианта со снятой фазой: зонд входит в расписание, значит "
        "величина определена только вместе с обстановкой (DESIGN_OBSERVER.md §3)",
        provenance,
        payload=pl,
    )
    return write(rec, out_dir, force)


# ================================================================================================
# 5. ЗАВЕДЕНИЕ ЗАПИСЕЙ ИЗ УЖЕ СНЯТЫХ ЗАМЕРОВ (ноль тактов GPU)
# ================================================================================================
def _cite(path, needles, label):
    """ВОРОТА ССЫЛКИ: число, переписанное из файла, обязано в этом файле НАХОДИТЬСЯ.

    Иначе запись калибровки тихо начинает описывать прошлое состояние источника -- ровно тот класс
    дефекта, ради которого двойник ПОРОЖДАЮТ, а не ведут руками.
    """
    full = os.path.join(TEMPO, path)
    if not os.path.exists(full):
        return ["%s: нет файла-источника %s" % (label, path)]
    txt = open(full, encoding="utf-8", errors="replace").read()
    return [
        "%s: в %s больше НЕТ строки %r" % (label, path, n)
        for n in needles
        if n not in txt
    ]


def parse_mio_wavefronts(path=None):
    path = path or os.path.join(TEMPO, "data", "mio_wavefronts.txt")
    pts = []
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 11 or f[0] == "тело":
            continue
        try:
            body, W = f[0], int(f[1])
            width, dup, strb = int(f[2]), int(f[3]), int(f[4])
            conf, model_wf = int(f[6]), float(f[7])
            meas_time = float(f[9])
        except (ValueError, IndexError):
            continue
        ncu = f[11] if len(f) > 11 else "-"
        pts.append(
            {
                "body": body,
                "warps": W,
                "width_bytes": width,
                "dup": dup,
                "stride_bytes": strb,
                "conflict_degree": conf,
                "model_wf": model_wf,
                "measured_cycles": meas_time,
                "ncu_wf": (float(ncu) if ncu not in ("-", "") else None),
            }
        )
    return pts


def ingest(out_dir=None, force=False, verbose=True):
    """Заводит записи калибровки из данных, которые уже сняты. Карту НЕ трогает."""
    out_dir = out_dir or CALIB_DIR
    made, skipped, gate_fail = [], [], []
    today = datetime.date.today().isoformat()

    # --- (1) ЗАКОН ВАЙВФРОНТОВ: стенд probe.cu, 17 точек сверены ncu -----------------------------
    pts = parse_mio_wavefronts()
    ncu_pts = [p for p in pts if p["ncu_wf"] is not None]
    gate_fail += _cite(
        "data/mio_wavefronts.txt", ["чужих процессов 0 на всех точках"], "закон MIO"
    )
    rec = _skeleton(
        "mio_law_probe",
        "law_points",
        "цена команды разделяемой памяти в ВАЙВФРОНТАХ как функция (шаг раскладки, ширина, "
        "кратность адреса)",
        "вайвфронтов на команду",
        {
            "tool": "tools/probe.cu + ncu 2024.1.1",
            "counter": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld",
            "cmd": "tools/ncu.py (LC_ALL=C)",
            "version": "2024.1.1",
        },
        {
            "kernel": "probe_kernel<...>",
            "warps": sorted({p["warps"] for p in pts}),
            "widths_bytes": sorted({p["width_bytes"] for p in pts}),
            "strides_bytes": sorted({p["stride_bytes"] for p in pts}),
        },
        {
            "index": 1,
            "foreign_procs": 0,
            "note": "в шапке замера записано 'чужих процессов 0 на всех точках'",
        },
        "без_возмущения",
        "счётчик вайвфронтов снимается без правки тела; тело стенда синтетическое и меряет само себя",
        {
            "source": ["data/mio_wavefronts.txt", "data/probe_measured.json"],
            "date": today,
            "who": "tempo/probe.cu",
        },
        binds=[
            {
                "symbol": "MIO_WAVEFRONT_BYTES",
                "value": 128.0,
                "note": "вайвфронт несёт не более 128 Б; конвейер отдаёт 1 вайвфронт/такт/SM",
            },
            {
                "symbol": "MIO_BYTES_PER_CYCLE",
                "value": 128.0,
                "note": "тот же замер, выраженный в байтах на такт на весь SM",
            },
        ],
        payload={
            "law": "вайвфронтов/команду = max(конфликтность, ceil(уник.Б/128), ширина/8Б, 1)",
            "points": pts,
            "n_points": len(pts),
            "n_ncu": len(ncu_pts),
        },
    )
    made += _try_write(rec, out_dir, force, skipped, verbose)

    # --- (2) СТАВКА MIO НА БОЕВЫХ ЯДРАХ ---------------------------------------------------------
    gate_fail += _cite(
        "README.md", ["**2.054**", "**2.221**", "**2.250**"], "ставка MIO на боевых"
    )
    rows = [
        {
            "kernel": "volta_fwd_block",
            "mix": "8 Б:288, 16 Б:8",
            "model_nominal": 2.054,
            "model_safe": 1.027,
            "measured": 2.054,
        },
        {
            "kernel": "volta_fwd_ws (фаза 0)",
            "mix": "4 Б:66, 8 Б:86, 16 Б:22",
            "model_nominal": 1.874,
            "model_safe": 1.126,
            "measured": 2.221,
        },
        {
            "kernel": "volta_fwd_ws (фаза 7)",
            "mix": "4 Б:66, 8 Б:82, 16 Б:22",
            "model_nominal": 1.871,
            "model_safe": 1.129,
            "measured": 2.250,
        },
    ]
    rec = _skeleton(
        "mio_wf_per_inst_shipped",
        "wf_per_inst",
        "средняя цена ОДНОЙ команды разделяемой памяти в отгруженном ядре",
        "вайвфронтов на команду",
        {
            "tool": "ncu 2024.1.1 через tools/ncu.py",
            "counter": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld / число команд LDS",
            "version": "2024.1.1",
        },
        {
            "kernels": [r["kernel"] for r in rows],
            "note": "формы малые: цена -- свойство РАСКЛАДКИ",
        },
        {
            "index": None,
            "foreign_procs": "не применимо",
            "note": "число чужих процессов в исходном отчёте не записано; счётчик от соседа не зависит, "
            "поэтому запись годна, а ВРЕМЕННЫЕ выводы из неё -- нет",
        },
        "без_возмущения",
        "счётчик, правки кода не требует",
        {
            "source": ["README.md:383-385", "tools/tempo.py:133-137"],
            "date": today,
            "who": "аудит конфликтности банков",
        },
        binds=[
            {
                "symbol": "MIO_WF_PER_INST." + r["kernel"].split(" ")[0],
                "value": r["measured"],
                "note": "замер ncu; модельная номинальная ставка %g"
                % r["model_nominal"],
            }
            for r in rows[:2]
        ],
        payload={
            "rows": rows,
            "note": "volta_fwd_block: замер = номинальная ставка ДО ТРЕТЬЕГО ЗНАКА -- "
            "совпадающих адресов между полосами в наших ядрах НЕТ ВОВСЕ. "
            "volta_fwd_ws: замер ВЫШЕ ставки -- это банковые конфликты, которых "
            "модель не видит, значит ставка 'ширина*32' не является и верхней оценкой.",
        },
    )
    made += _try_write(rec, out_dir, force, skipped, verbose)

    # --- (3) ДОЛЯ КОНФЛИКТОВ ПО ОТГРУЖЕННЫМ ЯДРАМ ------------------------------------------------
    gate_fail += _cite(
        "data/ncu/fwd128_run.txt", ["20.82 %"], "доля конфликтов forward"
    )
    gate_fail += _cite(
        "data/ncu/selftest_ncu.txt", ["34.65 %"], "доля конфликтов backward"
    )
    gate_fail += _cite(
        "data/smem_lint_selftest.txt", ["19.4 %", "1.2 %"], "доли конфликтов ws/декод"
    )
    rows = [
        {
            "kernel": "attention_kernel_batched_impl (forward, cutlass)",
            "wavefronts": 1427260.0,
            "conflicts": 297090.0,
            "fraction": 0.2082,
            "where": "эпилог накопителя B2bGemm.accumToSmem, STS.64 с 8-кратным конфликтом",
        },
        {
            "kernel": "volta_fwd_ws (байтовый форвард)",
            "wavefronts": None,
            "conflicts": None,
            "fraction": 0.194,
            "where": "sQ (шаг 130 слов), sRedM, sRedS; плитка K (LDK8=68 слов) -- 74.5 % конфликтов ядра",
        },
        {
            "kernel": "attention_kernel_backward_batched_impl",
            "wavefronts": 4932840.0,
            "conflicts": 1709240.0,
            "fraction": 0.3465,
            "where": "тот же accumToSmem; 289 команд из 7424 трогают shared, верхние все STS.64 N-way 8",
        },
        {
            "kernel": "split_defer_mqa_kernel (декод, EPT=4)",
            "wavefronts": 11460.0,
            "conflicts": 132.0,
            "fraction": 0.012,
            "where": "ЧИСТО: вайвфронты есть, конфликтов нет",
        },
    ]
    rec = _skeleton(
        "conflict_share_shipped",
        "conflict_share",
        "доля трафика разделяемой памяти, добавленная БАНКОВЫМИ КОНФЛИКТАМИ (то есть возвратимая "
        "дополнением раскладки)",
        "доля вайвфронтов",
        {
            "tool": "tools/bankaudit.py + tools/ncu.py",
            "counter": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st} / вайвфронты",
            "version": "2024.1.1",
        },
        {
            "note": "малые формы намеренно: доля -- свойство раскладки, а не размера задачи",
            "fwd": "B1 H2 S512 D128 causal",
            "decode": "EPT=4",
        },
        {
            "index": None,
            "foreign_procs": "не применимо",
            "note": "счётчик конфликтов от соседа по карте не зависит",
        },
        "без_возмущения",
        "счётчики, правки кода не требуют",
        {
            "source": [
                "data/ncu/fwd128_run.txt",
                "data/ncu/selftest_ncu.txt",
                "data/smem_lint_selftest.txt",
                "data/smem_lint_verify.txt",
            ],
            "date": today,
            "who": "задача 105, аудит конфликтности банков",
        },
        binds=[
            {
                "symbol": "MIO_CONFLICT_SHARE."
                + (
                    "fwd_cutlass"
                    if i == 0
                    else "volta_fwd_ws"
                    if i == 1
                    else "bwd_cutlass"
                    if i == 2
                    else "split_defer_mqa"
                ),
                "value": r["fraction"],
                "note": r["where"],
            }
            for i, r in enumerate(rows)
        ],
        payload={
            "rows": rows,
            "note": "модель по умолчанию (конфликтность=1) предсказывает ноль конфликтов ВЕЗДЕ; "
            "невязка равна самой доле.",
        },
    )
    made += _try_write(rec, out_dir, force, skipped, verbose)

    # --- (4) ФАЗОВОЕ РАЗЛОЖЕНИЕ (якорь форварда) ------------------------------------------------
    src = os.path.join(TEMPO, "data", "anchor_fwd_ws_phases.json")
    if os.path.exists(src):
        a = json.load(open(src, encoding="utf-8"))
        shares = {k: v / 100.0 for k, v in a["anchor"].items()}
        rec = _skeleton(
            "phases_fwd_ws_anchor",
            "phases",
            "доля времени ядра, занятая каждой названной фазой байтового форварда",
            "доля (база = 1.0)",
            {
                "tool": "РУЧНАЯ сборка вариантов со снятой фазой (DIAG=1..5), до появления phaseprof.py",
                "counter": "секундомер",
                "version": "-",
            },
            {
                "kernel": "volta_fwd_ws",
                "D": 256,
                "KVFMT": 8,
                "note": "роли, байтовый KV",
            },
            {
                "index": None,
                "foreign_procs": "не применимо",
                "note": "исходный замер соседей не записывал -- ВРЕМЕННАЯ величина без этого поля "
                "недоказуема, и это отражено в статусе",
            },
            "только_парами",
            "снятие фазы меняет перекрытие и расписание; в ЭТОМ замере пар НЕТ и варианта 'снять "
            "ВСЕ' НЕТ, поэтому доли -- НИЖНИЕ оценки, а невязка 22.3 % НЕ РАЗДЕЛЕНА на перекрытие "
            "и неназванное",
            {
                "source": [a.get("source", ""), "data/anchor_fwd_ws_phases.json"],
                "date": today,
                "who": "docs/VOLTA_SM70.md §3b",
            },
            payload={
                "base": 1.0,
                "shares": shares,
                "sum_named": sum(shares.values()),
                "s_all": None,
                "overlap": None,
                "unnamed": None,
                "residual_undivided": 1.0 - sum(shares.values()),
                "pairs": {},
                "speedup": a.get("speedup", {}),
                "note": "; ".join(a.get("unparsed", [])),
            },
        )
        made += _try_write(rec, out_dir, force, skipped, verbose)

    # --- (5) СТАВКИ КАНАЛОВ И ЗАДЕРЖЕК СО СТЕНДА -------------------------------------------------
    gate_fail += _cite(
        "tools/tempo.py",
        ["REG_OVERHEAD = 7", "MIO_BYTES_PER_CYCLE = 128.0"],
        "ставки стенда",
    )
    rec = _skeleton(
        "probe_channels",
        "rate",
        "ёмкости каналов и задержки, снятые стендом probe.cu",
        "тактов на команду (на планировщик), кроме MIO -- байт/такт на SM",
        {
            "tool": "tools/probe.cu, лестницы c1..c32 и свипы занятости",
            "counter": "секундомер + ptxas",
            "version": "-",
        },
        {
            "gpu": "V100-32GB (sm_70)",
            "warps": [4, 8, 12, 16, 24, 32],
            "note": "643 точки, из них 17 сверены ncu",
        },
        {"index": 1, "foreign_procs": 0, "note": "стенд гоняли на свободной карте"},
        "без_возмущения",
        "ёмкость канала и задержка -- свойства машины; тело стенда меряет само себя, чужого зонда в "
        "нём нет. ИСКЛЮЧЕНИЕ: задержка HMMA (8) -- ОЦЕНКА СВЕРХУ, не замер, поэтому она НЕ привязана "
        "и остаётся модельной",
        {
            "source": [
                "tools/tempo.py:18-35",
                "data/probe_measured.json",
                "data/knee_fit.txt",
            ],
            "date": today,
            "who": "tempo/probe.cu",
        },
        binds=[
            {"symbol": "SCHEDULERS", "value": 4, "note": "четыре планировщика на SM"},
            {
                "symbol": "CAP_PER_SCHED.TENSOR",
                "value": 1.0,
                "note": "ЕДИНИЦА ёмкости (слот/такт на планировщик); сам ЗАМЕР -- 1.95 HMMA/такт/SM, "
                "то есть 2.05 такта на HMMA на планировщик, и он сидит в footprint() как 2.0",
            },
            {"symbol": "REG_ISA_LIMIT", "value": 255, "note": "потолок ISA"},
            {
                "symbol": "REG_OVERHEAD",
                "value": 7,
                "note": "требуемых регистров = MaxLive + 7 РОВНО (три точки, свидетель LDL/STL)",
            },
            {
                "symbol": "LATENCY.LDS",
                "value": 26.0,
                "note": "замер s1x32 при одном варпе",
            },
            {"symbol": "LATENCY.SHFL", "value": 26.0, "note": "та же лестница"},
            {"symbol": "LATENCY.FFMA", "value": 4.0, "note": "лестница c1..c32"},
            {"symbol": "LATENCY.IADD3", "value": 4.0, "note": "лестница c1..c32"},
        ],
        payload={
            "note": "ставка BRANCH=1 такт добавлена ПОСЛЕ нарушения границы на теле c1x16 "
            "(модель давала 76 при замеренных 72.1) -- это тоже калибровка, но она "
            "уже вшита в footprint()."
        },
    )
    made += _try_write(rec, out_dir, force, skipped, verbose)

    if verbose:
        for m in made:
            print("  ЗАВЕДЕНО   %s" % m)
        for s in skipped:
            print("  УЖЕ ЕСТЬ   %s" % s)
        for g in gate_fail:
            print("  ВОРОТА ССЫЛКИ НЕ ПРОШЛИ: %s" % g)
    return made, skipped, gate_fail


def _try_write(rec, out_dir, force, skipped, verbose):
    try:
        path, W = write(rec, out_dir, force)
    except FileExistsError:
        skipped.append(rec["id"])
        return []
    except ValueError as ex:
        print("  ОТКАЗ      %s: %s" % (rec["id"], ex))
        return []
    if verbose:
        for w in W:
            print("    предупреждение [%s]: %s" % (rec["id"], w))
    return [os.path.relpath(path, TEMPO)]


# ================================================================================================
# 6. ОТЧЁТ О НЕВЯЗКЕ -- ГЛАВНОЕ
# ================================================================================================
def _model_safe_wf(width_bytes):
    """Ставка tempo ПО УМОЛЧАНИЮ: конфликтность принята равной 1 (законная нижняя граница)."""
    return max(width_bytes / 8.0, 1.0)


def _model_nominal_wf(width_bytes):
    """Ставка 'все полосы различны, соседние слова'."""
    return max(width_bytes / 4.0, width_bytes / 8.0, 1.0)


def residual(path=None, verbose=True):
    """Где модель расходится с калибровкой и на сколько."""
    recs, errs, warns = load_dir(path)
    by = {r["id"]: r for r in recs}
    out = {"sections": [], "errors": errs, "warnings": warns}
    P = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    P("=" * 108)
    P(
        "НЕВЯЗКА: МОДЕЛЬ tempo ПРОТИВ КАЛИБРОВКИ  (расхождение = указание, ЧЕГО МОДЕЛЬ НЕ ЗНАЕТ)"
    )
    P("=" * 108)

    # --- 1. ЗАКОН НА СТЕНДЕ -----------------------------------------------------------------
    r = by.get("mio_law_probe")
    if r:
        pts = [p for p in r["payload"]["points"] if p["ncu_wf"] is not None]
        e_safe, e_law = [], []
        worst = None
        for p in pts:
            w, dup, strb = p["width_bytes"], p["dup"], p["stride_bytes"]
            meas = p["ncu_wf"]
            safe = _model_safe_wf(w)
            addrs = (
                [((l // dup) * strb) // WORD for l in range(LANES)]
                if strb % WORD == 0
                else None
            )
            law = COST(addrs, w)[2] if addrs else float("nan")
            e_safe.append((meas - safe) / meas)
            if law == law:
                e_law.append(abs(law - meas) / meas)
            if worst is None or (meas - safe) / meas > worst[0]:
                worst = ((meas - safe) / meas, p["body"], safe, meas)
        med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
        P("")
        nbody = len({p["body"] for p in pts})
        P(
            "1. КАНАЛ MIO НА СТЕНДЕ  (%d строк со счётчиком ncu = %d тел x занятости, из %d строк "
            "стенда; закон -- max(конфликтность, ceil(уник.Б/128), ширина/8, 1))"
            % (len(pts), nbody, len(r["payload"]["points"]))
        )
        P("   ставка модели ПО УМОЛЧАНИЮ (конфликтность=1, законная граница):")
        P(
            "       медиана занижения %.1f %%, максимум %.1f %% (тело %s: модель %.3g, замер %.3g)"
            % (100 * med(e_safe), 100 * worst[0], worst[1], worst[2], worst[3])
        )
        P("   закон С КОНФЛИКТНОСТЬЮ (то, что модель УМЕЕТ, если раскладка известна):")
        P(
            "       медиана |невязки| %.3f %%, максимум %.3f %%  ->  %s"
            % (
                100 * med(e_law),
                100 * max(e_law),
                "МОДЕЛЬ ЗНАЕТ ТОЧНО" if max(e_law) < 1e-6 else "расхождение",
            )
        )
        P(
            "   ЧИТАТЬ ТАК: модель не ошибается в законе -- она не знает ОДНОГО ЧИСЛА, конфликтности,"
        )
        P("   и именно его и порождает перебор дополнений.")
        out["sections"].append(
            {
                "id": "law",
                "n": len(pts),
                "median_safe_understated": med(e_safe),
                "max_safe_understated": worst[0],
                "max_law_err": max(e_law),
            }
        )

    # --- 2. БОЕВЫЕ ЯДРА, ЦЕНА КОМАНДЫ ------------------------------------------------------
    r = by.get("mio_wf_per_inst_shipped")
    if r:
        P("")
        P("2. КАНАЛ MIO В БОЕВЫХ ЯДРАХ  (цена ОДНОЙ команды разделяемой памяти)")
        P(
            "   %-24s %10s %10s %10s %10s"
            % ("ядро", "модель", "граница", "ЗАМЕР", "невязка")
        )
        rows = []
        for row in r["payload"]["rows"]:
            # НЕВЯЗКА МОДЕЛИ = (замер - модель)/модель: на сколько модель промахнулась ОТ СЕБЯ.
            d = (row["measured"] - row["model_nominal"]) / row["model_nominal"]
            P(
                "   %-24s %10.3f %10.3f %10.3f %+9.1f %%"
                % (
                    row["kernel"],
                    row["model_nominal"],
                    row["model_safe"],
                    row["measured"],
                    100 * d,
                )
            )
            rows.append({"kernel": row["kernel"], "residual": d})
        P(
            "   ЧИТАТЬ ТАК: там, где невязка ~0 (volta_fwd_block), совпадающих адресов между полосами"
        )
        P(
            "   НЕТ и модель точна. Где невязка положительна -- это БАНКОВЫЕ КОНФЛИКТЫ, и модельная"
        )
        P("   ставка 'ширина*32' не является даже верхней оценкой.")
        out["sections"].append({"id": "wf_per_inst", "rows": rows})

    # --- 3. ДОЛЯ КОНФЛИКТОВ ----------------------------------------------------------------
    r = by.get("conflict_share_shipped")
    if r:
        P("")
        P("3. ДОЛЯ КОНФЛИКТОВ ПО ЯДРАМ  (модель по умолчанию предсказывает 0 % ВЕЗДЕ)")
        P("   %-46s %10s %12s" % ("ядро", "ЗАМЕР", "невязка"))
        for row in r["payload"]["rows"]:
            P(
                "   %-46s %9.1f %% %10.1f п.п."
                % (row["kernel"][:46], 100 * row["fraction"], 100 * row["fraction"])
            )
        P(
            "   ЧИТАТЬ ТАК: это ВЕРХНЯЯ оценка возвратимого ПО ТРАФИКУ, а не по времени; во время она"
        )
        P("   переходит настолько, насколько разделяемая память связывает.")
        out["sections"].append(
            {
                "id": "conflict_share",
                "rows": [
                    {"kernel": x["kernel"], "fraction": x["fraction"]}
                    for x in r["payload"]["rows"]
                ],
            }
        )

    # --- 4. ФАЗЫ ----------------------------------------------------------------------------
    r = by.get("phases_fwd_ws_anchor")
    if r:
        p = r["payload"]
        P("")
        P(
            "4. ФАЗЫ  (модель НЕ предсказывает долей вовсе -- у неё нет понятия фазы; невязка = 100 %)"
        )
        for k, v in sorted(p["shares"].items(), key=lambda kv: -kv[1]):
            P("       %-12s %6.1f %%" % (k, 100 * v))
        P("   названо                %6.1f %%" % (100 * p["sum_named"]))
        if p.get("s_all") is None:
            P(
                "   НЕВЯЗКА                %6.1f %%  -- НЕ РАЗДЕЛЕНА: варианта 'снять ВСЕ' нет,"
                % (100 * p["residual_undivided"])
            )
            P(
                "                                   значит перекрытие и неназванное НЕ различены,"
            )
            P("                                   а сами доли -- НИЖНИЕ оценки.")
        else:
            P("   перекрытие             %6.1f %%" % (100 * p["overlap"]))
            P("   неназванное            %6.1f %%" % (100 * p["unnamed"]))
        P(
            "   ЦЕЛЬ ДЛЯ УКЛАДКИ: доля канала на фазу -- это то, к чему обязан сойтись компилятор."
        )
        P(
            "   Пока она снята БЕЗ пар, у модели нет цели, к которой можно подгоняться законно."
        )
        out["sections"].append(
            {
                "id": "phases",
                "named": p["sum_named"],
                "residual": p.get("residual_undivided"),
            }
        )

    # --- 5. КРИВЫЕ ДОПОЛНЕНИЯ ---------------------------------------------------------------
    curves = [x for x in recs if x["kind"] == "padcurve"]
    P("")
    P(
        "5. КАЛИБРОВОЧНАЯ ПОВЕРХНОСТЬ (кривые дополнения)  --  замеренных кривых: %d"
        % len(curves)
    )
    if not curves:
        P("   ПОВЕРХНОСТЬ НЕ ОТКАЛИБРОВАНА НИ ДЛЯ ОДНОГО МАССИВА.")
        P(
            "   Значит цена канала в модели -- ОДНО число на команду, а замерено, что она гуляет в 32"
        )
        P(
            "   раза при неизменных данных. Ниже -- ПРЕДСКАЗАНИЕ модели по трём известным виновникам;"
        )
        P("   первый же прогон padsweep.py его подтвердит или опровергнет.")
        for line in forecast_lines():
            P("   " + line)
        out["sections"].append({"id": "padcurves", "n": 0, "forecast": forecast()})
    else:
        for c in curves:
            p = c["payload"]
            acc = p.get("access") or {}
            pred = predict_curve(
                p["stride_words_base"],
                acc.get("width_bytes", 8),
                acc.get("lanes_per_row", 1),
                acc.get("lane_step_words"),
            )
            pm = {x["pad"]: x for x in pred}
            base = next((x for x in p["curve"] if x["pad"] == 0), None)
            best = min(p["curve"], key=lambda x: (x["wavefronts"], x["pad"]))
            gain = (
                (base["wavefronts"] / best["wavefronts"])
                if base and best["wavefronts"]
                else float("nan")
            )
            pbest = curve_argmin(pred)
            agree = pbest and pbest["pad"] == best["pad"]
            P(
                "   %-28s массив %-10s шаг %d сл.: замер минимум на pad=%d (выигрыш x%.2f), "
                "модель -- pad=%d  %s"
                % (
                    c["id"],
                    p.get("array"),
                    p["stride_words_base"],
                    best["pad"],
                    gain,
                    pbest["pad"] if pbest else -1,
                    "СОШЛОСЬ" if agree else "РАЗОШЛОСЬ",
                )
            )
            errs_c = [
                abs(
                    x["wavefronts"] / (base["wavefronts"] or 1)
                    - pm[x["pad"]]["wf"] / (pm[0]["wf"] or 1)
                )
                for x in p["curve"]
                if x["pad"] in pm
            ]
            if errs_c:
                P(
                    "        форма кривой (нормировано на pad=0): макс расхождение %.2f"
                    % max(errs_c)
                )
        out["sections"].append({"id": "padcurves", "n": len(curves)})

    # --- ИТОГ -------------------------------------------------------------------------------
    rr, clash = rates(recs)
    P("")
    P("-" * 108)
    P(
        "ИТОГ: записей калибровки %d, ставок из ЗАМЕРА %d, негодных записей %d, предупреждений %d"
        % (len(recs), len(rr), len(errs), sum(len(v) for v in warns.values()))
    )
    for c in clash:
        P("   КОНФЛИКТ СТАВОК: %s" % c)
    for k, v in errs.items():
        P("   НЕГОДНА %s: %s" % (k, "; ".join(v)))
    out["n_records"] = len(recs)
    out["n_rates"] = len(rr)
    return out


# ------------------------------------------------------------------------------------------------
# ПРЕДСКАЗАНИЕ ПО ТРЁМ ИЗВЕСТНЫМ ВИНОВНИКАМ (модель, не замер)
# ------------------------------------------------------------------------------------------------
# Шаги взяты из разбора конфликтов 01.08.2026. КАРТА ДОСТУПА у первых двух ИЗ ИСХОДНИКА НЕ
# ВЫВЕДЕНА -- ровно поэтому и строится перебор. Поэтому модель считает кривую не для одной карты,
# а для СЕМЕЙСТВА правдоподобных карт, и сообщает, УСТОЙЧИВ ли аргминимум. Устойчивый аргминимум --
# это предсказание, годное к опровержению; неустойчивый -- честное "решает только замер".
SUSPECTS = [
    {
        "name": "volta_fwd_ws / плитка K",
        "stride_words": 68,
        "why": "LDK8 = 272 Б = 68 слов, 74.5 % конфликтов ядра",
    },
    {
        "name": "volta_fwd_ws / sQ",
        "stride_words": 130,
        "why": "шаг 130 слов; по строкам чист, конфликтует пролог двумя STS.64",
    },
    {
        "name": "cutlass эпилог accumToSmem",
        "stride_words": 130,
        "why": "шаг 130 слов, 130 mod 32 = 2, резонанс с шагом СТОЛБЦА итератора (2 слова); 89.9 %",
    },
]
ACCESS_FAMILY = [(w, g) for w in (4, 8, 16) for g in (1, 2, 4)]


def forecast():
    out = []
    for s in SUSPECTS:
        per_map, argmins = [], []
        for w, g in ACCESS_FAMILY:
            cur = predict_curve(s["stride_words"], w, lanes_per_row=g)
            b = curve_argmin(cur)
            zero = next(x for x in cur if x["pad"] == 0)
            per_map.append(
                {
                    "width_bytes": w,
                    "lanes_per_row": g,
                    "argmin_pad": b["pad"],
                    "wf0": zero["wf"],
                    "wf_best": b["wf"],
                    "gain": zero["wf"] / b["wf"] if b["wf"] else float("nan"),
                }
            )
            argmins.append(b["pad"])
        best_pads = sorted({a for a in argmins})
        # УСТОЙЧИВОСТЬ: есть ли дополнение, оптимальное при ВСЕХ картах семейства
        universal = []
        for pad in range(32):
            ok = True
            for w, g in ACCESS_FAMILY:
                cur = predict_curve(s["stride_words"], w, lanes_per_row=g)
                b = curve_argmin(cur)
                here = next(x for x in cur if x["pad"] == pad)
                if here["wf"] > b["wf"] + 1e-9:
                    ok = False
                    break
            if ok:
                universal.append(pad)
        # МИНИМАКС СОЖАЛЕНИЯ: если единого оптимума нет, осмысленный вопрос не "какое дополнение
        # лучшее", а "какое хуже всего проигрывает лучшему при неизвестной карте". Это и есть
        # решающее правило под незнанием -- и оно ЗАМЕНЯЕТ рассуждение о причине.
        curves = {
            (w, g): predict_curve(s["stride_words"], w, lanes_per_row=g)
            for w, g in ACCESS_FAMILY
        }
        bestwf = {k: curve_argmin(v)["wf"] for k, v in curves.items()}
        regret = []
        for pad in range(32):
            worst = max(
                next(x for x in curves[k] if x["pad"] == pad)["wf"] / bestwf[k]
                for k in curves
            )
            regret.append((worst, pad))
        regret.sort()
        out.append(
            {
                "name": s["name"],
                "why": s["why"],
                "stride_words": s["stride_words"],
                "maps": per_map,
                "argmin_set": best_pads,
                "universal_pads": universal,
                "minimax_regret_pad": regret[0][1],
                "minimax_regret": regret[0][0],
                "regret_pad0": next(r for r in regret if r[1] == 0)[0],
            }
        )
    return out


def forecast_lines():
    lines = []
    for f in forecast():
        lines.append(
            "· %s  (шаг %d сл.; %s)" % (f["name"], f["stride_words"], f["why"])
        )
        gains = [m["gain"] for m in f["maps"]]
        lines.append(
            "    по семейству карт доступа (ширина 4/8/16 Б x полос-на-строку 1/2/4):"
        )
        lines.append(
            "      аргминимумы: %s;  выигрыш модели x%.2f..x%.2f"
            % (f["argmin_set"], min(gains), max(gains))
        )
        if f["universal_pads"]:
            lines.append(
                "      ДОПОЛНЕНИЕ, ОПТИМАЛЬНОЕ ПРИ ВСЕХ КАРТАХ СЕМЕЙСТВА: pad = %s"
                % f["universal_pads"][:8]
            )
            lines.append(
                "      -> предсказание УСТОЙЧИВО к незнанию карты доступа: годится к проверке"
            )
        else:
            lines.append(
                "      единого оптимума по семейству НЕТ -> карту доступа знать ОБЯЗАТЕЛЬНО,"
            )
            lines.append(
                "      предсказать дополнение РАССУЖДЕНИЕМ нельзя (это и есть его цена)"
            )
        lines.append(
            "      минимакс сожаления при НЕИЗВЕСТНОЙ карте: pad = %d (проигрыш лучшему "
            "не более x%.2f); у отгруженного pad = 0 проигрыш до x%.2f"
            % (f["minimax_regret_pad"], f["minimax_regret"], f["regret_pad0"])
        )
    return lines


# ================================================================================================
# 7. САМОПРОВЕРКА
# ================================================================================================
def selftest(verbose=True):
    P = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    ok = bad = 0

    def gate(name, cond):
        nonlocal ok, bad
        if cond:
            ok += 1
            P("  СРАБОТАЛИ  %s" % name)
        else:
            bad += 1
            P("  НЕ СРАБОТАЛИ  %s" % name)

    P("ВОРОТА ФОРМАТА -- каждое обязано СРАБОТАТЬ на подложной записи")
    base = _skeleton(
        "t",
        "padcurve",
        "q",
        "u",
        {"tool": "t"},
        {"k": 1},
        {"index": 1, "foreign_procs": 0},
        "без_возмущения",
        "why",
        {"source": ["x"]},
        payload={
            "array": "a",
            "stride_words_base": 68,
            "curve": [
                {"pad": 0, "wavefronts": 8, "conflicts": 4, "smem_bytes": 1},
                {"pad": 1, "wavefronts": 1, "conflicts": 0, "smem_bytes": 2},
            ],
            "stop": {"reason": "исчерпание"},
        },
    )
    E, W = validate(base, "t")
    gate("годная запись проходит (ошибок нет)", not E)

    import copy

    r = copy.deepcopy(base)
    del r["card"]["foreign_procs"]
    gate(
        "нет числа чужих процессов -> отказ",
        any("foreign_procs" in e for e in validate(r, "t")[0]),
    )

    r = copy.deepcopy(base)
    r["observability"] = "как-нибудь"
    gate("статус наблюдаемости не из набора -> отказ", validate(r, "t")[0])

    r = copy.deepcopy(base)
    r["kind"] = "phases"
    r["payload"] = {"shares": {"a": 0.5}, "sum_named": 0.5}
    gate(
        "фазы со статусом 'без_возмущения' -> отказ",
        any("phases" in e for e in validate(r, "t")[0]),
    )

    r = copy.deepcopy(base)
    r["binds"] = [{"symbol": "НЕИЗВЕСТНЫЙ", "value": 1}]
    gate(
        "привязка к неизвестному символу -> отказ (ставка никуда не попала бы)",
        any("неизвестен" in e for e in validate(r, "t")[0]),
    )

    r = copy.deepcopy(base)
    r["payload"]["curve"] = [c for c in r["payload"]["curve"] if c["pad"] != 0]
    gate(
        "в кривой нет pad=0 -> отказ (не с чем сравнивать)",
        any("pad=0" in e for e in validate(r, "t")[0]),
    )

    r = copy.deepcopy(base)
    r["payload"]["winner_pad"] = 0
    gate(
        "победитель ОБЪЯВЛЕН вместо найденного -> отказ",
        any("ОБЪЯВЛЕН" in e for e in validate(r, "t")[0]),
    )

    r = copy.deepcopy(base)
    r["payload"]["curve"][0]["conflicts"] = 99
    gate(
        "конфликтов больше, чем вайвфронтов -> отказ",
        any("конфликтов" in e for e in validate(r, "t")[0]),
    )

    r = copy.deepcopy(base)
    r["id"] = "другое"
    gate(
        "id не совпадает с именем файла -> отказ",
        any("не совпадает" in e for e in validate(r, "t")[0]),
    )

    r = copy.deepcopy(base)
    del r["payload"]["curve"][1]["smem_bytes"]
    E2, W2 = validate(r, "t")
    gate(
        "нет smem_bytes -> ПРЕДУПРЕЖДЕНИЕ (ворота занятости не проверены), но не отказ",
        (not E2) and any("smem_bytes" in w for w in W2),
    )

    P("")
    P("ЗАКОН: модельная кривая обязана воспроизводить известные точки стенда")
    # шаг 128 Б = 32 слова, ширина 4 Б, обход по столбцу -> 32-кратный конфликт (замер ncu 32.000)
    c = predict_curve(32, 4, lanes_per_row=1)
    z = next(x for x in c if x["pad"] == 0)
    gate(
        "шаг 32 слова, ширина 4 Б -> 32 вайвфронта (ncu: 32.000)",
        abs(z["wf"] - 32.0) < 1e-9,
    )
    o = next(x for x in c if x["pad"] == 1)
    gate(
        "дополнение в ОДНО слово (шаг 132 Б) -> 1 вайвфронт (ncu: 1.000)",
        abs(o["wf"] - 1.0) < 1e-9,
    )
    c16 = predict_curve(32, 16, lanes_per_row=1)
    gate(
        "ширина 16 Б, 32 РАЗНЫХ адреса: ПОЛ = 4 вайвфронта (512 уник. Б / 128), "
        "дополнение ниже не опускает",
        abs(min(x["wf"] for x in c16) - 4.0) < 1e-9,
    )
    bc = COST([7] * LANES, 16)  # полная рассылка одного адреса, LDS.128
    gate(
        "ширина 16 Б, ПОЛНАЯ рассылка -> ровно 2 вайвфронта (ncu: 2.000, s8x128d32)",
        abs(bc[2] - 2.0) < 1e-9,
    )

    P("")
    P("ЗАГРУЗЧИК МОДЕЛИ: ставка обязана ДОЙТИ до отпечатка, а не просто прочитаться")
    try:
        import importlib
        import tempo as T

        importlib.reload(T)
        before = T.MIO_BYTES_PER_CYCLE
        T.MIO_BYTES_PER_CYCLE = -1.0  # заведомо негодная ставка
        took = T.load_calibration()
        gate(
            "load_calibration подменил ставку значением из замера (%g -> %g), ставок %d"
            % (-1.0, T.MIO_BYTES_PER_CYCLE, took),
            took > 0 and T.MIO_BYTES_PER_CYCLE == before,
        )
        gate(
            "замеренная цена команды долетела до модели (MIO_WF_PER_INST)",
            bool(T.MIO_WF_PER_INST),
        )
        # ТРЕТИЙ СЦЕНАРИЙ: та же нагрузка, ставка MIO из замера -> ResMII обязан вырасти ровно в
        # отношении (замер/номинал) по каналу MIO.
        binp = os.path.join(TEMPO, "build", "probe")
        done = False
        if os.path.exists(binp):
            try:
                kern = T.load_kernels(binp)
                for name in kern:
                    cfg = T.IS.CFG(kern[name])
                    ins, _ = T.mainloop_ins(cfg)
                    if not ins:
                        continue
                    nm = T.mio_nominal_mean(ins)
                    if not nm:
                        continue
                    a = T.analyse(ins, 8, cal_wf=nm * 2.0)
                    ok_scale = abs(a["mio_scale"] - 2.0) < 1e-9
                    ok_load = (
                        abs(a["res_cal"][T.CH_MIO] - 2.0 * a["res_nom"][T.CH_MIO])
                        < 1e-6
                    )
                    gate(
                        "ставка MIO x2 -> нагрузка канала MIO ровно x2 (тело %s)"
                        % name[:28],
                        ok_scale and ok_load,
                    )
                    done = True
                    break
            except Exception as ex:
                P("  НЕ ПРОВЕРЕНО  третий сценарий: %s" % ex)
        if not done:
            P(
                "  НЕ ПРОВЕРЕНО  третий сценарий: нет разобранного тела в build/probe "
                "(это НЕ 'сошлось')"
            )
    except Exception as ex:
        P("  НЕ ПРОВЕРЕНО  загрузчик модели: %s" % ex)

    P("")
    P("КАЛИБРОВКА НЕ КАЛИБРУЕТСЯ САМА СОБОЙ")
    recs, _, _ = load_dir()
    gate(
        "в data/calib нет ни одной записи, порождённой моделью",
        all(
            "predict" not in json.dumps(x.get("taken_with", {}), ensure_ascii=False)
            and x.get("taken_with", {}).get("tool")
            for x in recs
        ),
    )

    P("")
    P("ИТОГ САМОПРОВЕРКИ: сработало %d, не сработало %d" % (ok, bad))
    return ok, bad


# ================================================================================================
def cmd_list(args):
    recs, errs, warns = load_dir(args.dir, strict=False)
    print(
        "%-28s %-15s %-16s %-8s %s" % ("id", "вид", "наблюдаемость", "ставок", "форма")
    )
    for r in recs:
        sh = ", ".join(
            "%s=%s" % (k, v) for k, v in list(r.get("shape", {}).items())[:3]
        )
        print(
            "%-28s %-15s %-16s %-8d %s"
            % (
                r["id"],
                r["kind"],
                r.get("observability"),
                len(r.get("binds") or []),
                sh[:44],
            )
        )
    rr, clash = rates(recs)
    print("\nСТАВКИ ИЗ ЗАМЕРА (%d):" % len(rr))
    for s, v in sorted(rr.items()):
        print(
            "   %-28s = %-10g  <- %s  [%s]"
            % (s, v["value"], v["from"], v["observability"])
        )
    for k, v in errs.items():
        print("   НЕГОДНА %s: %s" % (k, "; ".join(v)))
    for k, v in warns.items():
        for w in v:
            print("   предупреждение [%s]: %s" % (k, w))
    return 1 if errs else 0


def cmd_validate(args):
    recs, errs, warns = load_dir(args.dir, strict=False)
    print(
        "записей %d, негодных %d, предупреждений %d"
        % (len(recs), len(errs), sum(len(v) for v in warns.values()))
    )
    for k, v in errs.items():
        print("  НЕГОДНА %s:" % k)
        for e in v:
            print("      %s" % e)
    for k, v in warns.items():
        for w in v:
            print("  предупреждение [%s]: %s" % (k, w))
    return 1 if errs else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="калибровка модели tempo")
    ap.add_argument(
        "cmd", choices=("validate", "list", "ingest", "residual", "predict", "selftest")
    )
    ap.add_argument("--dir", default=CALIB_DIR)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stride", type=int, default=68)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--lanes-per-row", type=int, default=1)
    ap.add_argument("--lane-step", type=int, default=None)
    ap.add_argument("--json")
    a = ap.parse_args(argv)

    if a.cmd == "validate":
        return cmd_validate(a)
    if a.cmd == "list":
        return cmd_list(a)
    if a.cmd == "ingest":
        made, skipped, gates = ingest(a.dir, a.force)
        print(
            "заведено %d, уже было %d, ворота ссылки не прошли %d"
            % (len(made), len(skipped), len(gates))
        )
        return 1 if gates else 0
    if a.cmd == "residual":
        out = residual(a.dir)
        if a.json:
            json.dump(
                out,
                open(a.json, "w", encoding="utf-8"),
                indent=1,
                ensure_ascii=False,
                default=str,
            )
        return 0
    if a.cmd == "predict":
        cur = predict_curve(a.stride, a.width, a.lanes_per_row, a.lane_step)
        b = curve_argmin(cur)
        print(
            "шаг %d слов, ширина %d Б, полос на строку %d"
            % (a.stride, a.width, a.lanes_per_row)
        )
        print("%5s %10s %8s %8s" % ("pad", "конфл", "ПОЛ", "вайвфр"))
        for p in cur:
            mark = "  <== минимум" if p["pad"] == b["pad"] else ""
            print(
                "%5d %10d %8.2f %8.2f%s"
                % (p["pad"], p["degree"], p["floor"], p["wf"], mark)
            )
        return 0
    if a.cmd == "selftest":
        ok, bad = selftest()
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
