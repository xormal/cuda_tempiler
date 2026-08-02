#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ПЛАНИРОВЩИК МИНИМАЛЬНОГО ВРЕМЕННОГО ОТПЕЧАТКА (веха M3).

ЧТО ДЕЛАЕТ.  По телу цикла (список команд с отпечатками и графом предшествований) строит
ЦИКЛИЧЕСКОЕ расписание периода T: каждой команде назначается момент s_i, повторение идёт с
шагом T, занятость каналов считается по остатку от деления на T.  Ищет наименьшее T, при
котором расписание существует, и выдаёт вместе с ним ДВА сертификата:

  * нижнюю границу  T >= max(ResMII, RecMII)  -- "меньше нельзя ни при какой укладке";
  * MaxLive -- сколько значений одновременно живо в найденном расписании.  Это ЧЕТВЁРТЫЙ
    ресурс: если MaxLive > 255, расписание нереализуемо, и цена его нарушения -- не такт, а
    разлив (замерено на стенде: тот же счёт команд при -maxrregcount=40 даёт x7.3 по времени).

ПОЧЕМУ ИМЕННО ТАК (теоретическая заметка `theory/q1_answer.md`).
  Задача в общем виде СИЛЬНО NP-ТРУДНА уже без предшествований (редукция от 3-Partition:
  каналы-стены и каналы-предметы, точное замощение периода).  Но при ФИКСИРОВАННОЙ
  СЕРИАЛИЗАЦИИ каналов (кто за кем занимает канал) оптимум вычислим за полином и равен
  максимуму по циклам отношения (сумма длин)/(число оборотов) -- задача о максимальном
  среднем цикле.  Поэтому правильная конструкция:
        внешний перебор/эвристика по сериализациям  x  внутреннее ТОЧНОЕ решение цикла.
  Здесь реализовано внутреннее точное (max cycle ratio -> потенциалы = расписание) и
  внешняя жадность (iterative modulo scheduling Rau с откатом).

ЗАПУСК
    python3 schedule.py <cubin> --kernel <regex> --warps 8
    python3 schedule.py --probe-body h8+c8x1 --warps 8     # тело со стенда
"""

import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Разборщик SASS переехал из бывшего vendor/ на сторону ПЛАГИНА (он архитектурный):
# tempo/plugins/sm70/isa_sass.py. Имя IS сохранено, тело файла не изменялось.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "tempo", "plugins", "sm70"
    ),
)
import tempo as T  # noqa: E402
import isa_sass as IS  # noqa: E402

REG_BUDGET = 255  # предел ISA на нить

# --------------------------------------------------------------------------------------------
# ДВА КРИТЕРИЯ, А НЕ ОДИН.  Расписание с БО́ЛЬШИМ периодом, но МЕНЬШИМ MaxLive, бывает БЫСТРЕЕ --
# это не соображение, а ЗАМЕР (tools/probe.cu, семейство q*, 192 команды ALU у всех членов):
#
#   тело      NC  NP(=MaxLive)   модель   cap=255   cap=64     cap=40
#   q192x1   192   1              776      783.5     783.5      783.5
#   q1x192     1 192              776      777.5    48469.7   56246.3
#
# Модель по каналам даёт ВСЕМ членам семейства ОДИН период (776), и при свободном бюджете так и
# выходит.  Как только бюджет прижат, порядок задаётся ИСКЛЮЧИТЕЛЬНО живостью: q192x1 -- ХУДШИЙ
# по периоду (783.5 против 776.3) и ЛУЧШИЙ по времени в 62-72 раза.  Поэтому планировщик обязан
# искать по паре (T, MaxLive), а не по T: здесь ищется наименьший период СРЕДИ ВЛЕЗАЮЩИХ по
# регистрам, и если не влезает ни один -- это сообщается, а не заминается.
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# 1. РЕЗЕРВАЦИОННАЯ ТАБЛИЦА ПО МОДУЛЮ
# --------------------------------------------------------------------------------------------
class ModuloTable(object):
    """Занятость каналов по остатку от деления на период.

    ЕДИНИЦЫ.  Расписание строится для ОДНОГО варпа, но на планировщике их wps штук, и все
    исполняют одно тело.  Поэтому команда держит канал не c тактов, а c*wps: доля соседей
    учтена растяжением, а не уменьшением ёмкости (иначе при wps>1 не влезает ничего).
    Канал MIO -- ресурс всего SM, там растяжение по ПОЛНОМУ числу варпов.
    """

    def __init__(self, period, warps):
        self.T = period
        self.warps = warps
        self.wps = max(1.0, warps / float(T.SCHEDULERS))
        self.use = collections.defaultdict(float)

    def span(self, ch, c):
        if ch == T.CH_MIO:
            return max(1, int(round(c * self.warps / T.MIO_BYTES_PER_CYCLE)))
        return max(1, int(round(c * self.wps)))

    def fits(self, fp, t):
        for ch, c in fp.items():
            for d in range(self.span(ch, c)):
                if self.use[(ch, (t + d) % self.T)] > 1e-9:
                    return False
        return True

    def place(self, fp, t, sign=1):
        for ch, c in fp.items():
            for d in range(self.span(ch, c)):
                self.use[(ch, (t + d) % self.T)] += sign


# --------------------------------------------------------------------------------------------
# 2. ITERATIVE MODULO SCHEDULING
# --------------------------------------------------------------------------------------------
def schedule(ins_list, warps, tmax_mult=3.0, budget=6000, regs=True, tries=48):
    """Наименьший период СРЕДИ РЕАЛИЗУЕМЫХ ПО РЕГИСТРАМ.

    regs=True -- расписание принимается, только если MaxLive <= Q(занятость) - 9 (ИЗМЕРЕННЫЙ
    порог разлива).  Иначе поиск продолжается: больший период даёт более короткие живости.
    Возвращает и лучшее по периоду БЕЗ учёта регистров -- чтобы разрыв между двумя оптимумами
    был виден числом, а не подразумевался.
    """
    n = len(ins_list)
    fps = [
        T.footprint(it, safe=False) for it in ins_list
    ]  # укладка -- по НОМИНАЛЬНОЙ ставке
    lats = [T.latency(it) for it in ins_list]
    edges = T.dep_graph(ins_list)
    pred = collections.defaultdict(list)
    for i, j, l, w in edges:
        pred[j].append((i, l, w))

    q, thr, _ = T.reg_verdict(0, warps)
    lo = max(_resmii(fps, warps), T.rec_mii(n, edges))
    period = max(1, int(lo + 0.999))
    hi = min(int(period * tmax_mult) + 8, period + tries)  # поиск ограничен: см. ниже
    first = None
    while period <= hi:
        s = _try_period(n, fps, lats, pred, period, warps, budget)
        if s is not None:
            ml = max_live(ins_list, s, period)
            if first is None:
                first = {"T": period, "maxlive": ml}
            if (not regs) or ml <= thr:
                return {
                    "T": period,
                    "sched": s,
                    "lower": lo,
                    "maxlive": ml,
                    "n": n,
                    "budget": q,
                    "thr": thr,
                    "fits": ml <= thr,
                    "T_noreg": first["T"],
                    "maxlive_noreg": first["maxlive"],
                }
        period += 1
    return {
        "T": None,
        "sched": None,
        "lower": lo,
        "maxlive": None,
        "n": n,
        "budget": q,
        "thr": thr,
        "fits": False,
        "T_noreg": first["T"] if first else None,
        "maxlive_noreg": first["maxlive"] if first else None,
    }


def _resmii(fps, warps):
    load = collections.Counter()
    for fp in fps:
        for ch, c in fp.items():
            load[ch] += c
    wps = warps / float(T.SCHEDULERS)
    best = 0.0
    for ch, v in load.items():
        best = max(
            best, v * warps / T.MIO_BYTES_PER_CYCLE if ch == T.CH_MIO else v * wps
        )
    return best


def _try_period(n, fps, lats, pred, period, warps, budget):
    """Rau: расписание с откатом -- команда вытесняет уже стоящих, если иначе не влезает."""
    tab = ModuloTable(period, warps)
    sched = [None] * n
    order = sorted(range(n), key=lambda i: -len(pred[i]))
    order = list(range(n))  # порядок тела: даёт устойчивый результат
    budget_left = budget
    stack = list(order)
    while stack and budget_left > 0:
        budget_left -= 1
        i = stack.pop(0)
        early = 0
        for p, l, w in pred[i]:
            if sched[p] is not None:
                early = max(early, sched[p] + l - w * period)
        early = max(0, int(early + 0.999))
        placed = False
        for t in range(early, early + period):
            if tab.fits(fps[i], t):
                tab.place(fps[i], t, +1)
                sched[i] = t
                placed = True
                break
        if not placed:
            t = early
            # вытеснить конфликтующих
            victims = [
                j
                for j in range(n)
                if sched[j] is not None
                and _conflict(fps[i], t, fps[j], sched[j], period, tab)
            ]
            if not victims:
                return None
            for j in victims:
                tab.place(fps[j], sched[j], -1)
                sched[j] = None
                stack.append(j)
            if tab.fits(fps[i], t):
                tab.place(fps[i], t, +1)
                sched[i] = t
            else:
                return None
    if any(x is None for x in sched):
        return None
    return sched


def _conflict(fpa, ta, fpb, tb, period, tab):
    for ch in fpa:
        if ch in fpb:
            span_a = tab.span(ch, fpa[ch])
            span_b = tab.span(ch, fpb[ch])
            for d in range(span_a):
                for e in range(span_b):
                    if (ta + d) % period == (tb + e) % period:
                        return True
    return False


def max_live(ins_list, sched, period):
    """MaxLive расписания: максимум по остаткам от числа ОДНОВРЕМЕННО живых значений.

    ВАЖНО И ЛЕГКО ОШИБИТЬСЯ.  Значение, которое команда и ЧИТАЕТ, и ПИШЕТ (x = xor(x, m)),
    живёт НЕ один такт, а ЦЕЛЫЙ ПЕРИОД: его читает следующая итерация.  Прежняя редакция брала
    расстояние def->use ВНУТРИ итерации и давала MaxLive=2 для тела с 240 живыми значениями --
    то есть ровно на том теле, ради которого этот счётчик и нужен.  Здесь живость считается по
    правилу modulo-расписания: use в позиции j <= i -- это СЛЕДУЮЩИЙ оборот, +period.
    """
    if sched is None:
        return None
    lastdef = {}
    for k, it in enumerate(ins_list):
        d = T.dst_of(it)
        if d:
            lastdef[d] = k
    uses = collections.defaultdict(list)
    for k, it in enumerate(ins_list):
        for r in T.srcs_of(it):
            uses[r].append(k)
    whole = 0
    live = collections.Counter()
    for r, i in lastdef.items():
        us = uses.get(r)
        if not us:
            continue
        life = 1
        for j in us:
            end = sched[j] if j > i else sched[j] + period
            life = max(life, end - sched[i])
        whole += life // period
        for t in range(sched[i], sched[i] + (life % period)):
            live[t % period] += 1
    return whole + (max(live.values()) if live else 0)


# --------------------------------------------------------------------------------------------
# 3. ЗАПУСК
# --------------------------------------------------------------------------------------------
def body_from_probe(binpath, variant_params):
    kern = T.load_kernels(binpath)
    key = ("Li%dE" * 9) % variant_params
    for k in kern:
        if key in k:
            cfg = IS.CFG(kern[k])
            ins, _ = T.mainloop_ins(cfg)
            return ins
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj", nargs="?")
    ap.add_argument("--kernel")
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--probe-body")
    ap.add_argument(
        "--probe",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "build", "probe"
        ),
    )
    ap.add_argument(
        "--measured",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "probe_measured.json",
        ),
    )
    args = ap.parse_args()

    bodies = []
    if args.probe_body:
        variants = T.parse_variants(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe.cu")
        )
        import json

        meas = json.load(open(args.measured))
        for name in args.probe_body.split(","):
            if name not in variants:
                print("нет варианта", name)
                continue
            ins = body_from_probe(args.probe, variants[name])
            bodies.append((name, ins, meas.get("%s|%d" % (name, args.warps))))
    else:
        kern = T.load_kernels(args.obj)
        for k in kern:
            if args.kernel and not re.search(args.kernel, k):
                continue
            cfg = IS.CFG(kern[k])
            ins, _ = T.mainloop_ins(cfg)
            if ins:
                bodies.append((k[:56], ins, None))

    print(
        "%-14s %5s %8s %8s %8s %7s %8s %7s %7s %s"
        % (
            "тело",
            "команд",
            "нижняя",
            "T_план",
            "замер",
            "запас%",
            "MaxLive",
            "бюджет",
            "порог",
            "вердикт",
        )
    )
    for name, ins, meas in bodies:
        if not ins:
            continue
        r = schedule(ins, args.warps)
        gap = ("%.1f" % (100.0 * (meas - r["T"]) / meas)) if (meas and r["T"]) else "-"
        verdict = "влезает" if r["fits"] else "РАЗЛИВ (цена x5..x75, не такт)"
        if r["T"] and r["T_noreg"] and r["T"] != r["T_noreg"]:
            verdict += "; оптимум по периоду T=%d при MaxLive=%s" % (
                r["T_noreg"],
                r["maxlive_noreg"],
            )
        print(
            "%-14s %5d %8.1f %8s %8s %7s %8s %7d %7d %s"
            % (
                name,
                r["n"],
                r["lower"],
                r["T"] if r["T"] else "нет",
                ("%.1f" % meas) if meas else "-",
                gap,
                r["maxlive"] if r["maxlive"] is not None else "-",
                r["budget"],
                r["thr"],
                verdict,
            )
        )


if __name__ == "__main__":
    main()
