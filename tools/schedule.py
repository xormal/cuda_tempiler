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


def max_live(ins_list, sched, period, wide=True):
    """MaxLive расписания: максимум по остаткам от числа ОДНОВРЕМЕННО живых ЯЧЕЕК.

    ВАЖНО И ЛЕГКО ОШИБИТЬСЯ (ДВАЖДЫ, и обе ошибки здесь уже совершены и исправлены).
    ПЕРВАЯ.  Значение, которое команда и ЧИТАЕТ, и ПИШЕТ (x = xor(x, m)), живёт НЕ один такт, а
    ЦЕЛЫЙ ПЕРИОД: его читает следующая итерация.  Ранняя редакция брала расстояние def->use
    ВНУТРИ итерации и давала MaxLive=2 для тела с 240 живыми значениями.  Лечится правилом
    modulo-расписания: use в позиции j <= i -- это СЛЕДУЮЩИЙ оборот, +period.
    ВТОРАЯ (задача 141).  Единицей счёта было ИМЯ регистра, а занимает команда ЯЧЕЙКИ:
    `LDS.U.128 R24` держит R24..R27.  Занижение ВДВОЕ -> вывод «влезает» на теле, которое
    разлило 32 байта.  Лечится расширением операнда по ширине (`wide=True`, LAW=L-REG-WIDTH-NOT-NAMES).
    """
    if sched is None:
        return None
    dst_of = (
        T.dst_regs_of if wide else (lambda it: [T.dst_of(it)] if T.dst_of(it) else [])
    )
    srcs_of = T.src_regs_of if wide else T.srcs_of
    lastdef = {}
    for k, it in enumerate(ins_list):
        for d in dst_of(it):
            lastdef[d] = k
    uses = collections.defaultdict(list)
    for k, it in enumerate(ins_list):
        for r in srcs_of(it):
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


def live_cyclic(ins, seq=None, wide=True, rounds=6):
    """MaxLive ЦИКЛИЧЕСКОГО тела: обратный поток данных по петле до неподвижной точки.

    ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ max_live ВЫШЕ, И ПОЧЕМУ НУЖНЫ ОБЕ.  max_live считает живость
    MODULO-РАСПИСАНИЯ: команды идут внахлёст, и вопрос стоит про остатки от деления на период.
    Здесь -- живость ПОСЛЕДОВАТЕЛЬНОГО тела, у которого есть только обратная дуга.  Ровно эту
    величину и объявляет сборщик числом «Used N registers», поэтому сверять с ptxas надо её.

    ПОЧЕМУ ПОТОК, А НЕ ИНТЕРВАЛЫ «первое определение -- последнее использование».  В машинном
    коде имена ФИЗИЧЕСКИЕ и переиспользуются: интервал по имени слил бы два независимых значения
    в одно длинное и посчитал бы не живость, а РАСПРЕДЕЛЕНИЕ (то есть ответ ptxas, списанный у
    ptxas же).  Обратный поток даёт живость ЗНАЧЕНИЙ: определение убивает, использование рождает.

    seq -- порядок команд (перестановка индексов); None -- порядок тела.
    """
    order = list(range(len(ins))) if seq is None else list(seq)
    body = [ins[g] for g in order]
    n = len(body)
    if not n:
        return 0
    dst_of = (
        T.dst_regs_of if wide else (lambda it: [T.dst_of(it)] if T.dst_of(it) else [])
    )
    srcs_of = T.src_regs_of if wide else T.srcs_of
    D = [set(x for x in dst_of(it) if x) for it in body]
    S = [set(srcs_of(it)) for it in body]
    live = set()
    for _ in range(rounds):  # неподвижная точка по петле; сходится за 2-3 оборота
        prev = frozenset(live)
        for k in range(n - 1, -1, -1):
            live -= D[k]
            live |= S[k]
        if frozenset(live) == prev:
            break
    m = len(live)
    for k in range(n - 1, -1, -1):
        live -= D[k]
        live |= S[k]
        m = max(m, len(live))
    return m


# --------------------------------------------------------------------------------------------
# 2b. УКЛАДКА В ТЕНЬ ТЕНЗОРНОЙ КОМАНДЫ (задача 135)
#
# ЗАЧЕМ ОТДЕЛЬНАЯ ЗАДАЧА, А НЕ «ПЕРЕУПОРЯДОЧИТЬ».  ЗАМЕРЕНО (reports/EV_p1p2_schedule.md §2):
# поле задержки задаёт КЛАСС СЛЕДУЮЩЕЙ команды.  Тензорная команда занимает свою трубу на два
# такта, а слот ВЫДАЧИ -- на один; если сразу за ней стоит нетензорная, планировщик выдаёт её в
# освободившийся такт, и она стоит НОЛЬ ДОБАВЛЕННЫХ ТАКТОВ.  Значит у каждой тензорной команды
# есть ТЕНЬ фиксированного габарита -- ровно один слот выдачи, -- и вопрос ставится как УКЛАДКА:
# разместить нетензорные предметы по теням, не меняя общей квадратуры.
#
# ЧИСЛО, КОТОРОЕ МЕНЯЕТ ПОСТАНОВКУ: теней ВТРОЕ БОЛЬШЕ, чем предметов (у разбираемого тела 512
# против 162).  Тень -- ресурс ИЗБЫТОЧНЫЙ, поэтому непоставленный предмет есть НЕУДАЧА УКЛАДКИ,
# а не нехватка места, и обязан быть назван с ПРИЧИНОЙ.  Именно поэтому здесь считается не
# агрегат, а ПЕРЕЧЕНЬ: агрегат («в тени 25.9 %») не отличает закон от недоработки.
#
# АЛГОРИТМ.  Списочное планирование по критическому пути на графе предшествований тела, с
# чередующим предпочтением: после тензорной берём готовую НЕтензорную, после нетензорной --
# готовую тензорную.  Приоритет внутри класса -- высота (длина оставшегося пути), то есть
# классический CP-priority.  Это ВЕРХНЯЯ оценка достижимого (найденное расписание существует),
# а нижняя даётся тензорным полом 2*nH - min(nN, nH) + nN; когда они сходятся -- укладка
# ОПТИМАЛЬНА, и на теле-свидетеле они сходятся ровно (1024 = 1024).
# --------------------------------------------------------------------------------------------
# Ставки задержки нетензорной команды, У КОТОРОЙ СОСЕД -- ТАКАЯ ЖЕ (то есть ВНЕ тени).
# ЗАМЕРЕНО на четырёх телах (EV_p1p2_schedule.md §2): LDS->LDS 4.00, STS->STS 4.00,
# LDG->LDG 4.00, IMAD->IMAD 2.45.  Ставка В ТЕНИ -- 1.00, и это тоже замер, а не допущение:
# у тела-свидетеля ВСЕ нетензорные команды мейнлупа идут ровно по 1.00.
SHADOW_RATE = {
    "загр.разд": 4.00,
    "зап.разд": 4.00,
    "загр.глоб": 4.00,
    "зап.глоб": 4.00,
    "разлив": 4.00,
    "ветвл./пред": 3.80,
    "барьеры": 5.00,
}
SHADOW_RATE_DEFAULT = 2.45  # счётные: IMAD->IMAD
SHADOW_RATE_IN = 1.00  # в тени -- ЗАМЕР на теле-свидетеле, не допущение
TENSOR_BUSY = 2.00  # HMMA.884 держит тензорную трубу два такта

# Пространства адресов: взаимный порядок обращений В ОДНО пространство сборщик обязан сохранить,
# потому что доказать непересечение адресов он не может.  Это НЕ истинная зависимость -- на
# двойном буфере разделяемой мы ЗНАЕМ, что запись и чтение идут в разные половины, а сборщик
# не знает.  Поэтому дуги этого вида помечены отдельным видом ("mem") и снимаются ключом.
_SPACE = {
    "LDS": "smem",
    "LDSM": "smem",
    "STS": "smem",
    "LDG": "gmem",
    "STG": "gmem",
    "LD": "gmem",
    "ST": "gmem",
    "RED": "gmem",
    "ATOM": "gmem",
    "ATOMG": "gmem",
    "ATOMS": "smem",
    "LDL": "local",
    "STL": "local",
}
# Ограда: команда, через которую сборщик не переносит НИЧЕГО ни в одну сторону.  Ветвление
# сюда входит намеренно: граница основного блока -- такая же ограда, и именно она (а не
# зависимость и не регистры) держит нашу тень пустой.
_FENCE = (
    "BAR",
    "MEMBAR",
    "BSSY",
    "BSYNC",
    "BRA",
    "BRX",
    "JMP",
    "CALL",
    "RET",
    "EXIT",
    "WARPSYNC",
    "DEPBAR",
)
_SETP = ("ISETP", "PSETP", "PLOP3", "FSETP", "HSETP2", "VOTE")


def _wrote_pred(it):
    if not any(it.op.startswith(p) for p in _SETP):
        return None
    pp = re.findall(r"\bP\d+\b", it.text)
    return pp[0] if pp else None


def shadow_edges(ins):
    """Граф предшествований тела -> {i: [(j, вид)]}, j < i.  Виды: data | mem | fence.

    Берутся ВСЕ три вида зависимости по регистрам и предикатам (RAW, WAR, WAW), а не только
    истинные: для ПЕРЕУКЛАДКИ анти- и выходные зависимости так же обязательны, как истинные, --
    иначе получится порядок, который «лучше» и НЕВЕРЕН.  (Для RecMII, наоборот, нужны только
    истинные: там ищется контур, а не законный порядок.  Две разные задачи -- два разных графа.)
    """
    n = len(ins)
    P = collections.defaultdict(list)
    lastdef, lastuse, lastmem = {}, collections.defaultdict(list), {}
    lastfence = None
    for i, it in enumerate(ins):
        if lastfence is not None:
            P[i].append((lastfence, "fence"))
        if it.base in _FENCE:  # ограда двусторонняя: до неё ничего не уедет ВНИЗ
            for j in range(0 if lastfence is None else lastfence + 1, i):
                P[i].append((j, "fence"))
        rd = set(it.srcs) | set(it.addrregs)
        if it.pred:
            rd.add(it.pred)
        for r in rd:
            if r in lastdef:
                P[i].append((lastdef[r], "data"))
        wr = set()
        if it.dst:
            wr.add(it.dst)
        p = _wrote_pred(it)
        if p:
            wr.add(p)
        for r in wr:
            for u in lastuse.get(r, ()):
                P[i].append((u, "data"))  # WAR
            if r in lastdef:
                P[i].append((lastdef[r], "data"))  # WAW
        sp = _SPACE.get(it.base)
        if sp is not None and sp in lastmem:
            P[i].append((lastmem[sp], "mem"))
        for r in rd:
            lastuse[r].append(i)
        for r in wr:
            lastdef[r] = i
            lastuse[r] = []
        if sp is not None:
            lastmem[sp] = i
        if it.base in _FENCE:
            lastfence = i
    return P


def shadow_rate(it):
    return SHADOW_RATE.get(it.cls, SHADOW_RATE_DEFAULT)


def shadow_cost(seq, ins):
    """Длина расписания последовательности по ЗАМЕРЕННОЙ таблице «класс соседа».

    Последовательность ЦИКЛИЧЕСКАЯ: за последней командой мейнлупа идёт первая.  Линейный счёт
    теряет ровно замыкание петли -- одну команду из полутора сотен, и на теле-свидетеле это
    видно как 148 против 149.
    """
    n = len(seq)
    tot = 0.0
    for k in range(n):
        a, b = ins[seq[k]], ins[seq[(k + 1) % n]]
        at, bt = a.cls == IS.TENSOR, b.cls == IS.TENSOR
        tot += (
            (TENSOR_BUSY if bt else SHADOW_RATE_IN)
            if at
            else (SHADOW_RATE_IN if bt else shadow_rate(a))
        )
    return tot


def shadow_floor(ins):
    """Нижняя граница: каждый предмет, которому хватило тени, стоит 1.00, остальные -- ставку.

    ПОЧЕМУ ЭТО ИМЕННО ГРАНИЦА.  Ни одна укладка не может занять больше теней, чем есть тензорных
    команд, и ни одна не может сделать тензорную часть меньше 2*nH - (занятых теней).
    """
    nh = sum(1 for it in ins if it.cls == IS.TENSOR)
    non = [it for it in ins if it.cls != IS.TENSOR]
    fit = min(len(non), nh)
    rest = sorted((shadow_rate(it) for it in non), reverse=True)[
        : max(0, len(non) - fit)
    ]
    return TENSOR_BUSY * nh - fit + SHADOW_RATE_IN * fit + sum(rest)


def shadow_maxlive(seq, ins, wide=True):
    """MaxLive НАЙДЕННОГО порядка -- в тех же единицах, в каких их печатает сборщик.

    Нужен ровно затем, чтобы отличить причину «(б) время жизни регистра» от «(а) зависимость»:
    укладка, требующая держать вдвое больше значений живыми, на 250 из 255 нереализуема, и это
    надо назвать ЧИСЛОМ, а не опасением.

    ДВЕ ПРАВКИ ЗАДАЧИ 141, и обе меняют вывод.  (1) Единица счёта -- ЯЧЕЙКА, а не имя
    (LAW=L-REG-WIDTH-NOT-NAMES).  (2) Проход не линейный, а ЦИКЛИЧЕСКИЙ: тело -- петля, и
    значение, дожившее до обратной дуги, живо и в начале следующего оборота.  Линейный проход
    занижал именно на нём -- то есть на накопителях, ради которых счётчик и заведён.
    """
    return live_cyclic(ins, seq, wide=wide)


def shadow_pack(ins, drop=(), edges=None):
    """Списочное планирование с ЧЕРЕДУЮЩИМ предпочтением -> порядок команд (список индексов).

    drop -- виды дуг, объявленные ЛОЖНЫМИ для этого прогона ("mem", "fence").  Это не поблажка
    себе: снятие вида дуги -- ГИПОТЕЗА о том, что даёт перестройка ИСХОДНИКА, и она обязана
    называться вслух.  Снять "mem" = «мы знаем, что запись и чтение идут в разные половины
    двойного буфера, а сборщик доказать этого не может».  Снять "fence" = «ветвления убраны из
    тела, и мейнлуп стал ОДНИМ основным блоком».
    """
    n = len(ins)
    P = edges if edges is not None else shadow_edges(ins)
    pr = {i: set(j for j, k in P.get(i, ()) if k not in drop) for i in range(n)}
    su = collections.defaultdict(list)
    for i, ps in pr.items():
        for j in ps:
            su[j].append(i)
    h = [0] * n
    for i in range(n - 1, -1, -1):
        h[i] = 1 + max([h[j] for j in su.get(i, ())] or [0])
    remain = {i: len(pr[i]) for i in range(n)}
    ready = set(i for i in range(n) if remain[i] == 0)
    seq, prev_tensor = [], None
    while ready:
        want = prev_tensor is not True
        cand = [i for i in ready if (ins[i].cls == IS.TENSOR) == want]
        if not cand:
            cand = list(ready)
        i = max(cand, key=lambda x: (h[x], -x))
        ready.discard(i)
        seq.append(i)
        prev_tensor = ins[i].cls == IS.TENSOR
        for j in su.get(i, ()):
            remain[j] -= 1
            if remain[j] == 0:
                ready.add(j)
    if len(seq) != n:  # цикл в графе -- отказ, а не молчаливо усечённый порядок
        raise ValueError(
            "граф предшествований не ацикличен: уложено %d из %d" % (len(seq), n)
        )
    return seq


def shadow_result(ins, drop=(), edges=None):
    """-> dict: расписание укладки, занятые тени, ОСТАТОК (индексы), MaxLive, граница."""
    seq = shadow_pack(ins, drop, edges)
    n = len(seq)
    tt = [ins[g].cls == IS.TENSOR for g in seq]
    inside = [(not tt[k]) and tt[(k + 1) % n] for k in range(n)]
    rest = [seq[k] for k in range(n) if (not tt[k]) and not inside[k]]
    nn = sum(1 for it in ins if it.cls != IS.TENSOR)
    return {
        "seq": seq,
        "sched": shadow_cost(seq, ins),
        "floor": shadow_floor(ins),
        "in": sum(inside),
        "non": nn,
        "rest": rest,
        "frac": (float(sum(inside)) / nn) if nn else 0.0,
        "maxlive": shadow_maxlive(seq, ins),
    }


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
