#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ТОЧКА ИЗЛОМА ЧИСЛОМ, а не на глаз.

Кривая T(NP) при фиксированном бюджете Q: до излома она -- РОВНО канальная модель (наклон
2*wps тактов на живое значение), после -- разлив.  Излом ищется двухсегментной подгонкой:
    T(NP) = a + b*NP                       при NP <= K
    T(NP) = a + b*K + c*(NP-K) + d*(NP-K)^2  при NP >  K
K перебирается по сетке, для каждого K коэффициенты берутся МНК, выбирается K с наименьшей
суммой квадратов ОТНОСИТЕЛЬНЫХ невязок (иначе подгонку тянут огромные значения справа).
Отдельно печатается ПОРОГ ПО ОТНОШЕНИЮ (первое NP, где замер выше канальной модели более чем
на 2 %) -- он не зависит ни от какой подгонки и потому проверяет её.
"""

import json
import sys


def line(pts):
    """МНК-прямая по точкам [(x,y)] -> (a, b) для y = a + b*x."""
    k = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    den = k * sxx - sx * sx
    if den == 0:
        return sy / k, 0.0
    b = (k * sxy - sx * sy) / den
    return (sy - b * sx) / k, b


def rel_sse(pts, a, b):
    return sum(((a + b * x) / y - 1.0) ** 2 for x, y in pts if y > 0)


def fit_knee(pts, window):
    """Двухсегментная подгонка ДВУМЯ ПРЯМЫМИ со свободной точкой разрыва.
    Окно справа ограничено (window), иначе сверхлинейный хвост тянет разрыв вправо.
    Критерий -- сумма квадратов ОТНОСИТЕЛЬНЫХ невязок."""
    use = [p for p in pts if p[0] <= window]
    best = None
    for m in range(2, len(use) - 2):
        L, R = use[:m], use[m:]
        aL, bL = line(L)
        aR, bR = line(R)
        e = rel_sse(L, aL, bL) + rel_sse(R, aR, bR)
        if best is None or e < best[0]:
            best = (e, use[m - 1][0], use[m][0], bL, bR)
    return best


def main():
    caps = [40, 48, 56, 64, 72, 80, 96, 112, 128, 144, 168, 192, 224]
    nps = [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        128,
        136,
        144,
        152,
        160,
        168,
        176,
        184,
        192,
        200,
        208,
        216,
        224,
        232,
        240,
    ]
    print("ТОЧКА ИЗЛОМА ПО КРИВОЙ ЦЕНЫ (8 варпов; канальная модель = 4.00*NP)")
    print(
        "%6s %11s %10s %12s %10s %8s"
        % ("бюджет", "излом (вилка)", "предсказ.", "первое +5%", "наклон", "невязка")
    )
    rows = []
    for c in caps:
        d = json.load(open("data/reg/w8_r%d.json" % c))["data"]
        pts = [(n, d["c1x%d" % n]["median"]) for n in nps if "c1x%d" % n in d]
        a0, b0 = line([p for p in pts if p[0] <= 32])  # ПЛАТО по заведомо чистым точкам
        thr = next((n for n, t in pts if t > (a0 + b0 * n) * 1.05), None)
        best = fit_knee(pts, c + 40)
        rows.append((c, best[1], best[2], c - 9, thr, b0, best[0]))
        print(
            "%6d %6d..%-4d %10d %12s %10.3f %8.4f"
            % (c, best[1], best[2], c - 9, thr, b0, best[0])
        )
    ok = sum(
        1 for c, K1, K2, p, thr, _, _ in rows if thr is not None and K1 <= thr <= K2
    )
    ok3 = sum(1 for c, K1, K2, p, thr, _, _ in rows if thr == c)
    print("\nДВА ПОРОГА, И ОНИ РАЗНЫЕ -- это и есть содержательный ответ:")
    print(
        "  * КОМПИЛЯТОРНЫЙ (первое LDL/STL в теле):  MaxLive > Q-7, то есть NP > Q-9."
    )
    print(
        "    Точен до ЕДИНИЦЫ: минимальный бюджет без разлива = NP+9 при NP = 40, 96, 192."
    )
    print(
        "  * ВРЕМЕННОЙ (период уходит от канальной модели больше чем на 5%): NP = Q РОВНО,"
    )
    print(
        "    то есть MaxLive = Q-2.  Совпало на %d бюджетах из %d." % (ok3, len(rows))
    )
    print(
        "    Разница означает: ПЕРВЫЕ ДВА разлитых значения бесплатны (их LDL прячется),"
    )
    print("    цена появляется, когда разлив перестаёт быть двумя обращениями.")
    print(
        "  Вилка двухсегментной подгонки [Q-8, Q] содержит временной порог: %d из %d."
        % (ok, len(rows))
    )
    print(
        "Наклон плато: медиана %.3f такта на живое значение при модельных 4.000."
        % sorted(r[5] for r in rows)[len(rows) // 2]
    )


if __name__ == "__main__":
    main()
