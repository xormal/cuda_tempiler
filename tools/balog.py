# -*- coding: utf-8 -*-
"""ЖУРНАЛ ПРОБ БАЛАНСИРОВКИ. Ручной заход обязан копить материал для решателя (T14/v2).

ЗАЧЕМ. Балансируем конвейер руками -- и каждая проба есть ТОЧКА в том самом пространстве, которое
потом будет искать решатель. Если писать их вразнобой по отчётам, через месяц получим отбалансированное
ядро и НОЛЬ данных, на которых решатель можно проверить. А гейт годности T14 требует именно этого:
модель обязана САМА воспроизвести уже известные ответы (разбиение 4+8 и его 1.22-1.46x; отвергнуть
16-варповый коллектив, замеренный как 1.50x медленнее).

ГЛАВНАЯ КОЛОНКА -- НЕ ВРЕМЯ, А ПАРА (ПРЕДСКАЗАНО, ЗАМЕРЕНО). Журнал, где лежит только замеренное,
годен для выбора победителя и НЕ годен для проверки модели: по нему нельзя сказать, ошибалась ли она.
Поэтому `predicted` пишется ДО прогона и не правится после -- иначе модель станет непроверяемой,
подгоняясь под каждый новый факт.

ЗАПИСЬ:
    python3 tools/balog.py add --axes n_carry=4,n_compute=8,BM=64,BN=128,BK=64,bits=16,M=32 \\
        --shape gate_up --predicted-us 172 --measured-us 165 --note "база, ролевое 4+8"
ЧТЕНИЕ:
    python3 tools/balog.py show                 # всё, с ошибкой модели
    python3 tools/balog.py show --axis bits     # разрез по одной оси
    python3 tools/balog.py check                # ГЕЙТ: воспроизводит ли журнал известные ответы
"""

import argparse
import json
import os
import sys

LOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "balance_log.jsonl"
)


def _rows():
    if not os.path.exists(LOG):
        return []
    out = []
    for i, l in enumerate(open(LOG, encoding="utf-8")):
        l = l.strip()
        if not l:
            continue
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError as e:
            # НЕ глотать: битая строка означает, что часть проб потеряна, а журнал выглядит целым.
            print(f"ОТКАЗ: строка {i + 1} журнала не разбирается: {e}", file=sys.stderr)
            sys.exit(2)
    return out


def _axes(s):
    d = {}
    for kv in s.split(","):
        if not kv.strip():
            continue
        k, _, v = kv.partition("=")
        k, v = k.strip(), v.strip()
        try:
            d[k] = int(v)
        except ValueError:
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    return d


def cmd_add(a):
    if a.measured_us is not None and a.predicted_us is None:
        print(
            "ОТКАЗ: --predicted-us обязателен. Журнал без предсказания не проверяет модель,\n"
            "       а только выбирает победителя. Если модели ещё нет -- пиши --predicted-us -1.",
            file=sys.stderr,
        )
        sys.exit(2)
    rec = dict(
        axes=_axes(a.axes),
        shape=a.shape,
        predicted_us=a.predicted_us,
        measured_us=a.measured_us,
        correct=a.correct,
        note=a.note or "",
        source=a.source or "",
    )
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("записано:", json.dumps(rec, ensure_ascii=False))


def cmd_show(a):
    rows = _rows()
    if a.axis:
        rows = [r for r in rows if a.axis in r["axes"]]
        rows.sort(key=lambda r: r["axes"][a.axis])
    if not rows:
        print("журнал пуст")
        return
    print(
        f"{'форма':<17}{'оси':<49}{'предск,мкс':>11}{'замер,мкс':>10}{'ошибка':>9}  примечание"
    )
    for r in rows:
        ax = ",".join(f"{k}={v}" for k, v in r["axes"].items())
        p, m = r.get("predicted_us"), r.get("measured_us")
        err = f"{100 * (p - m) / m:+.1f}%" if (p and m and p > 0) else "--"
        print(
            f"{r['shape'][:16]:<17}{ax[:48]:<49}{(p if p else 0):>11.1f}{(m if m else 0):>10.1f}{err:>9}  {r.get('note', '')}"
        )
    ok = [r for r in rows if r.get("predicted_us", -1) > 0 and r.get("measured_us")]
    if ok:
        e = [abs(r["predicted_us"] - r["measured_us"]) / r["measured_us"] for r in ok]
        e.sort()
        print(
            f"\nточек с предсказанием: {len(ok)} из {len(rows)}; "
            f"медиана |ошибки| {100 * e[len(e) // 2]:.1f} %, худшая {100 * e[-1]:.1f} %"
        )
    else:
        print(
            f"\nточек с предсказанием: 0 из {len(rows)} -- модель по этому журналу НЕ проверяема"
        )


def cmd_check(a):
    """ГЕЙТ ГОДНОСТИ: воспроизводит ли накопленное уже ИЗВЕСТНЫЕ ответы.

    Оба ответа замерены давно и лежат в отчётах, гадать не нужно:
      * ролевое разбиение 4+8 быстрее двухблочного варианта в 1.22-1.46 раза;
      * 16-варповый коллектив ХУЖЕ (замерен 1.50x медленнее torch на пределе) -- его журнал обязан
        отвергать, а не «не находить».
    """
    rows = _rows()

    def find(**kw):
        return [r for r in rows if all(r["axes"].get(k) == v for k, v in kw.items())]

    bad = []
    roles = find(n_carry=4, n_compute=8)
    if not roles:
        bad.append(
            "нет ни одной пробы ролевого 4+8 -- известный ответ не воспроизведён"
        )
    coll = [
        r
        for r in rows
        if r["axes"].get("n_compute") == 16 or r["axes"].get("collective") == 1
    ]
    if not coll:
        bad.append(
            "нет пробы 16-варпового коллектива -- известный ОТРИЦАТЕЛЬНЫЙ ответ не воспроизведён"
        )
    print("ГЕЙТ ГОДНОСТИ ЖУРНАЛА (T14):")
    for b in bad:
        print("  ОТКАЗ:", b)
    if not bad:
        print(
            "  оба известных ответа присутствуют -- журнал годен как проверочный набор"
        )
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add")
    p.set_defaults(f=cmd_add)
    p.add_argument(
        "--axes", required=True, help="n_carry=4,n_compute=8,BM=64,... через запятую"
    )
    p.add_argument("--shape", required=True)
    p.add_argument("--predicted-us", type=float, default=None)
    p.add_argument("--measured-us", type=float, default=None)
    p.add_argument(
        "--correct",
        default="",
        help="как сверена корректность (гейт РАНЬШЕ секундомера)",
    )
    p.add_argument("--note", default="")
    p.add_argument("--source", default="", help="файл/скрипт прогона")
    p = sub.add_parser("show")
    p.set_defaults(f=cmd_show)
    p.add_argument("--axis", default="")
    p = sub.add_parser("check")
    p.set_defaults(f=cmd_check)
    a = ap.parse_args()
    sys.exit(a.f(a) or 0)


if __name__ == "__main__":
    main()
