#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M4: СКВОЗНОЙ ФАЛЬСИФИКАТОР -- предсказать ПОРЯДОК форматов боевого ядра volta_fwd_ws.

ПРИЁМОЧНЫЙ ТЕСТ пункта 8/7. Модель ничего не знает об этих ядрах: все ставки взяты со стенда.
Замер -- НАШ СОБСТВЕННЫЙ (bench/fmt_ab.py, 41 раунд, парные отношения, чередование внутри
раунда, карта 1, чужих процессов 0), потому что числа в чужом дереве получены на другом
коммите и по формату 7 не воспроизводятся.

    Sk=8192   ф8 1.0588 > ф7 1.0556 > ф6 1.0417 > ф0
    Sk=32768  ф7 1.0624 > ф8 1.0596 > ф6 1.0405 > ф0
    Sk=131072 ф7 1.0652 > ф8 1.0602 > ф6 1.0419 > ф0

--core  считать только БЕЗУСЛОВНЫЙ ОСТОВ цикла (блоки, доминирующие обратную дугу).  Это не
        настройка, а условие ЗАКОННОСТИ: сумма по всем блокам -- оценка СВЕРХУ, и нижняя
        граница из неё границей не является.
"""

import os
import re
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor")
)
import tempo as T  # noqa: E402
import issue_slots as IS  # noqa: E402

SO = None
FMTS = [0, 6, 7, 8]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
CORE = "--core" in sys.argv or "--all-blocks" not in sys.argv
if args:
    SO = args[0]
if SO is None:
    SO = "../VLLM_fa2/work_e4m3/build/fa2sm70_prefill/fa2sm70_prefill.so"
if len(args) > 1:
    FMTS = [int(x) for x in args[1].split(",")]

MEAS = {
    8192: {0: 1.0000, 6: 1.0417, 7: 1.0556, 8: 1.0588},
    32768: {0: 1.0000, 6: 1.0405, 7: 1.0624, 8: 1.0596},
    131072: {0: 1.0000, 6: 1.0419, 7: 1.0652, 8: 1.0602},
}
REGS = {
    0: 134,
    1: 134,
    6: 134,
    7: 119,
    8: 134,
}  # cuobjdump -res-usage, разлива нет ни у кого

kern = T.load_kernels(SO)
pat = r"volta_fwd_wsILi32ELi64ELi256ELi2ELi4ELb1ELi64ELi1ELi1ELi256ELi(\d+)ELi0ELi0ELi0ELi0ELi1E"
rows = []
for k in kern:
    m = re.search(pat, k)
    if not m:
        continue
    f = int(m.group(1))
    if f not in FMTS:
        continue
    cfg = IS.CFG(kern[k])
    roles = IS.detect_roles(cfg)
    if not roles:
        continue
    a_all = T.analyse_roles(cfg, roles, core=False)
    a_core = T.analyse_roles(cfg, roles, core=True)
    tot = sum(r["n"] * r["warps"] for r in a_all["roles"].values())
    rows.append((f, a_all, a_core, tot))
rows.sort()

print("ФОРМАТ  выдач*варп  ResMII(все бл.)  ResMII(остов)  связывает  регистров")
for f, a, c, tot in rows:
    print(
        "%5d %11d %16.0f %14.0f  %-9s %8d"
        % (f, tot, a["ResMII"], c["ResMII"], c["binding"], REGS.get(f, 0))
    )

base = [r for r in rows if r[0] == FMTS[0]]
if base:
    ba, bc = base[0][1]["T_pred"], base[0][2]["T_pred"]
    print("\nОТНОШЕНИЯ к формату %d (>1 = ПРЕДСКАЗАНО БЫСТРЕЕ):" % FMTS[0])
    print(
        "%-8s %11s %11s %11s %11s %11s %11s"
        % ("формат", "ВСЕ блоки", "ОСТОВ", "счёт выдач", "8K", "32K", "128K")
    )
    for f, a, c, tot in rows:
        print(
            "%-8d %11.4f %11.4f %11.4f %11.4f %11.4f %11.4f"
            % (
                f,
                ba / a["T_pred"],
                bc / c["T_pred"],
                base[0][3] / float(tot),
                MEAS[8192].get(f, float("nan")),
                MEAS[32768].get(f, float("nan")),
                MEAS[131072].get(f, float("nan")),
            )
        )

    def order(d):
        return " > ".join("ф%d" % f for f in sorted(d, key=lambda x: -d[x]))

    pred_all = {f: ba / a["T_pred"] for f, a, c, t in rows}
    pred_core = {f: bc / c["T_pred"] for f, a, c, t in rows}
    print("\nПОРЯДОК:")
    print(
        "  по счёту выдач      %s"
        % order({f: base[0][3] / float(t) for f, a, c, t in rows})
    )
    print("  модель, ВСЕ блоки   %s" % order(pred_all))
    print(
        "  модель, ОСТОВ       %s   <-- законная форма (см. tempo.core_blocks)"
        % order(pred_core)
    )
    for sk in sorted(MEAS):
        print("  ЗАМЕР Sk=%-7d   %s" % (sk, order({f: MEAS[sk][f] for f in pred_core})))

    q = T.reg_budget(12)
    print(
        "\nРЕГИСТРЫ (launch_bounds(384,1) -> 12 варпов -> бюджет %d, порог MaxLive %d):"
        % (q, q - T.REG_OVERHEAD)
    )
    print(
        "  ни один формат к порогу не подходит (запас 27-42 значения), разлива нет ни у кого."
    )
    print("  ДВА КОНТРОЛЯ, УБИВАЮЩИЕ РЕГИСТРОВОЕ ОБЪЯСНЕНИЕ (оба на наших замерах):")
    print(
        "    * 6 против 8: регистров СТОЛЬКО ЖЕ (134=134), ответ ПОБИТОВО ТОТ ЖЕ, а время"
    )
    print("      различается на 1.8 % -- разница есть ТАМ, ГДЕ РЕГИСТРЫ ОДИНАКОВЫ;")
    print(
        "    * 7 против 8: регистров на 15 МЕНЬШЕ, а время различается на -0.3..+0.5 % и МЕНЯЕТ"
    )
    print(
        "      ЗНАК с длиной -- разницы НЕТ там, где регистры различаются сильнее всего."
    )
    print(
        "  Значит порядок 6/7/8 решают не регистры. Решает СТАТИЧЕСКИЙ ПЕРЕСЧЁТ РОЛИ ВЕЗУЩИХ:"
    )
    print(
        "  у формата 8 её мейнлуп это 25 блоков против 12 у формата 6 и 2 у формата 7, и сумма"
    )
    print("  по всем блокам начисляет условные ветви каждый оборот. Остов это снимает.")

json.dump(
    [
        {
            "fmt": f,
            "T_all": a["T_pred"],
            "T_core": c["T_pred"],
            "binding": c["binding"],
            "issues": tot,
            "regs": REGS.get(f),
            "roles_core": c["roles"],
        }
        for f, a, c, tot in rows
    ],
    open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "predict_fmt.json"
        ),
        "w",
    ),
    indent=1,
    ensure_ascii=False,
)
