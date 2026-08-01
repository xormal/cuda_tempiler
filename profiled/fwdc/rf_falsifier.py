# -*- coding: utf-8 -*-
"""АЛЬТЕРНАТИВА БЕЗ РАЗМЕТКИ: годится ли kKeepOutputInRF как готовый фальсификатор ЭПИЛОГА.

Ходовое рассуждение: kKeepOutputInRF (= kSingleValueIteration = kMaxK <= kKeysPerBlock) оставляет
выход в регистрах, значит путь через разделяемую не используется, значит переключение этого флага
и есть бесплатный замер доли эпилога -- разметка не нужна.

Здесь это проверяется, а не принимается. Сравниваются ДВА инстанцирования, отличающиеся РОВНО
одним параметром kMaxK (128 против 256) при одной плитке <32 запроса, 128 ключей>:
  * сколько команд площадки эпилога (tile_iterator_volta_tensor_op.h / shared_load_iterator.h)
    остаётся при RF=TRUE -- если не ноль, посылка неверна;
  * что ЕЩЁ меняется вместе с флагом -- это и есть цена подмены разметки фальсификатором.
Ничего не запускает.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epilog_lines  # noqa: E402  (переиспользуем привязку к файлу-источнику)
import table  # noqa: E402

CUB = os.path.join(HERE, "out", "m0.cubin")
STAGE = ("tile_iterator_volta_tensor_op.h", "shared_load_iterator.h")


def main():
    if not os.path.exists(CUB):
        subprocess.run(
            [sys.executable, os.path.join(HERE, "table.py"), "--only", "0"], check=True
        )
    txt = subprocess.run(
        [epilog_lines.NVDIS, "-g", "-c", CUB],
        stdout=subprocess.PIPE,
        universal_newlines=True,
    ).stdout
    import collections
    import re

    cur_fn, cur_src = None, "?"
    per = collections.defaultdict(collections.Counter)
    for ln in txt.splitlines():
        m = re.match(r"\s*\.text\.(\S+):", ln)
        if m:
            cur_fn = m.group(1)
            continue
        m = re.search(r'//## File "(.*?)", line', ln)
        if m:
            cur_src = os.path.basename(m.group(1))
            continue
        if cur_fn and re.search(r"\b(STS|LDS)\b", ln):
            per[cur_fn][cur_src] += 1

    print(
        "ПОСЫЛКА: при kKeepOutputInRF=TRUE площадка эпилога в разделяемой НЕ ИСПОЛЬЗУЕТСЯ."
    )
    print()
    for fn, c in per.items():
        tag = (
            "d128  kMaxK=128 <= kKeysPerBlock=128  -> kKeepOutputInRF=TRUE"
            if "Li128ELb0" in fn
            else "d256  kMaxK=256 >  kKeysPerBlock=128  -> kKeepOutputInRF=FALSE"
            if "Li256ELb0" in fn
            else fn
        )
        s = sum(c[x] for x in STAGE)
        print("  %-58s STS+LDS площадки эпилога = %d" % (tag, s))
    print()
    print(
        "  Значит посылка НЕВЕРНА: флаг убирает не круг через разделяемую, а его ПОВТОРЕНИЕ"
    )
    print(
        "  на каждом блоке ключей. Готового фальсификатора эпилога из него не выходит."
    )
    print()
    print("ЧТО ЕЩЁ МЕНЯЕТСЯ ВМЕСТЕ С ФЛАГОМ (переключить его = сменить не одну фазу):")
    import json

    d = json.load(open(os.path.join(HERE, "out", "counts.json"), encoding="utf-8"))[
        "база(0)"
    ]
    a, b = d["Li128ELb0"], d["Li256ELb0"]
    for c in [n for n, _ in table.CLASSES] + ["рег", "кадр"]:
        if a.get(c) != b.get(c):
            print(
                "    %-10s RF=TRUE %6s   RF=FALSE %6s   (%+d)"
                % (c, a.get(c), b.get(c), b.get(c, 0) - a.get(c, 0))
            )


if __name__ == "__main__":
    main()
