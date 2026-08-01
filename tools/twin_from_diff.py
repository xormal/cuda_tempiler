# -*- coding: utf-8 -*-
"""Превратить УЖЕ РАЗМЕЧЕННУЮ копию файла в НАЛОЖЕНИЕ (якорь -> замена) для twin.py.

Зачем отдельным инструментом: разметка прошлой попытки продумана и проверена, а переносить её
руками в якоря -- значит внести опечатку там, где её никто не увидит (якорь, отличающийся одним
пробелом, даёт ОТКАЗ, а не тихую ошибку -- но время съедает). Здесь якоря считаются машинно.

КАК. difflib по строкам даёт куски различий. Каждый кусок расширяется контекстом (поровну вверх и
вниз), пока текст якоря не станет встречаться в боевом файле РОВНО ОДИН РАЗ. Вставки (в боевом
файле нет ни строки) обязаны иметь непустой якорь -- контекст берётся сверху.

САМОПРОВЕРКА: применив полученные якоря к боевому файлу, инструмент обязан получить размеченную
копию ПОБАЙТОВО. Если нет -- отказ, наложение не пишется.

    python3 twin_from_diff.py --prod A.cuh --marked B.cuh --rel fa2_sm70/csrc/A.cuh \
        [--drop-hunk N ...] > часть_наложения.json
"""

import argparse
import difflib
import hashlib
import json
import sys


def build_edits(prod_lines, mark_lines, prod_text, drop=()):
    sm = difflib.SequenceMatcher(None, prod_lines, mark_lines, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
    edits, kept = [], []
    for idx, (tag, i1, i2, j1, j2) in enumerate(ops):
        if idx in drop:
            continue
        lo, hi = i1, i2
        ctx = 0
        while True:
            a = max(0, lo - ctx)
            b = min(len(prod_lines), hi + ctx)
            anchor = "".join(prod_lines[a:b])
            repl = (
                "".join(prod_lines[a:i1])
                + "".join(mark_lines[j1:j2])
                + "".join(prod_lines[i2:b])
            )
            if anchor and prod_text.count(anchor) == 1:
                edits.append({"anchor": anchor, "replace": repl})
                kept.append((idx, tag, i1 + 1, i2 + 1, a + 1, b))
                break
            ctx += 1
            if ctx > 60:
                raise SystemExit(
                    "кусок #%d (строки %d-%d): за 60 строк контекста якорь так и не "
                    "стал единственным" % (idx, i1 + 1, i2 + 1)
                )
    return edits, kept, len(ops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", required=True)
    ap.add_argument("--marked", required=True)
    ap.add_argument("--rel", required=True)
    ap.add_argument("--drop-hunk", type=int, action="append", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prod_text = open(args.prod, encoding="utf-8").read()
    mark_text = open(args.marked, encoding="utf-8").read()
    prod_lines = prod_text.splitlines(keepends=True)
    mark_lines = mark_text.splitlines(keepends=True)
    drop = set(args.drop_hunk)

    edits, kept, nops = build_edits(prod_lines, mark_lines, prod_text, drop)

    # --- самопроверка: якоря, приложенные к боевому файлу, обязаны дать разметку побайтово
    spans = []
    for e in edits:
        assert prod_text.count(e["anchor"]) == 1
        b = prod_text.index(e["anchor"])
        spans.append((b, b + len(e["anchor"]), e["replace"]))
    spans.sort()
    out, pos = [], 0
    for b, en, r in spans:
        out.append(prod_text[pos:b])
        out.append(r)
        pos = en
    out.append(prod_text[pos:])
    got = "".join(out)

    ent = {
        "file": args.rel,
        "md5": hashlib.md5(prod_text.encode("utf-8")).hexdigest(),
        "edits": edits,
    }
    js = json.dumps(ent, ensure_ascii=False, indent=2)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(js)
    else:
        print(js)

    exact = got == mark_text
    print(
        "# кусков различий: %d, взято: %d, отброшено: %s"
        % (nops, len(edits), sorted(drop)),
        file=sys.stderr,
    )
    for idx, tag, i1, i2, a, b in kept:
        print(
            "#   кусок %2d %-8s боевые строки %5d-%-5d  якорь %5d-%-5d (%d строк)"
            % (idx, tag, i1, i2, a, b, b - a + 1),
            file=sys.stderr,
        )
    if drop:
        print(
            "# САМОПРОВЕРКА пропущена (есть отброшенные куски): совпадение с размеченной "
            "копией: %s" % ("ДА" if exact else "нет -- и это ожидаемо"),
            file=sys.stderr,
        )
    elif exact:
        print(
            "# САМОПРОВЕРКА: наложение воспроизводит размеченную копию ПОБАЙТОВО.",
            file=sys.stderr,
        )
    else:
        raise SystemExit(
            "САМОПРОВЕРКА ПРОВАЛЕНА: наложение не воспроизводит размеченную копию"
        )


if __name__ == "__main__":
    main()
