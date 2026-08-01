# -*- coding: utf-8 -*-
"""ИЗВЛЕЧЕНИЕ НАЛОЖЕНИЯ ИЗ ПАРЫ «БОЕВОЙ ФАЙЛ -> РАЗМЕЧЕННЫЙ ФАЙЛ».

Запускается РЕДКО (один раз на редакцию разметки). Берёт боевой файл и его размеченную копию,
раскладывает разницу на правки вида (ЯКОРНЫЙ ТЕКСТ -> ЗАМЕНА) и записывает наложение в формате
tools/twin.py: список записей по файлам

    [ { "file": "<путь ОТНОСИТЕЛЬНО корня боевого дерева>",
        "md5":  "<md5 боевого файла В МОМЕНТ ПОРОЖДЕНИЯ>",
        "edits": [ {"anchor": "...", "replace": "...", "id": "...", "why": "..."} ] } ]

Номеров строк в результате НЕТ намеренно: файлы в этом проекте ездят под руками, и всякий отчёт,
привязанный к номерам, уже через день описывает не то место.

Якорь расширяется контекстом, пока не станет встречаться в боевом файле РОВНО ОДИН раз. Если
расширить до единственности не удалось -- извлечение ПАДАЕТ: неоднозначный якорь это готовая
тихая правка не того места.

    python3 mkoverlay.py <прод-файл> <размеченная-копия> <out.json> [--root R] [--append]
"""

import argparse
import difflib
import hashlib
import json
import os
import re
import sys

DEFAULT_ROOT = "../VLLM_fa2/solutions/fa2_sm70_cutlass_grade"


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def label(repl):
    """Человекочитаемая метка правки: первый маркер фазы, иначе первая непустая строка."""
    m = re.search(r"FMHA_PHASE(?:_ELSE)?\((\w+),\s*(\d+)\)", repl)
    if m:
        return "%s(бит %s)" % (m.group(1), m.group(2))
    m = re.search(r"\[ФАЗА[^\]]*\]", repl)
    if m:
        return m.group(0)
    for ln in repl.splitlines():
        if ln.strip():
            return ln.strip()[:60]
    return "правка"


def make_edits(prod_text, marked_text, min_ctx=1, max_ctx=40):
    a = prod_text.splitlines(keepends=True)
    b = marked_text.splitlines(keepends=True)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    edits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        for ctx in range(min_ctx, max_ctx + 1):
            lo, hi = max(0, i1 - ctx), min(len(a), i2 + ctx)
            anchor = "".join(a[lo:hi])
            if anchor and prod_text.count(anchor) == 1:
                repl = "".join(a[lo:i1]) + "".join(b[j1:j2]) + "".join(a[i2:hi])
                edits.append({"anchor": anchor, "replace": repl})
                break
        else:
            raise SystemExit(
                "НЕОДНОЗНАЧНЫЙ ЯКОРЬ: не удалось расширить до единственности, строки %d..%d"
                % (i1 + 1, i2 + 1)
            )
    # склейка правок с перекрывающимися якорями: иначе две правки метят в один текст
    merged = []
    for e in edits:
        if merged:
            prev = merged[-1]
            pi = prod_text.index(prev["anchor"])
            pe = pi + len(prev["anchor"])
            ci = prod_text.index(e["anchor"])
            if ci < pe:
                joint = prod_text[pi : ci + len(e["anchor"])]
                if prod_text.count(joint) != 1:
                    raise SystemExit("склейка дала неоднозначный якорь")
                prev["replace"] = prev["replace"] + e["replace"][pe - ci :]
                prev["anchor"] = joint
                continue
        merged.append(e)
    for n, e in enumerate(merged):
        e["id"] = "%02d-%s" % (n, label(e["replace"]))
        e["why"] = "разметка фазы для снятия по FMHA_STRIP_MASK"
    return merged


def selfcheck(prod_text, marked_text, edits):
    """Применить прямо здесь и потребовать ПОБАЙТОВОГО совпадения с размеченной копией.
    Без этого «наложение извлеклось» не значит ничего."""
    t = prod_text
    for e in edits:
        if t.count(e["anchor"]) != 1:
            raise SystemExit("самопроверка: якорь не единственный (%s)" % e["id"])
        t = t.replace(e["anchor"], e["replace"])
    if t != marked_text:
        raise SystemExit(
            "САМОПРОВЕРКА ПРОВАЛЕНА: наложение не воспроизводит размеченную копию"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prod")
    ap.add_argument("marked")
    ap.add_argument("out")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument(
        "--append",
        action="store_true",
        help="дописать запись в существующее наложение (несколько файлов)",
    )
    a = ap.parse_args()

    prod_text = open(a.prod, encoding="utf-8").read()
    marked_text = open(a.marked, encoding="utf-8").read()
    edits = make_edits(prod_text, marked_text)
    selfcheck(prod_text, marked_text, edits)

    rel = os.path.relpath(os.path.abspath(a.prod), os.path.abspath(a.root))
    entry = {"file": rel, "md5": md5(a.prod), "edits": edits}

    doc = []
    if a.append and os.path.exists(a.out):
        doc = json.load(open(a.out, encoding="utf-8"))
        if isinstance(doc, dict):
            doc = doc["files"]
        doc = [e for e in doc if e["file"] != rel]
    doc.append(entry)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print("%s: правок %d, md5 %s -> %s" % (rel, len(edits), entry["md5"], a.out))


if __name__ == "__main__":
    main()
