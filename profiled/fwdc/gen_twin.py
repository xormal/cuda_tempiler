# -*- coding: utf-8 -*-
"""ВТОРОЙ, НЕЗАВИСИМЫЙ ГЕНЕРАТОР ДВОЙНИКА (сверка против tools/twin.py).

Основной путь -- `tools/twin.py gen --overlay overlays/fwd_cutlass.json`. Этот файл делает то же
самое своим кодом и служит ЛОМКОЙ СИММЕТРИИ: два разбора одного наложения, написанные врозь,
обязаны дать ПОБАЙТОВО один двойник. Правило отсюда же: две сверки, читающие один промежуточный
буфер, соглашаются, будучи обе неверны, -- поэтому у ворот дрейфа две независимые реализации, а не
одна с двумя вызовами.

БОЕВОЕ ДЕРЕВО ОТКРЫВАЕТСЯ ТОЛЬКО НА ЧТЕНИЕ. Всё, что пишется, лежит под --out.

ВОРОТА ДРЕЙФА (падают громко, код возврата 3):
  1. md5 каждого источника сверяется с записанным в момент извлечения наложения;
  2. каждый ЯКОРЬ обязан находиться в источнике РОВНО ОДИН раз.
Молча собрать устаревший двойник генератор не имеет права: устаревший двойник даёт правдоподобные
доли ЧУЖОГО ядра.

Раскладка (почему копируется ОДИН файл, а не дерево): сборка идёт с -I на каталог двойника ПЕРЕД
боевым, кавычечный include находит там только kernel_forward.h / fwd_phase.h / fmha_phase.h, а всё
остальное (gemm/*, epilogue/*, cutlass/*) приезжает из БОЕВОГО дерева. Значит двойник не может
разойтись с боевым по вложенным заголовкам в принципе -- он их не копирует.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILED = os.path.dirname(HERE)
TOOLS = "./tools"
ROOT = "../VLLM_fa2/solutions/fa2_sm70_cutlass_grade"
OVERLAY = os.path.join(PROFILED, "overlays", "fwd_cutlass.json")


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def die(msg):
    sys.stderr.write("\n!!! ВОРОТА ДРЕЙФА: " + msg + "\n\n")
    sys.exit(3)


def entries(doc):
    return doc["files"] if isinstance(doc, dict) else doc


def apply_entry(ent, root, src_override=None):
    src = src_override or os.path.join(root, ent["file"])
    if not os.path.exists(src):
        die("источник наложения ИСЧЕЗ: %s" % src)
    have = md5(src)
    if have != ent["md5"]:
        die(
            "ИСХОДНИК ИЗМЕНИЛСЯ\n"
            "    файл : %s\n"
            "    md5 в наложении : %s\n"
            "    md5 сейчас      : %s\n"
            "  Наложение снималось с ДРУГОЙ редакции файла. Двойник поверх неё дал бы\n"
            "  правдоподобные доли ЧУЖОГО ядра. Пересними разметку и перевыпусти наложение:\n"
            "    python3 ../mkoverlay.py <прод> <размеченная копия> <out.json>"
            % (src, ent["md5"], have)
        )
    text = open(src, encoding="utf-8").read()
    for n, e in enumerate(ent["edits"]):
        cnt = text.count(e["anchor"])
        if cnt != 1:
            head = e["anchor"].strip().splitlines()[0][:100]
            die(
                "ЯКОРЬ НЕ НАЙДЕН ОДИН РАЗ\n"
                "    файл  : %s\n"
                "    правка: #%d из %d (%s)\n"
                "    якорь начинается с: %s\n"
                "    найдено вхождений : %d (нужно ровно 1)\n"
                "  Правка легла бы не туда (или не легла бы вовсе) -- сборка остановлена."
                % (src, n, len(ent["edits"]), e.get("id", "?"), head, cnt)
            )
        text = text.replace(e["anchor"], e.get("replace", e.get("repl")))
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default=OVERLAY)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=os.path.join(HERE, "twin"))
    ap.add_argument(
        "--source", default=None, help="переопределить путь источника (проверка ворот)"
    )
    ap.add_argument(
        "--check", action="store_true", help="только ворота, ничего не писать"
    )
    a = ap.parse_args()

    doc = json.load(open(a.overlay, encoding="utf-8"))
    texts = {}
    for ent in entries(doc):
        texts[ent["file"]] = apply_entry(ent, a.root, a.source)
    if a.check:
        print(
            "ВОРОТА ПРОЙДЕНЫ: файлов %d, правок %d, все якоря единственны"
            % (len(texts), sum(len(e["edits"]) for e in entries(doc)))
        )
        return

    for rel, text in texts.items():
        dst = os.path.join(
            a.out, os.path.basename(os.path.dirname(rel)), os.path.basename(rel)
        )
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "w", encoding="utf-8").write(text)
        print("  %s -> %s" % (rel, dst))
    dst_dir = os.path.join(a.out, "fmha_kernel")
    shutil.copyfile(
        os.path.join(HERE, "fwd_phase.h"), os.path.join(dst_dir, "fwd_phase.h")
    )
    shutil.copyfile(
        os.path.join(TOOLS, "fmha_phase.h"), os.path.join(dst_dir, "fmha_phase.h")
    )
    print("двойник порождён (независимая реализация): %s" % a.out)


if __name__ == "__main__":
    main()
