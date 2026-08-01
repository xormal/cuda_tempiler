# -*- coding: utf-8 -*-
"""ПОРОЖДЕНИЕ ДВОЙНИКА БЭКВАРДА (attention_kernel_backward_batched_impl) ИЗ НАЛОЖЕНИЯ.

БОЕВОЕ ДЕРЕВО ТОЛЬКО ЧИТАЕТСЯ. Двойник целиком лежит в ./profiled/bwd/inc и
подставляется В НАЧАЛО пути включения: kernel_backward.h и fused_qk_gradk.h берутся из двойника,
ВСЁ остальное (mma_from_smem.h, mma_pipelined.h, эпилоги, cutlass) -- из боевого дерева.

ВОРОТА ДРЕЙФА -- смысл этого инструмента. Копия, которую правят руками, расходится с боевым ядром,
и разложение начинает описывать ЧУЖОЕ ядро (правдоподобные доли не того кода). Поэтому порождение
ПАДАЕТ, а не чинится молча, если:
  (1) md5 боевого исходника отличается от записанного в наложении -- называется файл и оба md5;
  (2) якорь правки не найден ИЛИ найден больше одного раза -- называется файл, номер правки и
      первые строки якоря.
Выход 2 = дрейф. Выход 0 = двойник порождён и соответствует записанному боевому состоянию.
"""

import hashlib
import json
import os
import shutil
import sys

# Корень боевого дерева. Переопределяется ТОЛЬКО ради фальсификатора ворот (tools/bwd_drift_test.sh):
# ворота, которые никогда не падали, ничего не гарантируют, а испортить боевое дерево ради проверки
# нельзя. Использованный корень пишется в MANIFEST и печатается -- подмена не может пройти молча.
PROD = os.environ.get("PHASE_PROD_ROOT", "../VLLM_fa2/solutions/fa2_sm70_cutlass_grade")
TOOLS = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("PHASE_TWIN_OUT", "./profiled/bwd/inc")
OVERLAY = os.environ.get("PHASE_OVERLAY", "./profiled/bwd/overlay_bwd.json")
# Заголовки, принадлежащие САМОМУ наложению (в боевом дереве их нет).
OWN = ["fmha_phase.h", "fmha_phase_bwd.h"]


def die(msg):
    sys.stderr.write(
        "\n[ДРЕЙФ] " + msg + "\n"
        "Двойник НЕ порождён. Переразметить: tools/mk_bwd_overlay.py, затем\n"
        "перепроверить нулевую маску побайтово (tools/bwd_twin_check.sh).\n"
    )
    sys.exit(2)


def head(s, n=3):
    return "\n    | ".join(s.split("\n")[:n])


def main():
    ov = json.load(open(OVERLAY, encoding="utf-8"))
    os.makedirs(OUT, exist_ok=True)
    print("боевой корень: %s" % PROD)
    man = {"prod_root": PROD, "sources": [], "own": []}
    for f in ov:
        rel = f["file"]
        p = os.path.join(PROD, rel)
        if not os.path.exists(p):
            die("боевой исходник ИСЧЕЗ: %s" % p)
        raw = open(p, "rb").read()
        cur = hashlib.md5(raw).hexdigest()
        if cur != f["md5"]:
            die(
                "боевой исходник ИЗМЕНИЛСЯ: %s\n  записан md5=%s\n  сейчас  md5=%s"
                % (p, f["md5"], cur)
            )
        text = raw.decode("utf-8")
        for i, e in enumerate(f["edits"]):
            n = text.count(e["anchor"])
            if n != 1:
                die(
                    "якорь правки #%d в %s найден %d раз (нужно ровно 1):\n    | %s"
                    % (i, rel, n, head(e["anchor"]))
                )
            text = text.replace(e["anchor"], e["replace"], 1)
        dst = os.path.join(OUT, os.path.basename(rel))
        open(dst, "w", encoding="utf-8").write(text)
        man["sources"].append(
            {
                "file": rel,
                "md5_prod": cur,
                "md5_twin": hashlib.md5(text.encode("utf-8")).hexdigest(),
                "edits": len(f["edits"]),
            }
        )
        print(
            "порождён %-24s (правок %2d, боевой md5 %s)"
            % (os.path.basename(rel), len(f["edits"]), cur[:12])
        )
    for h in OWN:
        src = os.path.join(TOOLS, h)
        shutil.copyfile(src, os.path.join(OUT, h))
        man["own"].append(
            {"file": h, "md5": hashlib.md5(open(src, "rb").read()).hexdigest()}
        )
        print("скопирован %-22s (собственный заголовок наложения)" % h)
    json.dump(
        man,
        open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )
    print(
        "ВОРОТА ДРЕЙФА ПРОЙДЕНЫ: двойник соответствует записанному состоянию боевого дерева."
    )


if __name__ == "__main__":
    main()
