# -*- coding: utf-8 -*-
"""ФАЛЬСИФИКАТОР САМИХ ВОРОТ ДРЕЙФА.

Ворота, которые никто не ронял, -- это не ворота, а комментарий. Здесь они роняются НАМЕРЕННО
тремя способами, и каждый обязан дать код возврата 3 и сообщение с именем файла.

Все подделки делаются в ./profiled/drift/, БОЕВОЕ ДЕРЕВО ТОЛЬКО ЧИТАЕТСЯ.

  1. ИСХОДНИК УЕХАЛ -- в файле изменён один байт, md5 больше не совпадает.
     Это ловит «кто-то тронул боевой заголовок, а двойник собрался старым».
  2. ЯКОРЬ ПРОПАЛ -- строка внутри якоря переписана, а md5 в наложении ОБНОВЛЁН.
     Это ловит самый опасный случай: наложение формально «свежее», но правка легла бы не туда.
     (Именно так выглядит дрейф после безобидного рефакторинга.)
  3. ЯКОРЬ РАЗДВОИЛСЯ -- размеченный кусок продублирован; правка легла бы в оба места.
Контроль: неиспорченный источник обязан ворота ПРОЙТИ (иначе тест доказывал бы только то, что
генератор всегда падает).
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "drift")
PROD = (
    "../VLLM_fa2/solutions/fa2_sm70_cutlass_grade/fa2_src/fmha_kernel/kernel_forward.h"
)
OVERLAY = os.path.join(os.path.dirname(HERE), "overlays", "fwd_cutlass.json")


def md5b(b):
    return hashlib.md5(b).hexdigest()


def run(src, overlay):
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "gen_twin.py"),
            "--check",
            "--source",
            src,
            "--overlay",
            overlay,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return r.returncode, r.stdout.strip()


def run_canon(src, overlay):
    """Тот же прогон КАНОНИЧЕСКИМ инструментом tools/twin.py: подделка кладётся в фальшивый
    корень по тому же относительному пути. Обе реализации ворот обязаны отбить обе подделки --
    иначе «две сверки согласились, будучи обе неверны»."""
    rel = "fa2_src/fmha_kernel/kernel_forward.h"
    root = os.path.join(D, "root_" + os.path.basename(src))
    dst = os.path.join(root, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    r = subprocess.run(
        [
            sys.executable,
            "./tools/twin.py",
            "check",
            "--overlay",
            overlay,
            "--root",
            root,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return r.returncode


def case(name, text, refresh_md5):
    src = os.path.join(D, "src_%s.h" % name)
    ov = os.path.join(D, "ov_%s.json" % name)
    open(src, "wb").write(text)
    doc = json.load(open(OVERLAY, encoding="utf-8"))
    ent = doc["files"][0] if isinstance(doc, dict) else doc[0]
    if refresh_md5:
        ent["md5"] = md5b(text)
    json.dump(doc, open(ov, "w", encoding="utf-8"), ensure_ascii=False)
    rc, out = run(src, ov)
    rc2 = run_canon(src, ov)
    head = [l for l in out.splitlines() if l.strip()]
    print("--- %s : gen_twin.py -> %d ; tools/twin.py -> %d" % (name, rc, rc2))
    for l in head:
        print("    " + l)
    if (rc == 0) != (rc2 == 0):
        print("    ПРОВАЛ: две реализации ворот РАЗОШЛИСЬ в вердикте")
        return 99, out
    return rc, out


def main():
    os.makedirs(D, exist_ok=True)
    base = open(PROD, "rb").read()
    ok = True

    rc, out = case("КОНТРОЛЬ_чистый", base, refresh_md5=False)
    if rc != 0:
        ok = False
        print("    ПРОВАЛ: чистый источник обязан пройти ворота")

    # 1. один байт: комментарий -> md5 уехал
    rc, _ = case(
        "1_md5_уехал",
        base.replace(b"// 7. Calculate logsumexp", b"// 7. calculate logsumexp", 1),
        refresh_md5=False,
    )
    if rc != 3:
        ok = False
        print("    ПРОВАЛ: ждали код 3")

    # 2. якорь переписан, md5 ОБНОВЛЁН -- ворота обязаны удержаться на якоре
    anchor_line = b"      MM0::B2bGemm::accumToSmem(\n"
    assert base.count(anchor_line) == 1
    rc, _ = case(
        "2_якорь_пропал",
        base.replace(anchor_line, b"      MM0::B2bGemm::accum_to_smem(\n", 1),
        refresh_md5=True,
    )
    if rc != 3:
        ok = False
        print("    ПРОВАЛ: ждали код 3")

    # 3. якорь раздвоился
    dup = b"    if (kKeepOutputInRF) {\n      constexpr bool kIsFirst = true;\n"
    assert base.count(dup) == 1
    rc, _ = case(
        "3_якорь_раздвоился", base.replace(dup, dup + dup, 1), refresh_md5=True
    )
    if rc != 3:
        ok = False
        print("    ПРОВАЛ: ждали код 3")

    print()
    print(
        "ИТОГ: "
        + (
            "ВОРОТА РАБОТАЮТ (контроль прошёл, все три подделки отбиты)"
            if ok
            else "ВОРОТА ДЫРЯВЫ"
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
