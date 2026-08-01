# -*- coding: utf-8 -*-
"""ДВОЙНИК ЯДРА: наложение поверх БОЕВОГО дерева, ПОРОЖДАЕМОЕ, а не ведомое руками.

ЗАЧЕМ ИМЕННО ТАК. Размеченную копию, которую правят руками, невозможно удержать в согласии с
боевым ядром: боевое дерево живёт, копия отстаёт МОЛЧА, и разложение по фазам начинает описывать
ДРУГОЕ ядро. Доли при этом остаются правдоподобными -- это тот класс ошибок, который не ловится
ни одним тестом на значения. Поэтому двойник ПОРОЖДАЕТСЯ каждый раз заново из боевого исходника
плюс наложение, а наложение обязано ПАДАТЬ при любом расхождении.

КОМАНДЫ
-------
    twin.py check --overlay O.json [--root R]                 только ворота дрейфа, код != 0
    twin.py gen   --overlay O.json [--root R] [--out-root D]  ворота + порождение + manifest.json
    twin.py build --twin ИМЯ [--mask N]                       сборка ОТДЕЛЬНЫМ расширением
    twin.py diff  --twin ИМЯ [--file П] [--stat]              чем двойник отличается от БОЕВОГО СЕЙЧАС
    twin.py list                                              что порождено и не протухло ли
    twin.py apply --overlay O --root R --out DIR              (прежнее имя gen, порождение в свой каталог)
    twin.py freeze --overlay O --root R --yes                 ПЕРЕзакрепить md5 после ручной сверки

ФОРМАТ НАЛОЖЕНИЯ -- два вида, оба читаются:

(1) СПИСОК по файлам (основной; его порождает profiled/mkoverlay.py):

    [ { "file":  "<путь ОТНОСИТЕЛЬНО корня боевого дерева>",
        "md5":   "<md5 боевого файла В МОМЕНТ ПОРОЖДЕНИЯ наложения>",
        "edits": [ { "anchor":  "<ТОЧНЫЙ фрагмент боевого текста>",
                     "replace": "<чем он становится в двойнике>",
                     "mode":    "replace|insert_before|insert_after",   # необязательно
                     "required": true,                                  # необязательно
                     "id": "...", "why": "зачем эта правка" } ] } ]

(2) СЛОВАРЬ, когда двойнику нужны свои имя/сборка/добавочные файлы:

    { "name": "fwd_ws", "version": "1", "prod_root": "<боевое дерево>",
      "files": [ <записи вида (1)> ],
      "add":   [ {"path": "<куда в двойнике>", "from": "<откуда скопировать>"} ],
      "write": [ {"path": "<куда в двойнике>", "text": "..."} ],
      "prod_include_dirs": ["fa2_src/cutlass/include", ...],
      "build": { "harness": "<.cu в двойнике>", "env": {...},
                 "steps": [ {"name": "...", "cmd": "... {nvcc} {inc} {mask} {outdir} ..."} ] } }

Правки адресуются ЯКОРНЫМ ТЕКСТОМ, а не номерами строк: файлы в этом проекте уже поехали под
руками, и номера из прежних отчётов недействительны. Ключи "id"/"why" читаются только человеком.

ВОРОТА ДРЕЙФА -- пять независимых, любые падают с явным сообщением и НЕНУЛЕВЫМ кодом возврата:
    (1) боевого файла нет                             -> сказать, какой файл;
    (2) md5 боевого файла отличается от записанного   -> сказать, какой файл и оба md5;
    (3) якорь НЕ НАЙДЕН                               -> сказать, какой файл и какой якорь;
    (4) якорь найден БОЛЕЕ ОДНОГО РАЗА                -> адрес неоднозначен, это тоже отказ;
    (5) якоря перекрываются между собой               -> две правки метят в один текст.
Ворота НЕ ЧИНЯТСЯ автоматически: устаревший двойник, собранный молча, даёт правдоподобные доли
ЧУЖОГО ядра. Ворота стоят и на `build` тоже -- боевое дерево могло уехать ПОСЛЕ порождения.

Все правки применяются к ИСХОДНОМУ тексту по смещениям, найденным ДО применения. Значит порядок
правок в наложении ни на что не влияет, и одна правка не может создать или уничтожить якорь другой.
"""

import argparse
import datetime
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_ROOT = os.path.abspath(os.path.join(HERE, "..", "profiled"))
# Боевое дерево этого проекта. ТОЛЬКО ЧТЕНИЕ: инструмент в него не пишет ни байта.
DEFAULT_ROOT = "../VLLM_fa2/solutions/fa2_sm70_cutlass_grade"

NVCC = "/opt/conda/miniconda3/envs/cuda128/bin/nvcc"
CUOBJDUMP = "/opt/conda/miniconda3/envs/cuda128/bin/cuobjdump"
# Окружение сборки на этой машине. Наложение может переопределить его через build.env.
DEFAULT_ENV = {
    "CUDA_HOME": "/opt/conda/miniconda3/envs/cuda128",
    "CC": "/usr/bin/gcc",
    "CXX": "/usr/bin/g++",
    "CUDAHOSTCXX": "/usr/bin/g++",
}
# Куда смотреть за НЕразмеченными заголовками. Идут ПОСЛЕ каталогов двойника, поэтому
# размеченный файл перекрывает боевой одноимённый, а всё остальное берётся из боевого дерева
# как есть -- копировать cutlass целиком не нужно.
DEFAULT_PROD_INCLUDES = [
    "fa2_sm70/csrc",
    "fa2_src/fmha_kernel",
    "fa2_src/cutlass/include",
]
DEFAULT_BUILD_STEPS = [
    {
        "name": "nvcc -cubin",
        "cmd": "{nvcc} -arch=sm_70 -O3 -std=c++17 -cubin -lineinfo -Xptxas -v "
        "-Wno-deprecated-gpu-targets -ccbin /usr/bin/g++ {inc} "
        "-DFMHA_STRIP_MASK={mask}u {extra} -o {outdir}/{name}.cubin {harness}",
    },
    {
        "name": "cuobjdump -sass",
        "cmd": "{cuobjdump} -sass {outdir}/{name}.cubin > {outdir}/{name}.sass",
    },
]


# ======================================================================================
# мелочи
# ======================================================================================
def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_text(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _short(s, n=110):
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + " ...(+%d симв.)" % (len(s) - n)


def _lineno(text, pos):
    return text.count("\n", 0, pos) + 1


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class DriftError(Exception):
    pass


# ======================================================================================
# наложение
# ======================================================================================
def _check_entries(entries, path):
    for i, ent in enumerate(entries):
        for k in ("file", "md5", "edits"):
            if k not in ent:
                raise DriftError(
                    "наложение %s: в записи #%d нет ключа '%s'" % (path, i, k)
                )
        for j, e in enumerate(ent["edits"]):
            for k in ("anchor", "replace"):
                if k not in e:
                    raise DriftError(
                        "наложение %s: %s правка #%d -- нет ключа '%s'"
                        % (path, ent["file"], j, k)
                    )
            if not e["anchor"]:
                raise DriftError(
                    "наложение %s: %s правка #%d -- ПУСТОЙ якорь; пустой якорь "
                    "адресует всё сразу" % (path, ent["file"], j)
                )
            m = e.get("mode", "replace")
            if m not in ("replace", "insert_before", "insert_after"):
                raise DriftError(
                    "наложение %s: %s правка #%d -- неизвестный mode='%s' "
                    "(replace|insert_before|insert_after)" % (path, ent["file"], j, m)
                )


def load_overlay(path, root=None):
    """Читает наложение любого из двух видов и приводит к одному внутреннему словарю."""
    raw = json.loads(read(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.startswith("overlay_"):
        stem = stem[len("overlay_") :]
    elif stem.endswith("_overlay"):
        stem = stem[: -len("_overlay")]

    if isinstance(raw, list):
        ov = {"name": stem, "version": "1", "files": raw}
    elif isinstance(raw, dict) and "files" in raw:
        ov = dict(raw)
        ov.setdefault("name", stem)
        ov.setdefault("version", "1")
    else:
        raise DriftError(
            "наложение %s: ожидался СПИСОК записей по файлам либо СЛОВАРЬ с ключом 'files'.\n"
            "  Получен %s с ключами %s."
            % (
                path,
                type(raw).__name__,
                sorted(raw.keys()) if isinstance(raw, dict) else "-",
            )
        )
    _check_entries(ov["files"], path)

    ov["prod_root"] = os.path.abspath(root or ov.get("prod_root") or DEFAULT_ROOT)
    ov["_path"] = os.path.abspath(path)
    ov["_raw_is_list"] = isinstance(raw, list)
    if "build" not in ov:
        # сборка может лежать рядом с наложением отдельным файлом -- так список-наложение
        # остаётся минимальным, а сборка всё равно едет вместе с ним
        side = os.path.join(
            os.path.dirname(ov["_path"]),
            os.path.splitext(os.path.basename(path))[0] + ".build.json",
        )
        if os.path.exists(side):
            ov["build"] = json.loads(read(side))
            ov["_build_from"] = side
    return ov


def save_overlay(ov, path):
    payload = (
        ov["files"]
        if ov.get("_raw_is_list")
        else {k: v for k, v in ov.items() if not k.startswith("_")}
    )
    write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


# ======================================================================================
# ВОРОТА ДРЕЙФА
# ======================================================================================
def gate_file(root, ent, overlay_path):
    """Ворота дрейфа для одной записи. Возвращает (текст, [(нач, кон, замена, номер)])."""
    rel = ent["file"]
    src = os.path.join(root, rel)
    if not os.path.exists(src):
        raise DriftError(
            "ВОРОТА ДРЕЙФА (файл): %s\n"
            "  наложение %s ссылается на файл, которого в боевом дереве НЕТ.\n"
            "  Двойник НЕ ПОРОЖДЁН." % (src, overlay_path)
        )

    have = md5_of(src)
    want = ent["md5"]
    if want and have != want:
        raise DriftError(
            "ВОРОТА ДРЕЙФА (md5): %s\n"
            "  записан при порождении наложения : %s\n"
            "  в боевом дереве СЕЙЧАС           : %s\n"
            "  Боевой исходник изменился. Разметку НАДО СВЕРИТЬ с новым текстом руками; молча\n"
            "  собранный двойник дал бы правдоподобные доли ДРУГОГО ядра.\n"
            "  После сверки: twin.py freeze --overlay %s --root %s --yes\n"
            "  Двойник НЕ ПОРОЖДЁН." % (src, want, have, overlay_path, root)
        )

    text = read(src)
    spans = []
    for j, e in enumerate(ent["edits"]):
        anchor = e["anchor"]
        n = text.count(anchor)
        name = e.get("id") or e.get("name") or ("#%d" % j)
        if n == 0:
            if not e.get("required", True):
                continue
            raise DriftError(
                "ВОРОТА ДРЕЙФА (якорь не найден): %s, правка %s\n"
                "  якорь: %s\n"
                "  md5 файла %s, но текста якоря в нём НЕТ.\n"
                "  Правка молча НЕ ПРИМЕНИЛАСЬ БЫ -- двойник оказался бы боевым ядром под\n"
                "  именем двойника, и снятая «фаза» ничего бы не снимала.\n"
                "  Двойник НЕ ПОРОЖДЁН."
                % (
                    src,
                    name,
                    _short(anchor),
                    "совпал" if want and have == want else "НЕ ЗАКРЕПЛЁН",
                )
            )
        if n > 1:
            hits = []
            p = text.find(anchor)
            while p != -1:
                hits.append(_lineno(text, p))
                p = text.find(anchor, p + 1)
            raise DriftError(
                "ВОРОТА ДРЕЙФА (якорь неоднозначен): %s, правка %s\n"
                "  якорь встречается %d раз(а), строки %s: %s\n"
                "  Правка уехала бы в первое вхождение -- молча и, скорее всего, не туда.\n"
                "  Расширьте якорь контекстом.\n"
                "  Двойник НЕ ПОРОЖДЁН."
                % (src, name, n, ", ".join(map(str, hits)), _short(anchor))
            )
        beg = text.index(anchor)
        end = beg + len(anchor)
        mode = e.get("mode", "replace")
        if mode == "replace":
            spans.append((beg, end, e["replace"], j))
        elif mode == "insert_before":
            spans.append((beg, beg, e["replace"], j))
        else:  # insert_after
            spans.append((end, end, e["replace"], j))

    spans.sort()
    for a, b in zip(spans, spans[1:]):
        if a[1] > b[0]:
            raise DriftError(
                "ВОРОТА ДРЕЙФА (якоря перекрываются): %s\n"
                "  правка #%d [%d,%d) и правка #%d [%d,%d) метят в один и тот же текст.\n"
                "  Двойник НЕ ПОРОЖДЁН." % (src, a[3], a[0], a[1], b[3], b[0], b[1])
            )
    return text, spans


def apply_spans(text, spans):
    out, pos = [], 0
    for beg, end, rep, _ in spans:
        out.append(text[pos:beg])
        out.append(rep)
        pos = end
    out.append(text[pos:])
    return "".join(out)


def gate_all(ov, verbose=True, pin=False):
    """Ворота по всем файлам наложения. Возвращает [(ent, текст, spans)]; бросает DriftError."""
    gated, pinned = [], []
    for ent in ov["files"]:
        if not ent.get("md5") and pin:
            ent["md5"] = md5_of(os.path.join(ov["prod_root"], ent["file"]))
            pinned.append(ent["file"])
        text, spans = gate_file(ov["prod_root"], ent, ov["_path"])
        gated.append((ent, text, spans))
        if verbose:
            print(
                "  ok  %-52s md5 %s, правок %d" % (ent["file"], ent["md5"], len(spans))
            )
    if pinned:
        save_overlay(ov, ov["_path"])
        if verbose:
            print("  ЗАКРЕПЛЁН md5 (первое порождение): %s" % ", ".join(pinned))
    return gated


# ======================================================================================
# порождение
# ======================================================================================
def _emit(ov, gated, out_dir, clean=True, quiet=False):
    """Раскладывает двойника в out_dir и возвращает манифест."""
    if clean and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    manifest = {
        "name": ov["name"],
        "version": ov.get("version", "1"),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "generated_by": os.path.abspath(__file__),
        "overlay": ov["_path"],
        "overlay_md5": md5_of(ov["_path"]),
        "prod_root": ov["prod_root"],
        "note": (
            "ДВОЙНИК ПОРОЖДЁН, А НЕ СКОПИРОВАН. Править файлы в этом каталоге руками "
            "НЕЛЬЗЯ: правки будут стёрты следующим `twin.py gen` и, что хуже, не попадут "
            "в ворота дрейфа. Правьте наложение."
        ),
        "files": [],
        "added": [],
        "twin_include_dirs": [],
        "prod_include_dirs": [],
    }
    incs = []
    for ent, text, spans in gated:
        rel = ent["file"]
        dst = os.path.join(out_dir, rel)
        new = apply_spans(text, spans)
        write(dst, new)
        d = os.path.dirname(os.path.abspath(dst))
        if d not in incs:
            incs.append(d)
        manifest["files"].append(
            {
                "file": rel,
                "src_md5": ent["md5"],
                "twin": os.path.abspath(dst),
                "twin_md5": md5_text(new),
                "edits": len(spans),
                "bytes_src": len(text),
                "bytes_twin": len(new),
                # ЯКОРЯ В МАНИФЕСТЕ -- чтобы по одному манифесту было видно, ЧЕМ этот двойник
                # отличается от боевого, даже если наложение потеряно.
                "anchors": [
                    {
                        "id": ent["edits"][j].get("id")
                        or ent["edits"][j].get("name")
                        or "#%d" % j,
                        "why": ent["edits"][j].get("why", ""),
                        "mode": ent["edits"][j].get("mode", "replace"),
                        "line_in_prod": _lineno(text, beg),
                        "anchor": ent["edits"][j]["anchor"],
                        "anchor_md5": md5_text(ent["edits"][j]["anchor"]),
                    }
                    for (beg, _e, _r, j) in sorted(spans)
                ],
            }
        )
        if not quiet:
            print("  %-52s %d правок  ->  %s" % (rel, len(spans), dst))

    for a in ov.get("add", []):
        dst = os.path.join(out_dir, a["path"])
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copyfile(a["from"], dst)
        manifest["added"].append(
            {"path": a["path"], "from": a["from"], "md5": md5_of(a["from"])}
        )
        d = os.path.dirname(os.path.abspath(dst))
        if d not in incs:
            incs.append(d)
        if not quiet:
            print("  добавлен  %-42s <- %s" % (a["path"], a["from"]))
    for w in ov.get("write", []):
        dst = os.path.join(out_dir, w["path"])
        write(dst, w["text"])
        manifest["added"].append(
            {
                "path": w["path"],
                "from": "<наложение: write>",
                "md5": md5_text(w["text"]),
            }
        )
        d = os.path.dirname(os.path.abspath(dst))
        if d not in incs:
            incs.append(d)
        if not quiet:
            print("  записан   %s  (текст из наложения)" % w["path"])

    manifest["twin_include_dirs"] = incs
    manifest["prod_include_dirs"] = [
        os.path.join(ov["prod_root"], d)
        for d in ov.get("prod_include_dirs", DEFAULT_PROD_INCLUDES)
    ]
    manifest["build"] = ov.get("build", {})
    if ov.get("_build_from"):
        manifest["build_from"] = ov["_build_from"]
    write(
        os.path.join(out_dir, "manifest.json"),
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


# ======================================================================================
# КОМАНДЫ
# ======================================================================================
def cmd_check(args):
    ov = load_overlay(args.overlay, args.root)
    print(
        "ВОРОТА ДРЕЙФА: наложение «%s» v%s\n  %s\n  боевое дерево (ТОЛЬКО ЧТЕНИЕ): %s"
        % (ov["name"], ov.get("version"), ov["_path"], ov["prod_root"])
    )
    gate_all(ov, verbose=True, pin=False)
    print("ВОРОТА ДРЕЙФА ПРОЙДЕНЫ: %d файл(ов)." % len(ov["files"]))
    return 0


def cmd_gen(args):
    ov = load_overlay(args.overlay, args.root)
    out_dir = args.out or os.path.join(args.out_root or DEFAULT_OUT_ROOT, ov["name"])
    print(
        "ПОРОЖДЕНИЕ двойника «%s» v%s -> %s" % (ov["name"], ov.get("version"), out_dir)
    )
    gated = gate_all(ov, verbose=True, pin=True)
    m = _emit(ov, gated, out_dir, clean=not args.no_clean)
    print("ДВОЙНИК ПОРОЖДЁН: %s\n  манифест: %s/manifest.json" % (out_dir, out_dir))
    print(
        "  -I (в этом порядке, ПЕРЕД боевыми): %s"
        % " ".join("-I" + d for d in m["twin_include_dirs"])
    )
    print("  ПРАВИТЬ РУКАМИ НЕЛЬЗЯ -- правьте %s и порождайте заново." % ov["_path"])
    return 0


def cmd_apply(args):
    """Прежнее имя порождения: свой --out, без каталога двойников."""
    args.out_root = None
    args.no_clean = not args.clean
    return cmd_gen(args)


def _manifest(name, out_root=None):
    d = (
        name
        if os.path.isdir(name)
        else os.path.join(out_root or DEFAULT_OUT_ROOT, name)
    )
    mp = os.path.join(d, "manifest.json")
    if not os.path.exists(mp):
        raise DriftError(
            "нет двойника «%s»: не найден %s.\n"
            "  Сперва: twin.py gen --overlay <наложение>" % (name, mp)
        )
    return d, json.loads(read(mp))


def cmd_build(args):
    out_dir, m = _manifest(args.twin, args.out_root)

    # ВОРОТА СТОЯТ И ЗДЕСЬ: боевое дерево могло уехать ПОСЛЕ порождения, и тогда собранный
    # двойник -- чужое ядро. Молча этого не допускаем.
    if os.path.exists(m["overlay"]) and not args.no_gate:
        ov = load_overlay(m["overlay"], m["prod_root"])
        gate_all(ov, verbose=False, pin=False)
    elif not os.path.exists(m["overlay"]):
        print(
            "ВНИМАНИЕ: наложение %s исчезло -- ворота дрейфа проверить нечем."
            % m["overlay"]
        )

    build = dict(m.get("build") or {})
    harness = build.get("harness")
    if harness and not os.path.isabs(harness):
        cand = os.path.join(out_dir, harness)
        harness = cand if os.path.exists(cand) else os.path.abspath(harness)
    steps = build.get("steps") or (DEFAULT_BUILD_STEPS if harness else None)
    if args.cmd:
        steps = [{"name": "--cmd", "cmd": args.cmd}]
    if not steps:
        raise DriftError(
            "у двойника «%s» нет сборки: ни build.steps, ни build.harness в наложении.\n"
            '  Либо допишите в наложение блок "build", либо соберите разово:\n'
            "    twin.py build --twin %s --cmd '<строка сборки с {inc} {mask} {outdir}>'"
            % (m["name"], m["name"])
        )

    outdir = os.path.join(out_dir, "build", "mask%d" % args.mask)
    os.makedirs(outdir, exist_ok=True)
    inc = " ".join("-I %s" % d for d in m["twin_include_dirs"] + m["prod_include_dirs"])
    subst = {
        "twin": out_dir,
        "outdir": outdir,
        "mask": args.mask,
        "inc": inc,
        "nvcc": NVCC,
        "cuobjdump": CUOBJDUMP,
        "extra": args.extra or "",
        "name": m["name"],
        "harness": harness or "",
        "prod": m["prod_root"],
    }

    env = dict(os.environ)
    env.update(DEFAULT_ENV)
    # СВОЙ каталог сборки расширения. Боевой кэш torch-расширения не должен участвовать: иначе
    # соберётся (или переиспользуется) БОЕВАЯ .so, а мы решим, что померили двойника.
    env["FA2SM70_BUILD_DIR"] = os.path.join(out_dir, "_build_ext", "mask%d" % args.mask)
    os.makedirs(env["FA2SM70_BUILD_DIR"], exist_ok=True)
    for k, v in (build.get("env") or {}).items():
        env[k] = str(v).format(**subst)

    print(
        "СБОРКА двойника «%s», маска %d\n  выход: %s\n  FA2SM70_BUILD_DIR=%s"
        % (m["name"], args.mask, outdir, env["FA2SM70_BUILD_DIR"])
    )
    for st in steps:
        cmd = st["cmd"].format(**subst)
        print("\n  [%s] %s" % (st.get("name", "шаг"), cmd))
        r = subprocess.run(
            cmd,
            shell=True,
            env=env,
            cwd=out_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if r.stdout:
            tail = r.stdout.splitlines()[-args.tail :]
            print("  | " + "\n  | ".join(tail))
        if r.returncode != 0:
            print("\n  ШАГ УПАЛ (код %d) -- сборка прервана" % r.returncode)
            return r.returncode
    write(
        os.path.join(outdir, "build_info.json"),
        json.dumps(
            {
                "twin": m["name"],
                "mask": args.mask,
                "at": datetime.datetime.now().isoformat(timespec="seconds"),
                "inc": inc,
                "extra": args.extra,
                "src_md5": {f["file"]: f["src_md5"] for f in m["files"]},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    print("\nготово: %s" % outdir)
    return 0


def cmd_diff(args):
    out_dir, m = _manifest(args.twin, args.out_root)
    rc = 0
    for f in m["files"]:
        rel = f["file"]
        if args.file and args.file not in rel:
            continue
        src = os.path.join(m["prod_root"], rel)
        dst = f["twin"]
        cur = md5_of(src)
        drift = cur != f["src_md5"]
        rc = 2 if drift else rc
        print(
            "=== %s%s"
            % (rel, "   <<< БОЕВОЙ ФАЙЛ УЕХАЛ ПОСЛЕ ПОРОЖДЕНИЯ" if drift else "")
        )
        print(
            "    боевой при порождении md5=%s\n    боевой СЕЙЧАС         md5=%s"
            % (f["src_md5"], cur)
        )
        a, b = read(src).splitlines(keepends=True), read(dst).splitlines(keepends=True)
        d = list(
            difflib.unified_diff(
                a, b, "БОЕВОЙ/" + rel, "ДВОЙНИК/" + rel, n=args.context
            )
        )
        if args.stat:
            print(
                "    кусков %d, +%d / -%d строк, правок в наложении %d"
                % (
                    sum(1 for x in d if x.startswith("@@")),
                    sum(1 for x in d if x.startswith("+") and not x.startswith("+++")),
                    sum(1 for x in d if x.startswith("-") and not x.startswith("---")),
                    f["edits"],
                )
            )
            for an in f.get("anchors", []):
                print(
                    "      %-24s стр.%-6d %s"
                    % (an["id"], an["line_in_prod"], an["why"])
                )
        else:
            sys.stdout.writelines(d)
    return rc


def cmd_list(args):
    root = args.out_root or DEFAULT_OUT_ROOT
    rc = 0
    found = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if "manifest.json" not in filenames:
            continue
        dirnames[:] = []
        m = json.loads(read(os.path.join(dirpath, "manifest.json")))
        if "files" not in m or "prod_root" not in m:
            continue
        found += 1
        stale = [
            f["file"]
            for f in m["files"]
            if not os.path.exists(os.path.join(m["prod_root"], f["file"]))
            or md5_of(os.path.join(m["prod_root"], f["file"])) != f["src_md5"]
        ]
        rc = 2 if stale else rc
        print(
            "%-22s v%-3s %s  файлов %-2d  %s"
            % (
                m.get("name", "?"),
                m.get("version", "?"),
                m.get("generated_at", "?"),
                len(m["files"]),
                "ПРОТУХ: " + ", ".join(stale) if stale else "свеж",
            )
        )
        print("    %s" % dirpath)
    if not found:
        print("двойников нет в %s" % root)
    return rc


def cmd_freeze(args):
    if not args.yes:
        print(
            "ОТКАЗ: freeze переписывает md5 в наложении. Это можно делать ТОЛЬКО после того,\n"
            "как человек сверил разметку с НОВЫМ боевым текстом. Подтвердите: --yes",
            file=sys.stderr,
        )
        return 2
    ov = load_overlay(args.overlay, args.root)
    changed = []
    for ent in ov["files"]:
        src = os.path.join(ov["prod_root"], ent["file"])
        text = read(src)
        for j, e in enumerate(ent["edits"]):
            n = text.count(e["anchor"])
            if n != 1 and e.get("required", True):
                print(
                    "ОТКАЗ: %s правка #%d -- якорь встречается %d раз(а). Пока якоря не\n"
                    "  однозначны, md5 переписывать НЕЛЬЗЯ: это узаконило бы неверный адрес.\n"
                    "  якорь: %s" % (src, j, n, _short(e["anchor"])),
                    file=sys.stderr,
                )
                return 2
        have = md5_of(src)
        if have != ent["md5"]:
            changed.append((ent["file"], ent["md5"], have))
            ent["md5"] = have
    save_overlay(ov, ov["_path"])
    for rel, old, new in changed:
        print("  md5 обновлён: %s  %s -> %s" % (rel, old, new))
    print(
        "Обновлено записей: %d. ВСЕ ЗАМЕРЫ, снятые прежним двойником, теперь относятся к\n"
        "ДРУГОМУ боевому тексту -- их надо перемерить." % len(changed)
    )
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="порождение размеченного двойника ядра из наложения + ворота дрейфа",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_root(p):
        p.add_argument(
            "--root",
            default=None,
            help="корень БОЕВОГО дерева (только чтение); умолчание %s" % DEFAULT_ROOT,
        )

    p = sub.add_parser(
        "check", help="ворота дрейфа без порождения (код != 0 при расхождении)"
    )
    p.add_argument("--overlay", required=True)
    add_root(p)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("gen", help="ворота + порождение двойника")
    p.add_argument("--overlay", required=True)
    add_root(p)
    p.add_argument("--out-root", default=None, help="умолчание %s" % DEFAULT_OUT_ROOT)
    p.add_argument(
        "--out", default=None, help="явный каталог двойника (вместо out-root/имя)"
    )
    p.add_argument(
        "--no-clean", action="store_true", help="не вычищать каталог двойника"
    )
    p.set_defaults(fn=cmd_gen)

    p = sub.add_parser("apply", help="прежнее имя gen: порождение в явный --out")
    p.add_argument("--overlay", required=True)
    add_root(p)
    p.add_argument("--out", required=True)
    p.add_argument("--clean", action="store_true")
    p.set_defaults(fn=cmd_apply)

    p = sub.add_parser(
        "build", help="собрать двойника ОТДЕЛЬНЫМ расширением (свой BUILD_DIR)"
    )
    p.add_argument("--twin", required=True, help="имя двойника или путь к его каталогу")
    p.add_argument("--mask", type=int, default=0)
    p.add_argument("--out-root", default=None)
    p.add_argument("--extra", default="", help="доп. флаги в {extra}")
    p.add_argument(
        "--cmd", default=None, help="разовая строка сборки вместо build.steps"
    )
    p.add_argument("--tail", type=int, default=12)
    p.add_argument(
        "--no-gate",
        action="store_true",
        help="собрать, даже если боевое дерево уехало (меряется ЧУЖОЕ ядро)",
    )
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser(
        "diff", help="чем двойник отличается от боевого исходника СЕЙЧАС"
    )
    p.add_argument("--twin", required=True)
    p.add_argument("--file", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--context", type=int, default=3)
    p.add_argument("--stat", action="store_true")
    p.set_defaults(fn=cmd_diff)

    p = sub.add_parser("list", help="что порождено и не протухло ли")
    p.add_argument("--out-root", default=None)
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("freeze", help="ПЕРЕзакрепить md5 после РУЧНОЙ сверки разметки")
    p.add_argument("--overlay", required=True)
    add_root(p)
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_freeze)

    args = ap.parse_args()
    try:
        return args.fn(args)
    except DriftError as ex:
        print("\n" + str(ex) + "\n", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
