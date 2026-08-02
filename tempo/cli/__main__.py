#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""ТЕМПОЛЯТОР -- командная строка.

python3 -m tempo.cli doctor                     состояние окружения (карту не трогает)
python3 -m tempo.cli plugins                    какие плагины есть и что они умеют
python3 -m tempo.cli selftest [--arch sm70]     самопроверки плагинов
python3 -m tempo.cli gates                      гейты непротекания G1..G8
python3 -m tempo.cli recognize <файл.cu> [M N K]   наивный исходник -> спецификация
python3 -m tempo.cli space --arch sm70 -M .. -N .. -K ..   перечислить и отсечь БЕЗ СБОРКИ
python3 -m tempo.cli emit  --arch sm70 ... --out DIR       породить исходник ядра
python3 -m tempo.cli stubs --arch sm80          ЧЕСТНЫЙ перечень нереализованного
python3 -m tempo.cli laws [--unwired] [--kind ДОЛГ] [--scope M=64] [--md] [--check]
                                                РЕЕСТР ЗАКОНОВ: что открыто, чем доказано,
                                                чем опровергается, ГДЕ ЖИВЁТ в продукте
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def cmd_doctor(a):
    from tempo.cli import env

    rc = env.main([])
    print()
    print("ПЛАГИНЫ:")
    from tempo.plugins import registry

    for name in registry.available():
        try:
            p = registry.load(name)
            print("  %-16s %s" % (name, p.description))
        except Exception as e:
            print("  %-16s ОТКАЗ ЗАГРУЗКИ: %s" % (name, e))
    return rc


def cmd_plugins(a):
    from tempo.plugins import registry

    for name in registry.available():
        p = registry.load(name)
        print("%s  (arch-id %s, контракт %s)" % (name, p.id, p.contract))
        print("  %s" % p.description)
        try:
            caps = sorted(p.skeletons.capabilities())
            print("  умеет: %s" % (", ".join(caps) if caps else "-"))
        except Exception as e:
            print("  умеет: -- (%s)" % type(e).__name__)
        print("  заглушек объявлено: %d" % len(p.declared_stubs()))
        print()
    return 0


def cmd_selftest(a):
    from tempo.plugins import registry

    names = [a.arch] if a.arch else registry.available()
    bad = 0
    for name in names:
        p = registry.load(name)
        r = p.selftest()
        print(r.render())
        print()
        if not r.green:
            bad += 1
    return 1 if bad else 0


def cmd_gates(a):
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_gates

    return test_gates.main()


def cmd_recognize(a):
    from tempo.core.op.recognize import from_file
    from tempo.core.op.spec import describe, flops

    shapes = {}
    if a.M:
        shapes = {"M": a.M, "N": a.N, "K": a.K}
    op = from_file(a.source, shapes)
    print("РАСПОЗНАНО: " + describe(op))
    print("допуск relL2: %g; политика покрытия: %s" % (op.tol_rel_l2, op.coverage))
    try:
        print("работы: %.3g операций" % flops(op))
    except ValueError:
        pass
    return 0


def _op_from_args(a):
    from tempo.plugins.base import OpSpec

    return OpSpec(
        kind="gemm",
        dtype_a=a.dtype_a,
        dtype_b=a.dtype_b,
        dtype_c="fp16",
        dtype_acc="fp32",
        layout_a="k",
        layout_b="k",
        layout_c="n",
        shapes={"M": a.M, "N": a.N, "K": a.K},
        tol_rel_l2=1e-3,
    )


def cmd_space(a):
    from tempo.core.search.prune import prune, summary
    from tempo.core.search.space import axes_table
    from tempo.plugins import registry

    p = registry.load(a.arch)
    op = _op_from_args(a)
    print(axes_table(p))
    print()
    hs = list(p.skeletons.variants(op))
    cands = prune(p, op, hs)
    for c in sorted(cands, key=lambda c: (not c.kept, getattr(c.hyper, "key", ""))):
        print("  " + c.render())
    print()
    print(summary(cands))
    return 0


def cmd_emit(a):
    from tempo.core.emit.render import emit
    from tempo.core.search.prune import kept, prune
    from tempo.plugins import registry

    p = registry.load(a.arch)
    op = _op_from_args(a)
    hs = list(p.skeletons.variants(op))
    good = kept(prune(p, op, hs))
    if not good:
        print("ОТСЕКАТЕЛЬ НЕ ОСТАВИЛ НИ ОДНОГО ВАРИАНТА -- порождать нечего")
        return 2
    h = good[0].hyper
    path = emit(p, op, h, a.out)
    print("порождено: %s (гиперформа %s)" % (path, h.key))
    print("точка входа для доказательства маршрута: %s" % p.skeletons.entry_probe(h))
    return 0


def cmd_laws(a):
    """РЕЕСТР ЗАКОНОВ.  Отвечает на вопрос, на который сегодня не отвечает никто:
    «какие законы вообще действуют В ЭТОЙ ТОЧКЕ» -- из-за его отсутствия заявку дважды подряд
    проверяли не в той точке, где она сделана."""
    from tempo.core import laws as L

    all_laws = L.load(ROOT)
    if a.check:
        bad = []
        for l in all_laws:
            bad += L.check_record(l)
            bad += L.check_homes(l, ROOT)
        bad += L.check_links(all_laws)
        bad += L.check_selftests(all_laws)
        bad += L.check_forbidden(all_laws, ROOT, PRODUCT_DIRS, PRODUCT_SUFFIXES)
        for b in bad:
            print("  ПРЕТЕНЗИЯ  " + b)
        print(
            "ИТОГ ПРОВЕРКИ РЕЕСТРА: %s (записей %d, претензий %d)"
            % ("ПРОЙДЕНО" if not bad else "ЕСТЬ ПРЕТЕНЗИИ", len(all_laws), len(bad))
        )
        return 0 if not bad else 1

    sel = L.select(all_laws, arch=a.arch, kind=a.kind, unwired=a.unwired, scope=a.scope)
    if a.md:
        print(L.render_markdown(sel), end="")
        return 0
    if a.scope:
        print(
            "ЗАКОНЫ, ДЕЙСТВУЮЩИЕ В ТОЧКЕ %s (записей всего %d, в точке %d)"
            % (a.scope, len(all_laws), len(sel))
        )
        print(
            "  ОСЬ, НЕ НАЗВАННАЯ В ОБЛАСТИ, НЕ ОЗНАЧАЕТ «ДЕЙСТВУЕТ»: такие записи ниже помечены."
        )
        for l in sel:
            ok, why = L.acts_at(
                l, a.scope.split("=")[0].strip(), a.scope.split("=", 1)[1].strip()
            )
            print(
                "  %-26s %s  (%s)"
                % (l.id, "действует" if ok else "ОСЬ НЕ НАЗВАНА", why)
            )
        print()
    print(L.render_text(sel), end="")
    debts = [l for l in all_laws if l.kind == "ДОЛГ"]
    print(
        "ИТОГО: записей %d (законов %d, опровержений %d, ДОЛГОВ %d)"
        % (
            len(all_laws),
            sum(1 for l in all_laws if l.kind == "ЗАКОН"),
            sum(1 for l in all_laws if l.kind == "ОПРОВЕРЖЕНИЕ"),
            len(debts),
        )
    )
    if debts and not a.kind:
        print(
            "  ДОЛГ = открыто и в продукт НЕ внесено. Это честная картина, а не провал:"
        )
        print("  " + ", ".join(l.id for l in debts))
    return 0


# Что считается ПРОДУКТОМ при поиске маркеров и запрещённых формулировок.
PRODUCT_DIRS = (
    "tempo",
    "tools",
    "tests",
    "kernels",
    "docs",
    "bench",
    "README.md",
    "CHANGELOG.md",
)
PRODUCT_SUFFIXES = (".py", ".cu", ".cuh", ".cpp", ".h", ".md", ".json", ".inc")


def cmd_stubs(a):
    from tempo.plugins import registry

    p = registry.load(a.arch)
    print("ЧЕСТНЫЙ ПЕРЕЧЕНЬ НЕРЕАЛИЗОВАННОГО -- %s" % p.id)
    for s in p.declared_stubs():
        print("  * " + s)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="tempo",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    sub.add_parser("plugins").set_defaults(fn=cmd_plugins)

    s = sub.add_parser("selftest")
    s.add_argument("--arch")
    s.set_defaults(fn=cmd_selftest)

    sub.add_parser("gates").set_defaults(fn=cmd_gates)

    s = sub.add_parser("recognize")
    s.add_argument("source")
    s.add_argument("M", nargs="?", type=int)
    s.add_argument("N", nargs="?", type=int)
    s.add_argument("K", nargs="?", type=int)
    s.set_defaults(fn=cmd_recognize)

    for name, fn in (("space", cmd_space), ("emit", cmd_emit)):
        s = sub.add_parser(name)
        s.add_argument("--arch", default="sm70")
        s.add_argument("-M", type=int, required=True)
        s.add_argument("-N", type=int, required=True)
        s.add_argument("-K", type=int, required=True)
        s.add_argument("--dtype-a", default="fp16")
        s.add_argument("--dtype-b", default="fp16")
        if name == "emit":
            s.add_argument("--out", required=True)
        s.set_defaults(fn=fn)

    s = sub.add_parser("stubs")
    s.add_argument("--arch", default="sm70")
    s.set_defaults(fn=cmd_stubs)

    s = sub.add_parser("laws")
    s.add_argument(
        "--arch", help="только законы этой машины (по имени каталога плагина)"
    )
    s.add_argument("--kind", choices=("ЗАКОН", "ОПРОВЕРЖЕНИЕ", "ДОЛГ"))
    s.add_argument(
        "--unwired", action="store_true", help="только те, у кого нет места в продукте"
    )
    s.add_argument("--scope", help="какие законы действуют В ТОЧКЕ, например M=64")
    s.add_argument("--md", action="store_true", help="породить docs/LAWS.md")
    s.add_argument(
        "--check", action="store_true", help="проверки реестра (те же, что в G9-G11)"
    )
    s.set_defaults(fn=cmd_laws)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
