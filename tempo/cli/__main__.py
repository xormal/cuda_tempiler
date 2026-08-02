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
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
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
        kind="gemm", dtype_a=a.dtype_a, dtype_b=a.dtype_b, dtype_c="fp16", dtype_acc="fp32",
        layout_a="k", layout_b="k", layout_c="n",
        shapes={"M": a.M, "N": a.N, "K": a.K}, tol_rel_l2=1e-3,
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


def cmd_stubs(a):
    from tempo.plugins import registry

    p = registry.load(a.arch)
    print("ЧЕСТНЫЙ ПЕРЕЧЕНЬ НЕРЕАЛИЗОВАННОГО -- %s" % p.id)
    for s in p.declared_stubs():
        print("  * " + s)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tempo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
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

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
