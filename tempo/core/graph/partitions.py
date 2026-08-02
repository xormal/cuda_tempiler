# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""Partition lattice of a k-element set: join, refinement order, and the
zeta/Moebius join-convolution (pure stdlib).

ИСТОЧНИК: /opt/conda/hackerrank-harness/solutions/euler194/spherecut/zeta_identity.py,
строки 1-77 (весь файл), ТОЛЬКО ЧТЕНИЕ исходного дерева.

ЧТО ИЗМЕНЕНО при переносе:
  - тела функций canonp / all_partitions / join / refines / naive_joinconv /
    zeta_joinconv скопированы ПОБУКВЕННО, без правок;
  - проверочный прогон (строки 60-76 оригинала: случайные таблицы, сверка
    naive против zeta, печать счёта) выполнялся НА ИМПОРТЕ -- перенесён внутрь
    `if __name__ == "__main__":` и снабжён кодом возврата, потому что модуль
    предназначен для импорта (требование: никаких вызовов на импорте);
  - добавлены docstring'и и `__all__`; сама математика не тронута.

Что здесь есть (все разбиения кодируются restricted-growth строками, т.е.
кортежами меток длины k, где первое вхождение метки идёт по порядку 0,1,2,...):
  all_partitions(k)      -- перечислить все разбиения k-элементного множества (Bell(k));
  join(pa, pb)           -- решёточное соединение (наименьшее огрубление обоих);
  refines(p, q)          -- p <= q, т.е. p мельче q;
  naive_joinconv(T1,T2,k)-- прямая свёртка по join, O(|T1|*|T2|) -- эталон;
  zeta_joinconv(T1,T2,k) -- та же свёртка через дзета-преобразование по решётке,
                            поточечное умножение и обращение Мёбиуса.

Только стандартная библиотека. Ни stdin, ни input(), ни вызовов на импорте.
"""

# Validate: join-convolution of two partition tables == pointwise product in zeta space, then Mobius back.
import itertools, random, collections

__all__ = [
    "canonp",
    "all_partitions",
    "join",
    "refines",
    "naive_joinconv",
    "zeta_joinconv",
]


def canonp(labels):
    """Canonicalise a labelling into a restricted-growth string."""
    r = {}
    o = []
    for x in labels:
        if x not in r:
            r[x] = len(r)
        o.append(r[x])
    return tuple(o)


def all_partitions(k):
    """Yield every set partition of {0..k-1} as a restricted-growth tuple."""
    if k == 0:
        yield ()
        return
    for p in all_partitions(k - 1):
        mx = max(p) + 1 if p else 0
        for lab in range(mx + 1):
            yield p + (lab,)


def join(pa, pb):
    """Lattice join: the coarsest common refinement's dual -- merge blocks of both."""
    n = len(pa)
    par = list(range(n))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    def u(a, b):
        ra, rb = f(a), f(b)
        if ra != rb:
            par[ra] = rb

    for p in (pa, pb):
        g = {}
        for i, l in enumerate(p):
            if l in g:
                u(g[l], i)
            else:
                g[l] = i
    return canonp([f(i) for i in range(n)])


def refines(p, q):  # p <= q : p finer (same block in p => same in q)
    """True iff p is finer than q (every block of p sits inside a block of q)."""
    m = {}
    for a, b in zip(p, q):
        if a in m:
            if m[a] != b:
                return False
        else:
            m[a] = b
    return True


def naive_joinconv(T1, T2, k):
    """Direct join-convolution of two weight tables. Reference implementation."""
    R = collections.defaultdict(int)
    for p1, w1 in T1.items():
        for p2, w2 in T2.items():
            R[join(p1, p2)] += w1 * w2
    return dict(R)


def zeta_joinconv(T1, T2, k):
    """Join-convolution via zeta transform, pointwise product, Mobius inversion."""
    parts = list(all_partitions(k))

    def zeta(T):
        Z = {}
        for s in parts:
            Z[s] = sum(w for p, w in T.items() if refines(p, s))
        return Z

    Z1 = zeta(T1)
    Z2 = zeta(T2)
    Zp = {s: Z1[s] * Z2[s] for s in parts}
    # Mobius invert: g[sig] = sum_{tau <= sig} mu(tau,sig) Zp[tau]; process finest-first
    order = sorted(parts, key=lambda p: len(set(p)), reverse=True)  # finest first
    g = {}
    for sig in order:
        s = Zp[sig]
        for tau in parts:
            if tau == sig:
                continue
            if len(set(tau)) > len(set(sig)) and refines(tau, sig):
                s -= g[tau]
        g[sig] = s
    return {p: v for p, v in g.items() if v}


# ------------------------------------------------------------ self-check -----
def _selfcheck():
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(
            ("  PASS " if good else "  FAIL ")
            + name
            + ": got %r, want %r" % (got, want)
        )

    print("lattice basics (hand-checkable)")
    # Bell numbers B0..B6 = 1, 1, 2, 5, 15, 52, 203
    check(
        "Bell numbers",
        [len(list(all_partitions(k))) for k in range(7)],
        [1, 1, 2, 5, 15, 52, 203],
    )
    check("join of {01|2} and {0|12} = {012}", join((0, 0, 1), (0, 1, 1)), (0, 0, 0))
    check(
        "join with the finest partition is identity",
        join((0, 1, 2), (0, 0, 1)),
        (0, 0, 1),
    )
    check("finest refines everything", refines((0, 1, 2), (0, 0, 0)), True)
    check("coarsest does not refine the finest", refines((0, 0, 0), (0, 1, 2)), False)
    check("canonp relabels", canonp([5, 5, 7]), (0, 0, 1))

    print("zeta join-convolution identity (original validation, lines 60-76)")
    # ORIGINAL BLOCK, verbatim except for the accumulated pass/fail counters.
    bad = 0
    tot = 0
    rng = random.Random(0)
    for k in range(1, 7):
        parts = list(all_partitions(k))
        for trial in range(40):
            T1 = {}
            T2 = {}
            for p in parts:
                if rng.random() < 0.5:
                    T1[p] = rng.randint(-3, 3)
                if rng.random() < 0.5:
                    T2[p] = rng.randint(-3, 3)
            T1 = {p: w for p, w in T1.items() if w}
            T2 = {p: w for p, w in T2.items() if w}
            if not T1 or not T2:
                continue
            a = naive_joinconv(T1, T2, k)
            b = zeta_joinconv(T1, T2, k)
            a = {p: w for p, w in a.items() if w}
            tot += 1
            if a != b:
                bad += 1
            if a != b and bad <= 3:
                print(f"MISMATCH k={k}: naive={a} zeta={b}")
    print(f"  zeta join-convolution identity: {tot - bad}/{tot} correct")
    check("no mismatches", bad, 0)
    # Random(0) is seeded, so the trial count is deterministic: 182 non-empty pairs.
    check("all trials ran and agreed", (tot, tot - bad), (182, 182))

    print("ALL OK" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_selfcheck())
