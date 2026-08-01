"""Directed-graph primitives for scheduling: topological order, DAG longest path,
minimum/maximum mean cycle (exact rational), Dijkstra, disjoint-set union (pure stdlib).

Общая конвенция этого модуля (отличается от исходников!):
    вершины 0-индексные (0 .. n-1), рёбра — список пар (u, v).
Исходные решения были 1-индексными (benders-play, kingdom-connectivity) либо
индексировались значениями до 10^6 (favourite-sequence); всё приведено к 0-индексу.

ИСТОЧНИК (харвест из монолитных решений, ТОЛЬКО ЧТЕНИЕ исходных деревьев):

  * topo_order(n, edges)  -- Kahn на CSR
      /opt/conda/hr-mini/solutions/benders-play/benders-play_solution.py, строки 40-115
      (построение CSR: строки 5-38; сам Kahn: строки 39-63).
      ЧТО ИЗМЕНЕНО: убраны stdin/print и вся часть Шпрага-Гранди (mex/grundy,
      строки 64-113) — это логика задачи, не алгоритм; 1-индекс -> 0-индекс;
      возвращается порядок, а не "Bumi"/"Iroh". CSR на array('i') сохранён
      (главная ценность оригинала: одна плоская таблица вместо списка списков),
      но с падением на array('l') при n > 2^31 не заигрываем — тип 'i' достаточен.

  * topo_order_lexmin(n, edges)  -- Kahn на куче
      /opt/conda/hr-mini/solutions/favourite-sequence/favourite-sequence_solution.py,
      строки 62-79 (сам Kahn на min-heap), строки 26-60 (дедупликация рёбер через set).
      ЧТО ИЗМЕНЕНО: убраны stdin/print; вместо массивов размера MAX_VAL = 10^6 и
      множества present_nodes — обычные n-размерные списки; дедупликация рёбер
      сохранена (она нужна для корректности in_degree при кратных рёбрах).

  * has_cycle(n, edges), longest_path_dag(n, edges, weight)
      /opt/conda/hr-mini/solutions/kingdom-connectivity/kingdom-connectivity_solution.py,
      строки 66-98 (детекция цикла по счётчику обработанных вершин Kahn'а) и
      строки 100-124 (DP по топологическому порядку).
      ЧТО ИЗМЕНЕНО: убраны stdin/print, 1-индекс -> 0-индекс; отброшена вся обвязка
      достижимости 1->n и "критических вершин" (строки 29-65) — это логика задачи;
      DP смены: было СЧЁТ путей по модулю 10^9 (dp[v] += dp[u]) — стало ДЛИННЕЙШИЙ
      путь (dist[v] = max(dist[v], dist[u] + w)), та же схема "релаксация в
      топологическом порядке", ради чего примитив и брали.

  * min_mean_cycle / max_mean_cycle  -- НЕ БЫЛО В ЗАДАНИИ, найдено при харвесте
      /opt/conda/hr-mini/solutions/hacker-country/hacker-country_solution.py,
      строки 31-108 (формула Карпа по таблице W[k][v] + сравнение дробей
      перекрёстным умножением, БЕЗ float -- точная рациональщина).
      Это ровно тот объект, который задаёт пропускную способность циклического
      расписания (период = максимальный средний цикл), поэтому взят несмотря на
      то, что опись его не нашла.
      ЧТО ИЗМЕНЕНО: убраны stdin/print; матрица смежности N x N (плотная, с
      INF на диагонали ради "цикл длины >= 2" -- это условие ЗАДАЧИ, не алгоритма)
      заменена на список рёбер, так что кратные рёбра и петли поддерживаются
      честно; ручное сравнение num1*den2 > num2*den1 заменено на fractions.Fraction
      (тот же точный ответ, но не надо следить за знаком знаменателя); добавлено
      восстановление САМОГО критического контура (через таблицу предков), которого
      в оригинале не было; добавлен max_mean_cycle = -min_mean_cycle(-w).

  * dijkstra(n, adj, src)
      /opt/conda/hr-mini/solutions/going-office/going-office_solution.py, строки 49-65.
      ЧТО ИЗМЕНЕНО: убраны stdin/print и замыкание на внешние N/INF; рёбра были
      тройками (v, w, edge_index) — индекс ребра выброшен, adj[u] = [(v, w), ...];
      INF = 10**18 заменён на float('inf') (веса могут быть Fraction'ами, а
      сравнение точной дроби с inf корректно).

  * class DSU
      /opt/conda/hr-mini/solutions/components-in-graph/components-in-graph_solution.py,
      строки 17-46 (find с полным сжатием пути + union by size).
      ЧТО ИЗМЕНЕНО: замыкания find/union -> методы класса; убрана вся логика задачи
      (min/max размера компоненты, строки 48-61); добавлены connected()/size_of()/
      groups() и счётчик компонент.

Только стандартная библиотека. Ни stdin, ни input(), ни вызовов на импорте.
"""

from array import array
from collections import deque
from fractions import Fraction
from heapq import heappush, heappop

INF = float("inf")


def build_csr(n, edges):
    """CSR adjacency: (head, adj) with neighbours of u in adj[head[u]:head[u+1]].

    Harvested from benders-play (one flat array instead of a list of lists).
    """
    head = [0] * (n + 2)
    for u, _v in edges:
        head[u + 1] += 1
    for i in range(1, n + 1):
        head[i] += head[i - 1]
    adj = array("i", bytes(4 * len(edges)))
    cursor = list(head)
    for u, v in edges:
        adj[cursor[u]] = v
        cursor[u] += 1
    return head, adj


def topo_order(n, edges):
    """Kahn's topological sort on a CSR graph.

    Returns a list of vertices. If the graph has a cycle the returned list is
    SHORTER than n (it contains exactly the acyclic prefix) -- see has_cycle().
    O(n + m).
    """
    head, adj = build_csr(n, edges)
    indeg = [0] * n
    for _u, v in edges:
        indeg[v] += 1

    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for i in range(head[u], head[u + 1]):
            v = adj[i]
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order


def topo_order_lexmin(n, edges):
    """Lexicographically smallest topological order (Kahn driven by a min-heap).

    Parallel edges are de-duplicated first (otherwise in-degrees double-count).
    Returns a list shorter than n iff the graph has a cycle. O((n + m) log n).
    """
    seen = set()
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for u, v in edges:
        if (u, v) in seen:
            continue
        seen.add((u, v))
        adj[u].append(v)
        indeg[v] += 1

    pq = [i for i in range(n) if indeg[i] == 0]
    pq.sort()
    order = []
    while pq:
        u = heappop(pq)
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heappush(pq, v)
    return order


def has_cycle(n, edges):
    """True iff the directed graph contains a cycle (Kahn's processed-count test)."""
    return len(topo_order(n, edges)) < n


def longest_path_dag(n, edges, weight=None, sources=None):
    """Longest (critical) path in a DAG -- the scheduling workhorse.

    weight: list/tuple parallel to `edges`, or a callable f(u, v), or None (unit).
    sources: vertices allowed to start a path (default: all, each with value 0).

    Returns (dist, parent):
        dist[v]   = max total weight of a path ending at v (or -inf if unreachable
                    when `sources` is restricted),
        parent[v] = predecessor on such a path, or -1.
    Raises ValueError if the graph has a cycle. O(n + m).
    """
    order = topo_order(n, edges)
    if len(order) < n:
        raise ValueError("longest_path_dag: graph has a cycle")

    if weight is None:
        wf = lambda _u, _v, _i: 1
    elif callable(weight):
        wf = lambda u, v, _i: weight(u, v)
    else:
        wf = lambda _u, _v, i: weight[i]

    out = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        out[u].append((v, wf(u, v, i)))

    if sources is None:
        dist = [0] * n
    else:
        dist = [-INF] * n
        for s in sources:
            dist[s] = 0
    parent = [-1] * n

    for u in order:
        du = dist[u]
        if du == -INF:
            continue
        for v, w in out[u]:
            if du + w > dist[v]:
                dist[v] = du + w
                parent[v] = u
    return dist, parent


def min_mean_cycle(n, edges, weight=None):
    """Minimum cycle mean of a digraph, EXACTLY (Karp's theorem). O(n*m) time, O(n^2) memory.

    weight: list/tuple parallel to `edges`, or a callable f(u, v), or None (unit).
    Returns (Fraction lambda, cycle) where `cycle` is the list of vertices of an
    attaining circuit (v0, v1, ..., vk-1 with an edge vk-1 -> v0), or (None, None)
    if the graph is acyclic.

    Karp: lambda = min_v max_{0<=k<n} (D[n][v] - D[k][v]) / (n - k), where D[k][v] is
    the least weight of a walk of EXACTLY k edges ending at v, with D[0][v] = 0 for
    every v (equivalent to a virtual source feeding all vertices at zero cost, so the
    graph need not be strongly connected).

    This is the throughput object for cyclic scheduling: with edge weights
    (latency - period * distance) the binding circuit is the one attaining the
    extremum; see max_mean_cycle for the "period = max mean cycle" direction.
    """
    if n == 0 or not edges:
        return None, None

    if weight is None:
        wf = lambda _u, _v, _i: 1
    elif callable(weight):
        wf = lambda u, v, _i: weight(u, v)
    else:
        wf = lambda _u, _v, i: weight[i]

    out = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        out[u].append((v, wf(u, v, i)))

    # D[k][v] and the predecessor that realises it
    D = [[INF] * n for _ in range(n + 1)]
    par = [[-1] * n for _ in range(n + 1)]
    D[0] = [0] * n
    for k in range(1, n + 1):
        cur = D[k]
        pcur = par[k]
        prev = D[k - 1]
        for u in range(n):
            du = prev[u]
            if du == INF:
                continue
            for v, w in out[u]:
                cand = du + w
                if cand < cur[v]:
                    cur[v] = cand
                    pcur[v] = u

    best = None
    best_v = -1
    for v in range(n):
        if D[n][v] == INF:
            continue
        worst = None  # max over k of (D[n][v]-D[k][v])/(n-k)
        for k in range(n):
            if D[k][v] == INF:
                continue
            val = Fraction(D[n][v] - D[k][v], n - k)
            if worst is None or val > worst:
                worst = val
        if worst is not None and (best is None or worst < best):
            best = worst
            best_v = v
    if best is None:
        return None, None

    # Recover a circuit: walk n predecessor steps back from (n, best_v); the walk
    # must repeat a vertex, and one of the cycles it contains attains lambda.
    walk = [best_v]
    v = best_v
    for k in range(n, 0, -1):
        v = par[k][v]
        if v < 0:
            break
        walk.append(v)
    walk.reverse()  # walk[i] -> walk[i+1] are real edges

    wmap = {}
    for i, (a, b) in enumerate(edges):
        w = wf(a, b, i)
        if (a, b) not in wmap or w < wmap[(a, b)]:
            wmap[(a, b)] = w

    seen = {}
    cycles = []
    for i, x in enumerate(walk):
        if x in seen:
            cyc = walk[seen[x] : i]
            tot = sum(wmap[(cyc[j], cyc[(j + 1) % len(cyc)])] for j in range(len(cyc)))
            cycles.append((Fraction(tot, len(cyc)), cyc))
            seen = {}  # keep scanning for further disjoint cycles
        seen[x] = i
    cycles = [c for c in cycles if c[0] == best]
    return best, (cycles[0][1] if cycles else None)


def max_mean_cycle(n, edges, weight=None):
    """Maximum cycle mean, exactly. Same contract as min_mean_cycle.

    In cyclic scheduling this IS the minimum feasible period (initiation interval):
    with weight = latency and each edge carrying `distance` iterations, the binding
    circuit is max over circuits of (sum latency) / (sum distance); for the plain
    unit-distance case that is exactly this function.
    """
    if weight is None:
        neg = lambda _u, _v: -1
    elif callable(weight):
        neg = lambda u, v: -weight(u, v)
    else:
        neg = [-w for w in weight]
    lam, cyc = min_mean_cycle(n, edges, neg)
    return (None, None) if lam is None else (-lam, cyc)


def dijkstra(n, adj, src):
    """Single-source shortest paths, non-negative weights. adj[u] = [(v, w), ...].

    Returns (dist, parent); unreachable vertices have dist = inf, parent = -1.
    O((n + m) log n). Weights may be ints or Fractions (exact arithmetic kept).
    """
    dist = [INF] * n
    parent = [-1] * n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heappush(pq, (nd, v))
    return dist, parent


class DSU:
    """Disjoint-set union: full path compression + union by size."""

    __slots__ = ("parent", "size", "count")

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.count = n  # number of disjoint components

    def find(self, x):
        parent = self.parent
        root = x
        while root != parent[root]:
            root = parent[root]
        while x != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(self, x, y):
        """Merge the components of x and y. Returns True if they were separate."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        self.count -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def size_of(self, x):
        return self.size[self.find(x)]

    def groups(self):
        """dict root -> sorted list of members."""
        g = {}
        for i in range(len(self.parent)):
            g.setdefault(self.find(i), []).append(i)
        return g


# ------------------------------------------------------------ self-check -----
def _selfcheck():
    import random

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

    print("topo_order")
    # chain 0->1->2->3->4 has exactly ONE topological order
    chain = [(i, i + 1) for i in range(4)]
    check("chain of 5", topo_order(5, chain), [0, 1, 2, 3, 4])
    check("isolated vertices", topo_order(3, []), [0, 1, 2])
    # diamond 0->{1,2}->3: order must start at 0, end at 3
    dia = [(0, 1), (0, 2), (1, 3), (2, 3)]
    o = topo_order(4, dia)
    check("diamond length 4", len(o), 4)
    check("diamond first/last", (o[0], o[-1]), (0, 3))
    check("cycle truncates order", len(topo_order(3, [(0, 1), (1, 2), (2, 0)])), 0)

    print("topo_order_lexmin")
    # 0->2, 1->2: lexicographically smallest is [0, 1, 2]
    check("two sources", topo_order_lexmin(3, [(0, 2), (1, 2)]), [0, 1, 2])
    # 3->0, 1, 2 free: lexmin must place 3 before 0 but 1,2 as early as possible
    check("forced pair", topo_order_lexmin(4, [(3, 0)]), [1, 2, 3, 0])
    check("chain of 5", topo_order_lexmin(5, chain), [0, 1, 2, 3, 4])
    check(
        "duplicate edges tolerated",
        topo_order_lexmin(3, [(0, 1), (0, 1), (1, 2)]),
        [0, 1, 2],
    )

    print("has_cycle")
    check("chain acyclic", has_cycle(5, chain), False)
    check("triangle cyclic", has_cycle(3, [(0, 1), (1, 2), (2, 0)]), True)
    check("self-loop cyclic", has_cycle(2, [(0, 0)]), True)
    check("diamond acyclic", has_cycle(4, dia), False)

    print("longest_path_dag")
    # 0->1 (3), 0->2 (1), 1->3 (1), 2->3 (10): longest to 3 is 0->2->3 = 11
    E = [(0, 1), (0, 2), (1, 3), (2, 3)]
    W = [3, 1, 1, 10]
    dist, par = longest_path_dag(4, E, W)
    check("critical path value 11", dist[3], 11)
    check("critical path predecessor", par[3], 2)
    check("unit weights = 3 hops on a chain", longest_path_dag(5, chain)[0][4], 4)
    check("callable weight", longest_path_dag(4, E, lambda u, v: 1)[0][3], 2)
    try:
        longest_path_dag(3, [(0, 1), (1, 2), (2, 0)])
        check("cycle raises", False, True)
    except ValueError:
        check("cycle raises", True, True)

    print("min_mean_cycle / max_mean_cycle (exact rationals)")
    # triangle 0->1 (1), 1->2 (2), 2->0 (3): the only cycle, mean = 6/3 = 2
    tri = [(0, 1), (1, 2), (2, 0)]
    lam, cyc = min_mean_cycle(3, tri, [1, 2, 3])
    check("triangle mean = 2", lam, Fraction(2))
    check("triangle circuit has 3 vertices", sorted(cyc), [0, 1, 2])
    # add a cheap 2-cycle 0<->3 of mean 1: min = 1, max stays 2
    E2 = tri + [(0, 3), (3, 0)]
    W2 = [1, 2, 3, 1, 1]
    check("min over two cycles = 1", min_mean_cycle(4, E2, W2)[0], Fraction(1))
    check("max over two cycles = 2", max_mean_cycle(4, E2, W2)[0], Fraction(2))
    check("min circuit is the 2-cycle", sorted(min_mean_cycle(4, E2, W2)[1]), [0, 3])
    # non-integer mean: cycle 0->1->2->0 of total 5 over 3 edges = 5/3
    check("mean 5/3 exact", min_mean_cycle(3, tri, [1, 1, 3])[0], Fraction(5, 3))
    check("acyclic -> None", min_mean_cycle(3, [(0, 1), (1, 2)], [1, 1])[0], None)
    check(
        "self-loop mean = its weight", min_mean_cycle(2, [(0, 0)], [7])[0], Fraction(7)
    )
    check(
        "negative mean allowed",
        min_mean_cycle(2, [(0, 1), (1, 0)], [-3, 1])[0],
        Fraction(-1),
    )
    # unit weights: min mean cycle = 1 whatever the circuit
    check("unit weights", min_mean_cycle(4, E2)[0], Fraction(1))

    print("dijkstra")
    # 0-1 (4), 0-2 (1), 2-1 (2), 1-3 (1), 2-3 (5): dist to 3 = 0->2->1->3 = 4
    adj = [[(1, 4), (2, 1)], [(3, 1)], [(1, 2), (3, 5)], []]
    dist, par = dijkstra(4, adj, 0)
    check("dist = [0,3,1,4]", dist, [0, 3, 1, 4])
    check("parent of 3 is 1", par[3], 1)
    check("unreachable stays inf", dijkstra(3, [[(1, 2)], [], []], 0)[0][2], INF)
    fadj = [[(1, Fraction(1, 3))], [(2, Fraction(1, 6))], []]
    check("exact rational distance = 1/2", dijkstra(3, fadj, 0)[0][2], Fraction(1, 2))

    print("DSU")
    d = DSU(6)
    d.union(0, 1)
    d.union(1, 2)
    d.union(4, 5)
    check("components = 3", d.count, 3)
    check("0 ~ 2", d.connected(0, 2), True)
    check("0 !~ 3", d.connected(0, 3), False)
    check("size of 0's group = 3", d.size_of(0), 3)
    check("union of merged returns False", d.union(0, 2), False)
    check(
        "group partition",
        sorted(sorted(v) for v in d.groups().values()),
        [[0, 1, 2], [3], [4, 5]],
    )

    print("differential: topo_order / has_cycle vs brute DFS colouring")
    rng = random.Random(99)
    bad_cycle = bad_topo = bad_lex = 0
    for _ in range(300):
        n = rng.randint(1, 8)
        edges = [
            (rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(0, 12))
        ]

        # brute cycle detection (recursive 3-colour DFS)
        out = [[] for _ in range(n)]
        for u, v in edges:
            out[u].append(v)
        colour = [0] * n

        def dfs(u):
            colour[u] = 1
            for v in out[u]:
                if colour[v] == 1 or (colour[v] == 0 and dfs(v)):
                    return True
            colour[u] = 2
            return False

        brute_cyclic = any(colour[i] == 0 and dfs(i) for i in range(n))
        if has_cycle(n, edges) != brute_cyclic:
            bad_cycle += 1
        if not brute_cyclic:
            o = topo_order(n, edges)
            pos = {v: i for i, v in enumerate(o)}
            if len(o) != n or any(pos[u] >= pos[v] for u, v in edges):
                bad_topo += 1
            ol = topo_order_lexmin(n, edges)
            posl = {v: i for i, v in enumerate(ol)}
            if len(ol) != n or any(posl[u] >= posl[v] for u, v in edges):
                bad_lex += 1
            else:
                # brute-force lexicographically smallest order over all permutations
                from itertools import permutations

                if n <= 7:
                    best = min(
                        (
                            p
                            for p in permutations(range(n))
                            if all(p.index(u) < p.index(v) for u, v in edges)
                        ),
                        default=None,
                    )
                    if best is not None and list(best) != ol:
                        bad_lex += 1
    check("cycle detection matches brute", bad_cycle, 0)
    check("topo order valid", bad_topo, 0)
    check("lexmin order is truly minimal", bad_lex, 0)

    print("differential: dijkstra vs Bellman-Ford on random graphs")
    bad = 0
    for _ in range(200):
        n = rng.randint(2, 9)
        adj = [[] for _ in range(n)]
        E = []
        for _ in range(rng.randint(1, 3 * n)):
            u, v, w = rng.randrange(n), rng.randrange(n), rng.randint(0, 20)
            adj[u].append((v, w))
            E.append((u, v, w))
        dd = dijkstra(n, adj, 0)[0]
        bf = [INF] * n
        bf[0] = 0
        for _ in range(n):
            for u, v, w in E:
                if bf[u] + w < bf[v]:
                    bf[v] = bf[u] + w
        if dd != bf:
            bad += 1
    check("200 random graphs agree", bad, 0)

    print("differential: longest_path_dag vs brute enumeration of DAG paths")
    bad = 0
    for _ in range(200):
        n = rng.randint(2, 7)
        edges, W = [], []
        for u in range(n):
            for v in range(u + 1, n):  # random DAG by construction
                if rng.random() < 0.4:
                    edges.append((u, v))
                    W.append(rng.randint(1, 9))
        dist, _p = longest_path_dag(n, edges, W)
        brute = [0] * n
        for u in range(n):  # vertices already topologically sorted
            for i, (a, b) in enumerate(edges):
                if b == u and brute[a] + W[i] > brute[u]:
                    brute[u] = brute[a] + W[i]
        if dist != brute:
            bad += 1
    check("200 random DAGs agree", bad, 0)

    print("differential: mean cycle vs brute enumeration of ALL simple cycles")
    bad_lam = bad_cyc = 0
    for _ in range(200):
        n = rng.randint(2, 6)
        edges, W = [], []
        for u in range(n):
            for v in range(n):
                if rng.random() < 0.35:
                    edges.append((u, v))
                    W.append(rng.randint(-5, 9))
        if not edges:
            continue
        out = {}
        for i, (u, v) in enumerate(edges):
            if (u, v) not in out or W[i] < out[(u, v)]:
                out[(u, v)] = W[i]

        # brute force: every simple cycle, by DFS from its smallest vertex
        means = []

        def walk(start, u, path, total):
            for v in range(n):
                if (u, v) not in out or v < start:
                    continue
                if v == start:
                    means.append(Fraction(total + out[(u, v)], len(path)))
                elif v not in path:
                    path.add(v)
                    walk(start, v, path, total + out[(u, v)])
                    path.discard(v)

        for s in range(n):
            walk(s, s, {s}, 0)

        lam, cyc = min_mean_cycle(n, edges, W)
        lamx, _cx = max_mean_cycle(n, edges, W)
        if not means:
            if lam is not None or lamx is not None:
                bad_lam += 1
            continue
        if lam != min(means) or lamx != max(means):
            bad_lam += 1
            continue
        # the returned circuit must exist and attain lambda
        if cyc is None or len(cyc) != len(set(cyc)):
            bad_cyc += 1
            continue
        tot = sum(out[(cyc[j], cyc[(j + 1) % len(cyc)])] for j in range(len(cyc)))
        if Fraction(tot, len(cyc)) != lam:
            bad_cyc += 1
    check("200 random graphs: lambda matches brute", bad_lam, 0)
    check("200 random graphs: circuit attains lambda", bad_cyc, 0)

    print("differential: DSU vs brute connectivity")
    bad = 0
    for _ in range(200):
        n = rng.randint(1, 10)
        pairs = [
            (rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(0, 12))
        ]
        d = DSU(n)
        for u, v in pairs:
            d.union(u, v)
        lab = list(range(n))  # brute: repeated relabelling
        changed = True
        while changed:
            changed = False
            for u, v in pairs:
                if lab[u] != lab[v]:
                    lo = min(lab[u], lab[v])
                    old = max(lab[u], lab[v])
                    for i in range(n):
                        if lab[i] == old:
                            lab[i] = lo
                    changed = True
        if d.count != len(set(lab)) or any(
            d.connected(i, j) != (lab[i] == lab[j]) for i in range(n) for j in range(n)
        ):
            bad += 1
    check("200 random unions agree", bad, 0)

    print("ALL OK" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_selfcheck())
