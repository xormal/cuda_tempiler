# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""Network-flow and matching primitives (pure stdlib).

ИСТОЧНИК (харвест из монолитных решений, ТОЛЬКО ЧТЕНИЕ исходных деревьев):

  * class Dinic
      /opt/conda/hr-mini/solutions/computer-game/computer-game_solution.py, строки 96-213
      (итеративный Диниц с current-arc внутри solve()).
      ЧТО ИЗМЕНЕНО при переносе:
        - убраны stdin/print/solve(), замыкания и глобальные SRC/SNK/TOTAL_NODES ->
          обычный класс с явными (s, t);
        - ПОЧИНЕНЫ ДВА ОТСТУПЛЕНИЯ ОТ ДИНИЦА, ломающие свойство "фаза = блокирующий
          поток". ЗАМЕРЕНО: величина потока у оригинала ВЕРНА (внешний цикл всё равно
          останавливается только когда в остаточной сети нет s-t пути), но фаз до
          46 вместо 3 при n=8, т.е. O(F) фаз вместо гарантированных O(V):
          (1) `flow` (бутылочное горло) не восстанавливался при backtrack -> путь
              продавливался ЗАНИЖЕННЫМ значением (горлом брошенной ветки). Теперь
              горло считается по фактически собранному пути в момент достижения стока;
          (2) current-arc сдвигался (`ptr[curr] = i + 1`) ДО спуска по ребру, т.е.
              остаток ёмкости этого ребра терялся до конца фазы. Теперь указатель
              сдвигается только когда ребро исчерпано или поддерево мертво, плюс
              отсечение мёртвых вершин (level[u] = -1).
          На единичных ёмкостях (регламент исходной задачи) оба отступления невидимы.
        - добавлены edge_flow()/min_cut() (нужны, чтобы читать САМО назначение, а не
          только величину потока).

  * class MinCostFlow
      /opt/conda/hr-mini/solutions/jumping-rooks/jumping-rooks_solution.py, строки 104-197
      (SSP + SPFA, приём "выпуклая стоимость параллельными рёбрами" из строк 118-125).
      ЧТО ИЗМЕНЕНО:
        - убраны stdin/print и захардкоженные s=0, t=1 -> явные (s, t, maxf);
        - аугментация не по 1 единице за проход, а по бутылочному горлу пути
          (тот же ответ, на порядок меньше итераций SPFA);
        - возвращается пара (flow, cost) вместо печати стоимости.
      ПРИЁМ СОХРАНЁН: выпуклая по объёму стоимость задаётся ПАРАЛЛЕЛЬНЫМИ рёбрами
      единичной ёмкости с неубывающими стоимостями 0,1,2,... — SSP сам берёт их по
      возрастанию, см. convex_edges() и самопроверку.

  * kuhn_matching
      /opt/conda/hr-mini/solutions/bike-racers/bike-racers_solution.py, строки 53-95
      (Kuhn/венгерский поиск дополняющих путей по МАТРИЦЕ расстояний с порогом).
      ЧТО ИЗМЕНЕНО:
        - матрица dist[u][v] <= limit заменена на СПИСКИ СМЕЖНОСТИ adj[u] = [v, ...],
          порог/бинпоиск остались снаружи (это была логика задачи, не алгоритма);
        - рекурсия -> явный стек (нет sys.setrecursionlimit, глубина = nleft);
        - возвращаются оба массива паросочетания, а не только счётчик.

Только стандартная библиотека. Ни stdin, ни input(), ни вызовов на импорте.
"""

from collections import deque

INF = float("inf")


# ---------------------------------------------------------------- max flow ---
class Dinic:
    """Max flow, Dinic with BFS levels + iterative DFS with current-arc.

    Complexity O(V^2 E) general, O(E sqrt(V)) on unit-capacity / bipartite nets.
    Capacities may be ints or Fractions (exact arithmetic preserved).
    """

    __slots__ = ("n", "graph", "_edges", "level", "it")

    def __init__(self, n):
        self.n = n
        # graph[u] = list of [v, cap, rev_index]
        self.graph = [[] for _ in range(n)]
        self._edges = []  # eid -> (u, index in graph[u])
        self.level = [-1] * n
        self.it = [0] * n

    def add_edge(self, u, v, cap):
        """Directed edge u->v with capacity cap. Returns an edge id."""
        if cap < 0:
            raise ValueError("negative capacity")
        self.graph[u].append([v, cap, len(self.graph[v])])
        self.graph[v].append([u, 0, len(self.graph[u]) - 1])
        self._edges.append((u, len(self.graph[u]) - 1))
        return len(self._edges) - 1

    def edge_flow(self, eid):
        """Flow pushed through the edge returned by add_edge."""
        u, idx = self._edges[eid]
        v, cap, rev = self.graph[u][idx]
        return self.graph[v][rev][1]  # residual of the back edge

    def _bfs(self, s, t):
        level = self.level
        for i in range(self.n):
            level[i] = -1
        level[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            lu = level[u]
            for v, cap, _rev in self.graph[u]:
                if cap > 0 and level[v] < 0:
                    level[v] = lu + 1
                    q.append(v)
        return level[t] >= 0

    def _augment(self, s, t):
        """Find ONE augmenting path in the level graph and push its bottleneck."""
        graph, level, it = self.graph, self.level, self.it
        path = []  # list of (u, edge_index_in_graph[u])
        u = s
        while True:
            if u == t:
                f = min(graph[x][i][1] for x, i in path)
                for x, i in path:
                    e = graph[x][i]
                    e[1] -= f
                    graph[e[0]][e[2]][1] += f
                return f
            advanced = False
            gu = graph[u]
            while it[u] < len(gu):
                v, cap, _rev = gu[it[u]]
                if cap > 0 and level[v] == level[u] + 1:
                    path.append((u, it[u]))
                    u = v
                    advanced = True
                    break
                it[u] += 1
            if not advanced:
                level[u] = -1  # dead end: never enter this node again
                if not path:
                    return 0
                pu, pi = path.pop()
                it[pu] += 1  # current-arc advances ONLY on a dead subtree
                u = pu

    def max_flow(self, s, t, limit=INF):
        if s == t:
            raise ValueError("source == sink")
        total = 0
        while total < limit and self._bfs(s, t):
            for i in range(self.n):
                self.it[i] = 0
            while total < limit:
                f = self._augment(s, t)
                if f == 0:
                    break
                if total + f > limit:
                    f = limit - total
                total += f
        return total

    def min_cut(self, s):
        """After max_flow: set of nodes reachable from s in the residual graph."""
        seen = [False] * self.n
        seen[s] = True
        q = deque([s])
        while q:
            u = q.popleft()
            for v, cap, _rev in self.graph[u]:
                if cap > 0 and not seen[v]:
                    seen[v] = True
                    q.append(v)
        return {i for i in range(self.n) if seen[i]}


# ------------------------------------------------------------ min-cost flow ---
class MinCostFlow:
    """Min-cost max-flow, successive shortest paths with SPFA (Bellman-Ford queue).

    Handles negative edge costs on forward edges as long as the input graph has no
    negative-cost cycle. Costs/capacities may be ints or Fractions.
    """

    __slots__ = ("n", "graph", "_edges")

    def __init__(self, n):
        self.n = n
        # graph[u] = list of [v, cap, cost, rev_index]
        self.graph = [[] for _ in range(n)]
        self._edges = []

    def add_edge(self, u, v, cap, cost):
        """Directed edge u->v, capacity cap, cost per unit. Returns an edge id."""
        if cap < 0:
            raise ValueError("negative capacity")
        self.graph[u].append([v, cap, cost, len(self.graph[v])])
        self.graph[v].append([u, 0, -cost, len(self.graph[u]) - 1])
        self._edges.append((u, len(self.graph[u]) - 1))
        return len(self._edges) - 1

    def convex_edges(self, u, v, marginal_costs):
        """Convex cost curve u->v: k-th unit costs marginal_costs[k].

        The trick harvested from jumping-rooks: one unit-capacity parallel edge per
        marginal cost. SSP always saturates the cheapest remaining one first, so the
        total cost of x units is the prefix sum -- exactly a convex piecewise-linear
        cost, with NO extra machinery. marginal_costs must be non-decreasing.
        """
        prev = None
        for c in marginal_costs:
            if prev is not None and c < prev:
                raise ValueError("marginal costs must be non-decreasing (convex)")
            prev = c
        return [self.add_edge(u, v, 1, c) for c in marginal_costs]

    def edge_flow(self, eid):
        u, idx = self._edges[eid]
        v, cap, cost, rev = self.graph[u][idx]
        return self.graph[v][rev][1]

    def flow(self, s, t, maxf=INF):
        """Push up to maxf units s->t at minimum cost. Returns (flow, cost)."""
        if s == t:
            raise ValueError("source == sink")
        n, graph = self.n, self.graph
        total_flow = 0
        total_cost = 0
        while total_flow < maxf:
            dist = [INF] * n
            par_node = [-1] * n
            par_edge = [-1] * n
            in_queue = [False] * n
            dist[s] = 0
            q = deque([s])
            in_queue[s] = True
            while q:
                u = q.popleft()
                in_queue[u] = False
                du = dist[u]
                for idx, e in enumerate(graph[u]):
                    v, cap, cost, _rev = e
                    if cap > 0 and du + cost < dist[v]:
                        dist[v] = du + cost
                        par_node[v] = u
                        par_edge[v] = idx
                        if not in_queue[v]:
                            q.append(v)
                            in_queue[v] = True
            if dist[t] == INF:
                break

            # bottleneck along the found path
            f = maxf - total_flow
            cur = t
            while cur != s:
                p, idx = par_node[cur], par_edge[cur]
                if graph[p][idx][1] < f:
                    f = graph[p][idx][1]
                cur = p

            cur = t
            while cur != s:
                p, idx = par_node[cur], par_edge[cur]
                graph[p][idx][1] -= f
                graph[cur][graph[p][idx][3]][1] += f
                cur = p

            total_flow += f
            total_cost += dist[t] * f
        return total_flow, total_cost


# ------------------------------------------------------------- bipartite ------
def kuhn_matching(adj, nleft, nright):
    """Maximum bipartite matching (Kuhn / augmenting paths), iterative.

    adj[u] = iterable of right-vertex indices for left vertex u (0 <= u < nleft).
    Returns (size, match_left, match_right) where
        match_left[u]  = matched right vertex or -1,
        match_right[v] = matched left  vertex or -1.
    Complexity O(V * E).
    """
    adj = [list(a) for a in adj]
    match_left = [-1] * nleft
    match_right = [-1] * nright
    size = 0

    for root in range(nleft):
        if match_left[root] != -1:
            continue
        visited = [False] * nright
        stack = [[root, 0]]  # [vertex, next edge index]
        while stack:
            u, i = stack[-1]
            if i >= len(adj[u]):
                stack.pop()
                continue
            stack[-1][1] = i + 1
            v = adj[u][i]
            if visited[v]:
                continue
            visited[v] = True
            w = match_right[v]
            if w == -1:
                # augment: every frame's last-used edge becomes a matched pair
                for uu, ii in stack:
                    vv = adj[uu][ii - 1]
                    match_right[vv] = uu
                    match_left[uu] = vv
                size += 1
                break
            stack.append([w, 0])
    return size, match_left, match_right


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

    print("Dinic")
    # hand-computable: s->a 3, s->b 2, a->t 3, b->t 2  =>  max flow 5
    d = Dinic(4)
    s, a, b, t = 0, 1, 2, 3
    ea = d.add_edge(s, a, 3)
    d.add_edge(s, b, 2)
    d.add_edge(a, t, 3)
    d.add_edge(b, t, 2)
    check("hand-made max flow = 5", d.max_flow(s, t), 5)
    check("edge_flow(s->a) = 3", d.edge_flow(ea), 3)

    # CLRS fig. 26.1 textbook network, known answer 23
    d = Dinic(6)
    for u, v, c in [
        (0, 1, 16),
        (0, 2, 13),
        (1, 2, 10),
        (2, 1, 4),
        (1, 3, 12),
        (3, 2, 9),
        (2, 4, 14),
        (4, 3, 7),
        (3, 5, 20),
        (4, 5, 4),
    ]:
        d.add_edge(u, v, c)
    check("CLRS network max flow = 23", d.max_flow(0, 5), 23)

    # min cut value must equal max flow (cut edges: 1->3 12, 4->3 7, 4->5 4 = 23)
    S = d.min_cut(0)
    cutval = sum(
        c
        for u, v, c in [
            (0, 1, 16),
            (0, 2, 13),
            (1, 2, 10),
            (2, 1, 4),
            (1, 3, 12),
            (3, 2, 9),
            (2, 4, 14),
            (4, 3, 7),
            (3, 5, 20),
            (4, 5, 4),
        ]
        if u in S and v not in S
    )
    check("min cut value = 23", cutval, 23)

    # capacity > 1 on a shared bottleneck (kills the current-arc bug of the source)
    d = Dinic(4)
    d.add_edge(0, 1, 5)
    d.add_edge(1, 2, 3)
    d.add_edge(1, 3, 4)
    d.add_edge(2, 3, 3)
    check("bottleneck 5 -> max flow = 5", d.max_flow(0, 3), 5)

    print("MinCostFlow")
    # s->a cap2 cost1, s->b cap2 cost2, a->t cap1 cost1, b->t cap3 cost1
    # best 3 units: 1 via a (cost 2) + 2 via b (cost 3 each) = 8
    m = MinCostFlow(4)
    m.add_edge(0, 1, 2, 1)
    m.add_edge(0, 2, 2, 2)
    m.add_edge(1, 3, 1, 1)
    m.add_edge(2, 3, 3, 1)
    check("min-cost flow (3, 8)", m.flow(0, 3, 3), (3, 8))

    # convex cost via parallel unit edges: marginals 0,1,2 -> 3 units cost 0+1+2 = 3
    m = MinCostFlow(3)
    m.add_edge(0, 1, 3, 0)
    m.convex_edges(1, 2, [0, 1, 2])
    check("convex parallel edges (3, 3)", m.flow(0, 2, 3), (3, 3))
    # ... and only 2 units cost 0+1 = 1
    m = MinCostFlow(3)
    m.add_edge(0, 1, 3, 0)
    m.convex_edges(1, 2, [0, 1, 2])
    check("convex, 2 units (2, 1)", m.flow(0, 2, 2), (2, 1))

    # negative costs on forward edges (SPFA must cope)
    m = MinCostFlow(3)
    m.add_edge(0, 1, 1, -5)
    m.add_edge(1, 2, 1, 2)
    check("negative cost path (1, -3)", m.flow(0, 2, 1), (1, -3))

    print("kuhn_matching")
    k33 = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    check("K3,3 matching = 3", kuhn_matching(k33, 3, 3)[0], 3)
    # three left vertices fighting over two right ones -> 2
    check("3 left / 2 right = 2", kuhn_matching([[0], [0, 1], [1]], 3, 2)[0], 2)
    check("empty adjacency = 0", kuhn_matching([[], [], []], 3, 3)[0], 0)
    # path 0-0, 1-0, 1-1 -> perfect on 2x2
    sz, ml, mr = kuhn_matching([[0], [0, 1]], 2, 2)
    check("path graph matching = 2", sz, 2)
    check(
        "matching is consistent",
        all(mr[v] == u for u, v in enumerate(ml) if v >= 0),
        True,
    )

    print("differential: Dinic vs MinCostFlow max-flow value (random graphs)")
    rng = random.Random(12345)
    diffs = 0
    for _ in range(200):
        n = rng.randint(4, 9)
        edges = []
        for _ in range(rng.randint(n, 3 * n)):
            u = rng.randrange(n - 1)
            v = rng.randrange(u + 1, n)
            edges.append((u, v, rng.randint(1, 9)))
        d = Dinic(n)
        m = MinCostFlow(n)
        for u, v, c in edges:
            d.add_edge(u, v, c)
            m.add_edge(u, v, c, rng.randint(0, 5))
        if d.max_flow(0, n - 1) != m.flow(0, n - 1)[0]:
            diffs += 1
    check("200 random graphs agree", diffs, 0)

    print("differential: kuhn_matching vs Dinic on random bipartite graphs")
    diffs = 0
    for _ in range(200):
        nl = rng.randint(1, 7)
        nr = rng.randint(1, 7)
        adj = [[v for v in range(nr) if rng.random() < 0.4] for _ in range(nl)]
        d = Dinic(nl + nr + 2)
        S, T = nl + nr, nl + nr + 1
        for u in range(nl):
            d.add_edge(S, u, 1)
            for v in adj[u]:
                d.add_edge(u, nl + v, 1)
        for v in range(nr):
            d.add_edge(nl + v, T, 1)
        if kuhn_matching(adj, nl, nr)[0] != d.max_flow(S, T):
            diffs += 1
    check("200 random bipartite graphs agree", diffs, 0)

    print("ALL OK" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_selfcheck())
