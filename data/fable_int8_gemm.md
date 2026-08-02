<!--
Fable consult: int8/fp16 GEMM schedule (partition + relabelling), 2026-08-02
model: claude-fable-5 | max-turns 12 | 1 turn used | reasoning 1245 s (20.75 min) | subtype=success, is_error=False
prompt: /tmp/claude-1000/-mnt-d1-alex-VLLM-fa2/44a942db-7eb7-4d06-ad86-a66ac5ce5f04/scratchpad/gemm8/gemm8_schedule_prompt.txt
        (280 lines, 22126 chars, pure ASCII; guard-scan EN/RU/loose-stem all empty)
raw events: .../scratchpad/gemm8/answer.json (774 events)
Glossary back to hardware: cluster=SM, dispatcher=warp scheduler, group=warp, executor=lane,
  heavy act=HMMA.884 (8x4x8 per quad-pair), private slot=register, fast store=shared memory,
  slow store=HBM, service round=wavefront, class=shared-memory bank, tick=cycle,
  starting=instruction issue, w=LDS width in 4-byte cells, nu=aux instructions per HMMA.
-->

## Answers first

**(1a)** Within your schedule class the heavy count is exactly MKN/256 — a one-line partition count. Strassen escapes the class and, re-priced correctly, **is profitable: one level for M ≥ ≈350 (so M = 512 and 4096 of your list, +14.3%), a second at M = 4096 (+30.6%, conditional on the accumulator margin). The earlier "32" that killed it was the lane factor: element-operations were priced as startings; a starting performs 32.**
**(1b)** ν\* = (1/m+1/n)·h_f/(2w) + 4/M + 4/N + 16/K, attained at G = 8, 64×64 tile, w = 4, 8-bit slow-store holding, 16-bit fast-store holding: ν\* = 1/16 + ε ≈ 0.067, share law 2/max(2,1+ν) = 100%; attainable ≈ 95–96%. Crossover: **M\* = 512h_s/(6.87η) = 87 at h_s = 1, η = 0.86** — your measured knee 128 is exactly 149×0.86, a slow-store shortfall, *not* structural.
**(1c)** The eight row slots **cannot** carry eight contraction segments — proved below; M = 1 is pure delivery: **≥ 107,300 ticks ideal, ≈ 124,800 at η = 0.86** for (15360, 3840).
**(1d)** G = 8 with the square tile, in both regimes; E4 forces it because Q(G)·G ≈ 2048 is invariant while cost scales with tile perimeter/area.
**(2a/2b)** Conflicts vanish **identically**: a GF(2)-rank-5 condition on σ, always satisfiable at your shapes, simultaneously for w = 1, 2, 4, realized by a tile-major-plus-XOR bijection. Padding can never do this: it acts by *translation* on classes; the condition is *linear*. The dual: k = 16 distinct addresses is forced by the two-to-one maps if executors fetch only what they present; k = 8 by over-fetch; both reach the absolute floor rounds = max(w/2, 1).
**(2c)** After σ\* the residual gap is **not** class conflicts. It is: bare-chain issue overhead (3.2%, your 96.8), placement tax ≈ 0.49·ν\*/2 ≈ 1%, slow-store η = 0.86 in Regime I, and C-drain 4/(4r) bytes per MAC at shallow flush depths — arithmetic below.

---

## Q1a — invariance, and Strassen re-priced

**Invariance.** Every box covers exactly 8·4·8 = 256 lattice triples and boxes partition L, so #boxes = MKN/256; heavy startings = #boxes. (Requires 8|M, 4|K, 8|N; K, N always comply, M = 1 does not — see 1c.) ∎

**Strassen, exact counts per level** (split M, K, N by 2; boxes then live inside half-panels — the top bit of each axis is consumed by the level, otherwise the partition of Q1b is unchanged):

- Saves MKN/2048 heavy startings (⅛).
- **A-side:** 5 block-additions of MK/4 elements. One auxiliary starting performs 32 lane-additions, so **5MK/128 startings**, not in the free additive input.
- **B-side:** 0 startings if pre-combined — but the 7 combination blocks are 9-bit, i.e. 7·(KN/4)·2 = 3.5KN units, which **violates your free-preparation capacity clause (≤ KN units)**. The real price is delivery: 3.5KN units against KN·h_s.
- **Output side:** the 7 products appear 12 times (signed) in the four C-blocks. Chaining accumulators through the free additive input (compute M2, seed M4's product into the same accumulator, etc.) realizes **7 occurrences free; the remaining 5 cost 5MN/128 startings.**

Added ν per level: (5MK/128 + 5MN/128)/(7MKN/2048) = **(80/7)(1/K + 1/N) ≤ 0.009** at all four shapes — negligible, and this is where the old verdict inverts: your "2MN output startings" were element-ops; divide by 32 lanes.

**Levels.** Operands grow as 2^ℓ·128 < 2^11 ⇒ **ℓ ≤ 3**. Worst-case accumulator exactness needs flush depth D·2^(14+2ℓ) < 2^24 ⇒ D ≤ 2^(9−2ℓ); the measured √-growth law scales it by 4^ℓ from your depth-16384 point, admitting ℓ = 2 generically, excluding ℓ = 3.

**Law in M.** Bytes per multiply-add with pre-combined 16-bit B: 3.5KN/((7/8)MKN) = 4/M; heavy-bound requires 4/M ≤ 6.87η/512, i.e. **M ≥ 347**. Level 2: 8/M ⇒ M ≥ 694. Verdict across all four shapes (the shape enters only through the 0.009): **M ≤ 128 harmful; M = 512 one level, gain 8/7; M = 4096 two levels, gain 64/49 (second level conditional as above).** An in-fast-store combination variant (store the four original blocks, 8-bit, legal prep; form combos per residency at 40/(7M) startings) keeps traffic at KN and pushes the knee down toward M ≈ 100, at the price of 3.5×-narrower panels; use it only if panel perimeter (2c formula) stays sub-critical.

## Q1b — the floor on ν and the partition

Per group tile 8m×8n, one depth-4 slab: each executor presents 2 A-values per act it serves at each of m row-blocks and 2 B-values at each of n column-blocks — the two-to-one presentation maps force **duplicated** conversion in partner executors (private slots cannot be shared), so per executor per slab: 2(m+n) value-instances over mn acts.

- **Loads:** every presented byte transits fast store→slots at ≤ 4w bytes/executor/starting (w ≤ 4 is the machine maximum): ν_load = (1/m+1/n)·h_f/(2w).
- **Presentation:** hold B 8-bit in the *slow* store (h_s = 1, halving delivery) but present **once per fast-store residency** and hold the 16-bit form there (h_f = 2). Cost: (V/64 startings)/(MV/256 acts) = **4/M** for B, symmetrically **4/N** for A. This settles the re-presentation trade: re-presentation multiplicity R = residency passes; ν_pres = 4R/M, and R = 1 in both regimes below. Storing 16-bit in slow store instead costs a factor 2 in delivery (knee 149 vs 87) to save 4/M — never worth it.
- **Flush:** 2mn accumulator values/executor per mn·K/4 acts, ≈ 2 startings/value (scale-FMA + pack; store rides w = 4): **16/K**.
- **Class conflicts: 0** with σ\* of Q2.

**ν\*(m,n) = (1/m+1/n)h_f/(2w) + 4/M + 4/N + 16/K.** Minimality: each term is a counting floor (lane width 32, access payload 4w, the established presentation floor, accumulator size). The tile bound: 2mn + 4(m+n) staging ≤ Q(8) − 7 = 248 excludes m=n=16 (512) and 8×16 (2mn=256); by AM-GM the perimeter (1/m+1/n) at fixed mn is minimized square: **m = n = 8** is optimal-admissible (E3's 95.2% row). So ν\* = 1/16 + 4/M + 4/N + 16/K ≈ **0.067** at (3840-class shapes, M ≥ 149).

**Regimes.** Heavy-bound iff slow bytes/MAC = h_s/M + perimeter terms ≤ 6.87η/512: crossover **M\* = 74.5h_s/η = 87** at h_s = 1. 
*Regime I (M ≤ 86):* duration = (KN + 2MK⌈N/W⌉)/(80·6.87η); clusters split N into 80 slices; A resident where 2MK ≤ capacity, else widest W; B streamed exactly once; tile 8m×64 with m = min(M/8, 8); share = ηM/74.5 exactly (your 86.0/83.4 rows confirm η). 
*Regime II (M ≥ 87):* clusters tile C in an 8×10 grid (per-cluster block ~M/8 × N/10, adjusted to divisors); traversal k-innermost with flush depth 4r = min(K, 4096-generic / 256-worst-case), then n, then m; B-slice resident per cluster panel, delivered once.

## Q1c — the degenerate end

**Obstruction theorem.** An act's outputs are acc(i,j) = Σ_{u∈U} P_A(i,u)P_B(u,j) with **one** U shared by both operands. Carrying segment κ_i in row-slot i requires P_B(u,j) = B(g_i(u), t_j) to depend on i — impossible: P_B is presented once per act. And within the act, useful outputs ≤ 8·rank(P_A) ≤ 8·(#distinct available rows) = 8 at M = 1, i.e. ≤ 32 useful multiply-adds of 256. Replicating B with shifted k-origins would need 8× its storage. **Forbidden.** ∎

So M = 1 is delivery: lower bound = KN/(80·6.87) = **107,319 ticks** for (15360, 3840); at η = 0.86, **≈ 124,800**. Matching construction: σ\* of Q2, w = 4 all-distinct accesses (rounds at floor), padded acts (KN/32 startings ⇒ 11,520 dispatcher-ticks/cluster ≪ bound — fully hidden). Nothing else can matter.

## Q1d — residency decided

Q(G) = min(255, 8⌊256/G⌋) makes **Q·G ≈ 2048 invariant**: cluster-wide slot capacity does not grow with G. Residency (latency hiding) is already saturated by the slot law once ν ≤ 1 — which ν\* = 0.067 satisfies eight-fold — so G's only surviving effect is that per-executor slots halve at G = 16, forcing 2mn ≤ ~100, hence perimeter (1/m+1/n) ≥ 0.5 vs 0.25: **every ν term doubles, duration cannot decrease. G = 8 square wins Regime II; Regime I is slow-store-set (G immaterial), tie broken toward G = 8 by A-residency capacity.** That is E4 as a mechanism, not a coincidence: residency, traffic, and accesses-per-act compete for one invariant slot pool, and accesses-per-act is the only term the duration law still sees.

## Q2a — the admissible bijections

Work at cell granularity; class π = the low 5 bits of the cell index. Choose boxes as 2-adic cosets (free). For each table, an act-side's request set is a coset of a fixed difference space V (3 row/col bits + 2 k bits; dim ≤ 5 cells' worth at any holding).

**Theorem.** For requests of R distinct w-runs, service rounds achieve the absolute floor ⌈wR/32⌉ for w = 1, 2, 4 **simultaneously** iff the GF(2)-linear map π∘σ restricted to V has rank min(5, dim_cell V) *and* σ is run-contiguous on each executor's presentation stream. In general c = 2^(dim − rank) × floor, which is your factor-32 pathology: naive row-major with 32 | ld/4 kills the k-digits, rank deficit 5 → 0 on the k-direction.

**Satisfiability.** dim of class space = 5; every axis has 2-adic valuation ≥ 8 (3840 = 2^8·15, 15360 = 2^10·15, 4096 = 2^12), so σ can place 5 independent V-digits into the class positions for all four shapes at once; the odd part 15 lives in high digits and never touches class arithmetic. **No obstruction exists at these shapes**; one would appear only if a box-side spanned fewer than rank-many independent 2-adic digits (e.g. an odd leading dimension under 32 cells). **Simultaneity across w is automatic** because w = 2, 4 runs refine the same linear structure (run interiors occupy consecutive classes; distinct runs are separated by the rank condition on the quotient). **Why padding dies and this doesn't (D3):** padding adds a per-row constant — an affine *translation* of classes — and the translations demanded by w = 2 and w = 4 granularities are incompatible; the rank condition is on the *linear* part, which one σ fixes for all w at once.

## Q2b — constructions and the dual

**(i) Unit multiplication mod 32:** multiplying the cell index by 15 is an automorphism of Z_32 composed after the defective map — rank is preserved, **no**. Digit reordering: **yes iff** the five class positions receive 2 digits of k mod 4 and 3 of n mod 8 (for B; rows for A) — that is the exact condition, a special case of the theorem.
**(ii) XOR relabelling:** store B tile-major (each box-side contiguous) and XOR the executor-stream digits into the class field — explicit positions in the trace below; conflict-free for w = 1, 2, 4 at once by the theorem. This is the observed seed, now proved.
**(iii) Cosets:** **yes, and it is the same construction** — tile-major σ makes every access set a coset of the fixed subgroup generated by the low 5 cell-digits; π restricted to it is the identity homomorphism, injectivity automatic rather than checked. No invertible recoding of *values* is needed for classes; value-recoding has exactly one profitable use here: the Strassen B-combinations of Q1a.

**The dual (coincidence).** The A-map's kernel is {0, l⊕4}, the B-map's {0, l⊕8} (read off the given bit-formulas: bit 2 resp. bit 3 unused). If each executor fetches only operands it presents, the address map factors injectively through the 16-element image: **k = 16 is forced, DUP = 2 — this alone repairs E5.** k = 8 (DUP = 4) is attainable by over-fetch (an executor keeps a whole run of which it uses part; slots permitting), k = 2 in the limit — but each halving of k halves useful bytes per round, and once rounds sit at the bandwidth floor max(w/2, 1) further coincidence is pure waste. Since at the optimal tile rounds/act = 0.125 ≪ the 0.5 budget, **kernel-pair duplication (k = 16) is the optimum; deeper coincidence is provably never needed in Regime II and useful only where the dispatcher idles anyway.**

## Q2c — what it is worth

With σ\* and the Q1b partition: ν\* ≈ 0.067, rounds at floor, share by the dispatcher law = 2/max(2, 1.067) = **100%**. Against measurement: the bare chain itself reaches only 96.8% (issue/pipeline overhead of the heavy chain — irreducible per E2), and E6's placement tax adds ≈ 0.49·ν\*/2 ≈ 1%: **attainable ≈ 95–96%.** The residual against E2's 46–76% decomposes, with arithmetic:

- **C-drain:** flushing every 4r costs ≈ 4/(4r) bytes/MAC round-trip; at depth 256 with B at 2/M this caps share near (6.87η/512)·(6/256)⁻¹ ≈ 49% — your measured 46. It vanishes at 4r = 4096 (generic exactness holds by the √-law).
- **Panel perimeter:** slow traffic 2h_s(1/W_m + 1/W_n) per MAC; with per-cluster C-blocks ≈ 480×384 at the live shapes this is 1/106 B/MAC — uncapped. It, not conflicts, throttled the mid-depth E2 points.
- **η = 0.86:** the slow store sustains 86% of nominal at these shapes (three independent confirmations: 86.0 and 83.4 measured shares, and 149·0.86 = 128 = your knee). **The 15% discrepancy is a slow-store shortfall, not structural.**

**Class conflicts contribute zero after σ\* — and, at G = 8 tiles, they were already sub-budget (0.26 rounds/act vs 0.5); they were never the E2 gap. The subject closes on: install σ\* (for the access-count floor and E5), flush at 4r = 4096, hold B 8-bit in slow store / 16-bit in fast store, and accept ≈ 96% as the ceiling; the last 4% is the bare-chain overhead plus placement, which no partition or bijection reaches.**

## Numeric trace — M = 8, K = 64, N = 64, one cluster, one group

**Boxes** (128 = 8·64·64/256): S = {0..7}; U_q = {4q..4q+3}, q < 16; T_t = {8t..8t+7}, t < 8. Schedule: q outer, t inner; all 8 accumulators (16 values/executor) live across the q-sweep.

**σ_B on digits.** Write k = 4q + 2h + r₁, n = 8t + s; slot = 8h + s ∈ [0,16). Unit address u = 1024q + 64·slot + 4t + 2r₁ + b (16-bit fast-store holding, b the byte). Cell = u/4 = 256q + 16·slot + t, then **XOR the two high slot-digits into the class field: cell′ = cell ⊕ 2⌊slot/2⌋.** Each executor's 8-act stream (its slot, t = 0..7) is 8 contiguous cells — two w = 4 runs.

**First access** (q = 0, t = 0..3): 32 executors, kernel-pairs (l, l⊕8) coincide → 16 distinct runs, 64 cells, c = 2 = 64/32: **rounds = 2, the floor w/2.** 
**First two acts** (q=0,t=0), (q=0,t=1) — 32 cells (act 1 / act 2 = same +1):

| slot | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cell (t=0) | 0 | 16 | 34 | 50 | 68 | 84 | 102 | 118 | 136 | 152 | 170 | 186 | 204 | 220 | 238 | 254 |
| class | 0 | 16 | 2 | 18 | 4 | 20 | 6 | 22 | 8 | 24 | 10 | 26 | 12 | 28 | 14 | 30 |

Act 1 hits the 16 even classes once each; act 2 (cells +1) the 16 odd: the two acts fill all 32 classes bijectively.

**Totals.** Heavy: 128 startings (256 heavy-channel ticks). Auxiliary, steady-state: B-loads 32 (2048 cells ÷ 64/access; 64 rounds), A-loads 2, A-presentations 8, output scale 16 + pack 4 + store 4, rank-one row-sums 8 (column-sums and seeding free): **74**. One-time B preparation into fast store (64 presentations + 8 loads + 16 stores = 88) amortizes over B's reuse. **ν = 74/128 = 0.578; share = 2/max(2, 1.578) = 100%; duration = 256 ticks** (290 ticks, 88.3%, if the one-time prep is charged to a single product). Rounds 76 ≤ 256: hidden. In the wild M = 8 is Regime I: share = ηM/74.5 = **9.2%**, delivery-bound exactly as the law says.