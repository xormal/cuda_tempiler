// SPDX-License-Identifier: LicenseRef-TRL-1.0
// Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
// Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
//
// СКЕЛЕТ ТЕМПОЛЯТОРА: плотное fp16-умножение C = A * B^T на HMMA.884 (sm_70).
//
// Это НЕ ядро, а СЕМЕЙСТВО ядер: всё, что здесь стоит шаблонным параметром, -- ось поиска.
// Ядро выдаётся конвейером как ОДНА точка этого произведения; выбор точки делает замер, а не
// рассуждение. Оси и то, какой ресурс каждая двигает:
//
//   BM,BN,BK       геометрия плитки блока      -> трафик глобальной, ЗАПИСЬ в разделяемую, smem
//   WM,WN          раскладка варпов по плитке  -> ПЛИТКА ВАРПА (MB x NB) = ЧТЕНИЕ разделяемой
//   STAGES         число буферов в разделяемой -> сокрытие задержки глобальной, smem
//   GSTAGE         глубина регистровой пересылки (сколько плиток В ПОЛЁТЕ) -> long_scoreboard, регистры
//   FPREF          предвыборка фрагментов внутри плитки -> short_scoreboard (задержка LDS = 26)
//   GROUP          переупорядочение блоков под L2 -> попадания в L2, трафик DRAM
//   EPI            эпилог: 0 = прямые парные записи, 1 = через разделяемую (сплошные STG.128)
//   PRED           предикация по M (грид с округлением вверх)
//
// ЗАКОН, ПО КОТОРОМУ ОСИ ОТСЕКАЮТСЯ ДО СБОРКИ (выведен из карты фрагмента, см.
// tempo/core/model/gemm_hmma_bound.py):
//   вайвфронтов разделяемой на ТАКТ ТЕНЗОРНОГО КОНВЕЙЕРА
//       nu_mio = CONF*(MB+NB)/(2*MB*NB)  [чтение фрагментов]  + 4/BM + 4/BN  [запись плитки]
//   MIO отдаёт 1 вайвфронт/такт/SM, тензорный -- 2 КВАДРОПАРЫ (SASS HMMA)/такт/SM.
//   Компоновка 128x128, 4 варпа, плитка варпа 64x64 даёт 0.3125 при CONF=1 -- разделяемая НЕ
//   связывает. ЗАМЕР подтвердил: MIO занят на 46.8%, у cuBLAS на том же месте 46.4%.
//   ВЫВОД ОТСЕКАТЕЛЯ: геометрия плитки -- НЕ рычаг на этой форме, и полный перебор геометрий это
//   подтвердил (44 гиперформы, разброс 0.68..0.84, победила уже отгруженная).
//
// ЕДИНИЦА, НА КОТОРОЙ ЗАКОН БЫЛ СНАЧАЛА НЕВЕРЕН В 4 РАЗА. Первая редакция считала mma.sync.m8n8k4
// ОДНОЙ командой ценой 2 такта. На деле она распадается на ЧЕТЫРЕ квадропарных SASS HMMA, и 2 такта
// -- цена КВАДРОПАРЫ. Закон давал 1.25 ("разделяемая связывает") вместо 0.3125 ("не связывает"), и
// рычаг был бы назван неверно. Опровергнуто счётчиком sm__inst_executed_pipe_tensor.
//
// ЛОВУШКА, СТОИВШАЯ ПРОЕКТУ ЯДРА (не убирать этот абзац):
// mma.sync.aligned.m8n8k4 выдаётся ПО КВАДРОПАРАМ, а не по варпу. Наивная подача "полоса l даёт
// строку l" считает ЧЕТВЕРТЬ плитки и ПРОХОДИТ сверку значений, потому что сверка смотрит ровно те
// ячейки, которые посчитаны. Карта подачи ниже -- замеренная (tools/hmma_map_probe.py боевого
// дерева), проверять её надо БИЕКЦИЕЙ покрытия, а не значениями.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace tempo {
namespace gen {

// --------------------------------------------------------------------------------------------
// Инструкция. Накопление НА МЕСТЕ через "+f": форма с раздельными C и D стоит 8 копий регистров
// на выдачу, и при 16 mma на шаг это 128 MOV, конкурирующих за слоты выдачи с самими HMMA.
// НЕ volatile: volatile прибивает порядок и запрещает ptxas переслаивать LDS шага k+1 с HMMA шага k.
// --------------------------------------------------------------------------------------------
__device__ __forceinline__ void hmma884(float (&d)[8], const uint32_t (&a)[2], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 800)
  asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
#else
  (void)d; (void)a; (void)b;
#endif
}

// Замеренная карта: четыре квадропары ложатся на четыре квадранта ПЛОТНОЙ плитки 16x16.
__device__ __forceinline__ int a_row(int l) { return (l & 3) | ((l & 16) >> 2) | (l & 8); }
__device__ __forceinline__ int b_col(int l) { return (l & 3) | ((l & 16) >> 2) | ((l & 4) << 1); }
__device__ __forceinline__ int acc_row(int l, int r) { return ((l & 1) | ((l & 16) >> 2) | (l & 8)) + 2 * ((r >> 1) & 1); }
__device__ __forceinline__ int acc_col(int l, int r) { return ((l & 2) | ((l & 4) << 1)) + (r & 1) + 4 * ((r >> 2) & 1); }

// --------------------------------------------------------------------------------------------
// Раскладка разделяемой: XOR-свиззл вместо дополнения.
// Дополнение до LD=BK+4 даёт бесконфликтные 8-байтовые чтения, но ЗАПРЕЩАЕТ 16-байтовые (шаг 72 Б
// оставляет нечётные строки выровненными лишь на 8). Свиззл снимает дополнение: чанк c строки r
// лежит на c ^ (r & (NC-1)). Обе стороны становятся одиночными 128-битными обращениями.
// --------------------------------------------------------------------------------------------
// SWZ=0 -- как в отгруженном ядре: x = row & (NC-1).
// SWZ=1 -- ИСПРАВЛЕННЫЙ: x = (row >> SH) & (NC-1), где SH -- число НИЖНИХ разрядов строки, уже
//          съеденных шагом строки.
//
// ПОЧЕМУ SWZ=0 НЕВЕРЕН (замерено, не выведено). Строка занимает BK/2 слов. Стартовый банк строки r
// равен (r*BK/2) mod 32, и он ПОВТОРЯЕТСЯ с периодом P = 64/BK строк: при BK=32 период 2, при BK=16
// период 4. XOR по НИЖНИМ разрядам строки не различает строки, отличающиеся на P, -- он крутит ровно
// те разряды, которые шаг строки уже обнулил. В итоге 16 строк фрагмента садятся на 4 стартовые
// позиции вместо 8, и 64 слова ложатся в 16 банков вместо 32: четырёхкратная конфликтность там, где
// шапка отгруженного заголовка обещает бесконфликтность.
//
// SWZ=1 ОПРОВЕРГНУТ ЗАМЕРОМ: счётчик конфликтов не сдвинулся НИ НА ПРОЦЕНТ (177.93 -> 178.05 млн).
// Рассуждение "период стартового банка" верно, но НЕ ТА ЕДИНИЦА: банковый конфликт считается не по
// варпу, а по ФАЗЕ ИЗ ВОСЬМИ ПОЛОС (8 полос x 16 Б = 128 Б = один вайвфронт). Внутри одной фазы
// сталкиваются не соседние строки, а строки, отличающиеся на 8.
//
// ОТКУДА БЕРЁТСЯ ПРАВИЛЬНЫЙ ОТВЕТ -- ИЗ КАРТЫ ПОЛОС, а не из шага строки:
//   a_row(l) = (l&3) | ((l&16)>>2) | (l&8)   -- в фазе полос 0..7 даёт 4 РАЗНЫЕ строки (каждая
//              дважды: рассылка, бесплатна)  -> сторона A конфликтов не даёт;
//   b_col(l) = (l&3) | ((l&16)>>2) | ((l&4)<<1) -- в той же фазе даёт ВОСЕМЬ разных строк
//              {0,1,2,3,8,9,10,11}          -> сторона B и есть источник конфликта.
// Шаг строки при BK=32 (16 слов) сохраняет только разряд 0 строки, XOR по (row&3) -- разряды 0,1.
// Разряд 3, которым и отличаются 0 и 8, не участвует НИ ГДЕ -> строки 0 и 8 садятся на один банк.
// ЛЕЧЕНИЕ: свернуть в свиззл ровно те разряды, которые различают строки ВНУТРИ ФАЗЫ и не заняты
// шагом: {1,3} при BK=32, {3} при BK=16, {0,1,3} при BK=64.
//
// ЗАМЕР ДО ЛЕЧЕНИЯ: 2.04 конфликта на команду LDS.128 (60.03 млн на 29.49 млн команд), при том что
// у cutlass-ядра cuBLAS на ТОМ ЖЕ числе команд (30.01 млн) конфликтов 0.11. Предсказание модели:
// 16 команд стороны B из 32 x 4 лишних вайвфронта / 32 = РОВНО 2.0. Совпало с 2.04.
template <int BK, int SWZ>
struct Sw {
  static constexpr int NC = BK / 8;
  static constexpr int P = 64 / BK > 1 ? 64 / BK : 1;              // период повтора стартового банка
  static constexpr int SH = P == 8 ? 3 : (P == 4 ? 2 : (P == 2 ? 1 : 0));
  static_assert(BK % 8 == 0 && (NC & (NC - 1)) == 0, "BK/8 должно быть степенью двойки");
  // SWZ=2 -- ФАЗОВО-ИНЪЕКТИВНЫЙ. Выведен из КАРТЫ ПОЛОС, а не из шага строки (см. ниже).
  __device__ static __forceinline__ int x_of(int row) {
    if (SWZ == 0) return row & (NC - 1);
    if (SWZ == 1) return (row >> SH) & (NC - 1);
    if (SWZ >= 10) {   // ПЕРЕБОРНАЯ форма: SWZ = 10*b1 + b0, x = бит b0 строки | (бит b1)<<1
      const int b0 = SWZ % 10, b1 = (SWZ / 10) % 10;
      return ((row >> b0) & 1) | (((row >> b1) & 1) << 1);
    }
    if (BK == 64) return (row & 3) | ((row >> 1) & 4);          // разряды строки 0,1,3
    if (BK == 32) return ((row >> 1) & 1) | ((row >> 2) & 2);    // разряды строки 1,3
    if (BK == 16) return (row >> 3) & 1;                         // разряд строки 3
    return row & (NC - 1);
  }
  __device__ static __forceinline__ int off(int row, int c) { return row * BK + ((c ^ x_of(row)) * 8); }
};

// --------------------------------------------------------------------------------------------
// Пересылка глобальная -> регистры -> разделяемая. DEPTH плиток В ПОЛЁТЕ одновременно:
// это и есть ось, закрывающая long_scoreboard (задержка LDG ~400 тактов), ценой регистров.
// --------------------------------------------------------------------------------------------
template <int ROWS, int BK, int NTHR, int DEPTH, int SWZ>
struct Loader {
  static constexpr int kChunks = ROWS * BK / 8;
  static constexpr int kPer = kChunks / NTHR;
  static constexpr int kPerRow = BK / 8;
  static_assert(kPer >= 1, "нитей больше, чем 16-байтовых порций в плитке");
  static_assert(kChunks % NTHR == 0, "плитка не делится на нити нацело");
  uint4 buf[DEPTH][kPer];
  int row[kPer], col[kPer];

  __device__ __forceinline__ void init(int tid) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const int idx = tid + i * NTHR;
      row[i] = idx / kPerRow;
      col[i] = (idx % kPerRow) * 8;
    }
  }
  template <bool PRED>
  __device__ __forceinline__ void ldg(int slot, const __half* g, int ld, int k0, int rows) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      if (!PRED || row[i] < rows) buf[slot][i] = *reinterpret_cast<const uint4*>(g + (long)row[i] * ld + k0 + col[i]);
      else buf[slot][i] = make_uint4(0, 0, 0, 0);
    }
  }
  __device__ __forceinline__ void sts(int slot, __half* s) const {
#pragma unroll
    for (int i = 0; i < kPer; ++i)
      *reinterpret_cast<uint4*>(s + Sw<BK, SWZ>::off(row[i], col[i] / 8)) = buf[slot][i];
  }
};

// --------------------------------------------------------------------------------------------
// Плитка варпа. FPREF -- на сколько чанков вперёд идёт предвыборка фрагментов.
// FPREF=1 -- как в отгруженном ядре; FPREF=2 покрывает 26 тактов задержки LDS при малом MB*NB.
// --------------------------------------------------------------------------------------------
template <int MB, int NB, int BK, int FPREF, int SWZ>
__device__ __forceinline__ void warp_tile(float (&acc)[MB][NB][8],
                                          const __half* sA, int rowA,
                                          const __half* sB, int rowB) {
  constexpr int NC = BK / 8, kMask = NC - 1, DEP = FPREF + 1;
  const __half* pA = sA + rowA * BK;
  const __half* pB = sB + rowB * BK;
  const int xa = Sw<BK, SWZ>::x_of(rowA), xb = Sw<BK, SWZ>::x_of(rowB);
  uint4 fa[DEP][MB], fb[DEP][NB];
#pragma unroll
  for (int p = 0; p < FPREF && p < NC; ++p) {
#pragma unroll
    for (int i = 0; i < MB; ++i) fa[p][i] = *reinterpret_cast<const uint4*>(pA + (long)(i * 16) * BK + ((p ^ xa) * 8));
#pragma unroll
    for (int j = 0; j < NB; ++j) fb[p][j] = *reinterpret_cast<const uint4*>(pB + (long)(j * 16) * BK + ((p ^ xb) * 8));
  }
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    const int b = c % DEP;
    if (c + FPREF < NC) {
      const int nb_ = (c + FPREF) % DEP, cc = c + FPREF;
#pragma unroll
      for (int i = 0; i < MB; ++i) fa[nb_][i] = *reinterpret_cast<const uint4*>(pA + (long)(i * 16) * BK + ((cc ^ xa) * 8));
#pragma unroll
      for (int j = 0; j < NB; ++j) fb[nb_][j] = *reinterpret_cast<const uint4*>(pB + (long)(j * 16) * BK + ((cc ^ xb) * 8));
    }
    uint32_t alo[MB][2], ahi[MB][2], blo[NB][2], bhi[NB][2];
#pragma unroll
    for (int i = 0; i < MB; ++i) { alo[i][0] = fa[b][i].x; alo[i][1] = fa[b][i].y; ahi[i][0] = fa[b][i].z; ahi[i][1] = fa[b][i].w; }
#pragma unroll
    for (int j = 0; j < NB; ++j) { blo[j][0] = fb[b][j].x; blo[j][1] = fb[b][j].y; bhi[j][0] = fb[b][j].z; bhi[j][1] = fb[b][j].w; }
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j) hmma884(acc[i][j], alo[i], blo[j]);
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j) hmma884(acc[i][j], ahi[i], bhi[j]);
  }
}

// Переупорядочение блоков под L2: подряд идущие блоки образуют компактный столбец GROUP x всё-N.
template <int GROUP>
__device__ __forceinline__ void block_swizzle(int bid, int nm, int nn, int& bm, int& bn) {
  if (GROUP <= 1) { bm = bid / nn; bn = bid % nn; return; }
  const int per = GROUP * nn;
  const int grp = bid / per, first = grp * GROUP;
  const int gs = min(GROUP, nm - first);
  bm = first + (bid % per) % gs;
  bn = ((bid % per) / gs);
}

// --------------------------------------------------------------------------------------------
// Ядро.
// --------------------------------------------------------------------------------------------
template <int BM, int BN, int BK, int WM, int WN, int STAGES, int GSTAGE, int FPREF,
          int GROUP, int EPI, int SWZ, bool PRED, int MINB>
__global__ __launch_bounds__(WM* WN * 32, MINB) void k_gemm(const __half* __restrict__ A,
                                                            const __half* __restrict__ B,
                                                            __half* __restrict__ C,
                                                            int M, int N, int K) {
  constexpr int NTHR = WM * WN * 32;
  constexpr int MB = BM / (16 * WM);
  constexpr int NB = BN / (16 * WN);
  constexpr int EPI_LD = BN + 8;                 // шаг строки плиты эпилога, в half
  extern __shared__ __half smem[];
  __half* sA = smem;
  __half* sB = smem + STAGES * BM * BK;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int bmi, bni;
  block_swizzle<GROUP>(blockIdx.x, (M + BM - 1) / BM, N / BN, bmi, bni);
  const int bm = bmi * BM, bn = bni * BN;
  const int rows = PRED ? min(BM, M - bm) : BM;

  Loader<BM, BK, NTHR, GSTAGE, SWZ> la; la.init(tid);
  Loader<BN, BK, NTHR, GSTAGE, SWZ> lb; lb.init(tid);
  const __half* gA = A + (long)bm * K;
  const __half* gB = B + (long)bn * K;
  const int NT = K / BK;

  // Пролог: заполнить STAGES-1 буферов, затем поднять GSTAGE плиток в полёт.
#pragma unroll 1
  for (int t = 0; t < STAGES - 1; ++t) {
    la.template ldg<PRED>(0, gA, K, t * BK, rows);
    lb.template ldg<false>(0, gB, K, t * BK, BN);
    la.sts(0, sA + t * BM * BK);
    lb.sts(0, sB + t * BN * BK);
  }
#pragma unroll
  for (int g = 0; g < GSTAGE; ++g) {
    const int t = STAGES - 1 + g;
    if (t < NT) {
      la.template ldg<PRED>(g, gA, K, t * BK, rows);
      lb.template ldg<false>(g, gB, K, t * BK, BN);
    }
  }
  __syncthreads();

  const int wm = warp / WN, wn = warp % WN;
  const int rowA = wm * MB * 16 + a_row(lane), rowB = wn * NB * 16 + b_col(lane);
  float acc[MB][NB][8];
#pragma unroll
  for (int i = 0; i < MB; ++i)
#pragma unroll
    for (int j = 0; j < NB; ++j)
#pragma unroll
      for (int r = 0; r < 8; ++r) acc[i][j][r] = 0.f;

  // Катящиеся индексы вместо остатков: остаток от непостоянного делителя -- это деление в теле,
  // а тело исполняется K/BK раз.
  int cur = 0;                       // буфер, из которого СЧИТАЕМ плитку t
  int wbuf = (STAGES - 1) % STAGES;  // буфер, в который СДАЁМ плитку t+STAGES-1
  int slot = 0;                      // регистровая ячейка пересылки
#pragma unroll 1
  for (int t = 0; t < NT; ++t) {
    warp_tile<MB, NB, BK, FPREF, SWZ>(acc, sA + cur * BM * BK, rowA, sB + cur * BN * BK, rowB);
    if (t + STAGES - 1 < NT) {
      la.sts(slot, sA + wbuf * BM * BK);
      lb.sts(slot, sB + wbuf * BN * BK);
    }
    __syncthreads();
    const int tldg = t + STAGES - 1 + GSTAGE;
    if (tldg < NT) {
      la.template ldg<PRED>(slot, gA, K, tldg * BK, rows);
      lb.template ldg<false>(slot, gB, K, tldg * BK, BN);
    }
    cur = (cur + 1 == STAGES) ? 0 : cur + 1;
    wbuf = (wbuf + 1 == STAGES) ? 0 : wbuf + 1;
    slot = (slot + 1 == GSTAGE) ? 0 : slot + 1;
  }

  const int r0 = bm + wm * MB * 16, c0 = bn + wn * NB * 16;
  if (EPI == 0) {
    // Прямые записи парами: два соседних накопителя (r, r+1) -- одна строка и соседние столбцы,
    // то есть один 32-битный STG вместо двух 16-битных.
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int r = 0; r < 8; r += 2) {
          const int m = r0 + i * 16 + acc_row(lane, r), n = c0 + j * 16 + acc_col(lane, r);
          if (m < M && n + 1 < N) {
            const __half2 v = __floats2half2_rn(acc[i][j][r], acc[i][j][r + 1]);
            *reinterpret_cast<__half2*>(C + (long)m * N + n) = v;
          } else if (m < M && n < N) {
            C[(long)m * N + n] = __float2half(acc[i][j][r]);
          }
        }
  } else {
    // Через разделяемую: накопители складываются в плиту 16*WM строк x BN столбцов, затем плита
    // выносится СПЛОШНЫМИ 128-битными записями. Прямая запись даёт по 8 байт на строку варпа
    // (карта накопителя раскидывает полосы), то есть 1/16 полезной ширины транзакции.
    __half* ep = smem;
    const int slab = 16 * WM;
#pragma unroll 1
    for (int i = 0; i < MB; ++i) {
      __syncthreads();
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int r = 0; r < 8; r += 2) {
          const int sr = wm * 16 + acc_row(lane, r);
          const int sc = wn * NB * 16 + j * 16 + acc_col(lane, r);
          *reinterpret_cast<__half2*>(ep + (long)sr * EPI_LD + sc) = __floats2half2_rn(acc[i][j][r], acc[i][j][r + 1]);
        }
      __syncthreads();
      // сплошной вынос: 8 half на нить
      constexpr int PER_ROW = BN / 8;
      for (int idx = tid; idx < slab * PER_ROW; idx += NTHR) {
        const int sr = idx / PER_ROW, sc = (idx % PER_ROW) * 8;
        const int gm = bm + (sr / 16) * MB * 16 + i * 16 + (sr % 16);
        const int gn = bn + sc;
        if (gm < M) *reinterpret_cast<uint4*>(C + (long)gm * N + gn) = *reinterpret_cast<const uint4*>(ep + (long)sr * EPI_LD + sc);
      }
    }
  }
}

// Размер разделяемой: мейнлуп либо плита эпилога, что больше.
template <int BM, int BN, int BK, int WM, int WN, int STAGES, int EPI>
struct SmemBytes {
  static constexpr int kMain = STAGES * (BM + BN) * BK * (int)sizeof(__half);
  static constexpr int kEpi = EPI ? (16 * WM) * (BN + 8) * (int)sizeof(__half) : 0;
  static constexpr int value = kMain > kEpi ? kMain : kEpi;
};

}  // namespace gen
}  // namespace tempo
