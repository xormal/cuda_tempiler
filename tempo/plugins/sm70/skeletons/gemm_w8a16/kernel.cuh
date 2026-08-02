// SPDX-License-Identifier: LicenseRef-TRL-1.0
// Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
// Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
//
// СКЕЛЕТ ТЕМПОЛЯТОРА №2: БАЙТОВЫЙ ВЕС + fp16-АКТИВАЦИИ (W8A16) на тех же HMMA.884.
//
// ЗАЯВКИ ПРО СКОРОСТЬ СЧЁТА ЗДЕСЬ НЕТ И БЫТЬ НЕ МОЖЕТ. У sm_70 нет IMMA; байт, уложенный в мантиссу
// fp16, исполняется ТОЙ ЖЕ инструкцией HMMA.884, что и fp16. Одна инструкция -- одна скорость.
// Байтовый вес покупает ПОЛОСУ И ЁМКОСТЬ, и только там, где полоса связывает (малое M).
// Поэтому соперник здесь -- НЕ cuBLAS, а: (1) наш же fp16-выход того же конвейера, (2) отгруженный
// путь W8A16, (3) наивный int8-вход.
//
// ОСЬ, РАДИ КОТОРОЙ ЭТОТ СКЕЛЕТ И НАПИСАН (вопрос заказчика "две стадии или три"):
//   UNPACK=0 ("A") -- разворачивает ВЕЗУЩАЯ сторона: байты разворачиваются ПЕРЕД записью в
//                     разделяемую, в разделяемой лежит fp16. Трафик разделяемой -- как у fp16
//                     (2 Б на элемент туда, 2 Б обратно), чтение глобальной вдвое меньше,
//                     разворот выполняется РОВНО ОДИН РАЗ на элемент плитки.
//   UNPACK=1 ("B") -- разворачивает СЧИТАЮЩАЯ сторона: в разделяемой лежит УПАКОВАННОЕ (1 Б),
//                     трафик разделяемой 1+1 = 2 Б вместо 4 Б на элемент. РАСПЛАТА: плитку весов
//                     читают ВСЕ варпы, делящие диапазон M, поэтому разворот выполняется WM раз
//                     вместо одного. Экономим трафик -- ДУБЛИРУЕМ работу.
// Третья компоновка ("C", отдельная роль-распаковщик) в этом скелете НЕ представлена: у плотного
// умножения нет разделения варпов на везущих и считающих -- все варпы делают и то, и другое.
// Чтобы измерить "C", нужен скелет с ролями; он в v1 не построен, и это записано честно.
//
// СМЕЩЕНИЕ РАЗВОРОТА -- РАНГ-1, ЦЕНОЙ НОЛЬ. expand_i8x4 отдаёт значения, сдвинутые на 1152:
//     sum_k a_k (1152 + b_k) = sum_k a_k b_k + 1152 * sum_k a_k
// Слагаемое не зависит от столбца, поэтому его место -- ЗАТРАВКА аккумулятора, а не эпилог.
// Суммы строк A считаются один раз отдельным ядром (и переиспользуются всеми матрицами, читающими
// ту же активацию: q/k/v делят одну x, gate/up делят одну x).
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace tempo {
namespace gen8 {

static constexpr uint32_t kMagic4 = 0x64646464u;   // 0x6400|u == 1024+u в каждом полуслове
static constexpr float kBiasSigned = 1152.0f;      // 1024 + 128 (снятие знака XOR-ом)

__device__ __forceinline__ void expand_i8x4(uint32_t w, uint32_t& f01, uint32_t& f23) {
  w ^= 0x80808080u;                                // v+128 сразу на четыре байта
  f01 = __byte_perm(w, kMagic4, 0x4140);
  f23 = __byte_perm(w, kMagic4, 0x4342);
}

__device__ __forceinline__ void hmma884(float (&d)[8], const uint32_t (&a)[2], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 800)
  asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
#endif
}
__device__ __forceinline__ int a_row(int l) { return (l & 3) | ((l & 16) >> 2) | (l & 8); }
__device__ __forceinline__ int b_col(int l) { return (l & 3) | ((l & 16) >> 2) | ((l & 4) << 1); }
__device__ __forceinline__ int acc_row(int l, int r) { return ((l & 1) | ((l & 16) >> 2) | (l & 8)) + 2 * ((r >> 1) & 1); }
__device__ __forceinline__ int acc_col(int l, int r) { return ((l & 2) | ((l & 4) << 1)) + (r & 1) + 4 * ((r >> 2) & 1); }

// Свиззл ПОЛУСЛОВНОЙ плитки (как в скелете fp16, исправленная форма: разряды строки 1 и 3).
template <int BK>
struct SwH {
  static constexpr int NC = BK / 8;
  __device__ static __forceinline__ int x_of(int row) {
    return BK == 32 ? (((row >> 1) & 1) | ((row >> 2) & 2)) : ((row & 3) | ((row >> 1) & 4));
  }
  __device__ static __forceinline__ int off(int row, int c) { return row * BK + ((c ^ x_of(row)) * 8); }
};
// Свиззл БАЙТОВОЙ плитки: чанк = 16 байт, шаг строки BK байт = BK/4 слов.
// При BK=32 строка занимает 8 слов, стартовый банк повторяется каждые 4 строки, а в одной фазе
// сталкиваются строки, отличающиеся на 8 -> в свиззл сворачивается разряд 3.
// ГЕЙТ ПОЙМАЛ ЗДЕСЬ НАСТОЯЩИЙ ДЕФЕКТ (rel=0.994 при BK=64, M=32), и его стоит записать:
// свиззл обязан зависеть ТОЛЬКО от разрядов строки 0..3. Строка фрагмента j равна rowB + j*16, и
// xb выносится из цикла по j -- значит любой разряд >= 4 делает вынесенное значение НЕВЕРНЫМ.
// Первая редакция брала x = (row>>3) & (NC-1): при NC=2 (BK=32) это разряд 3 -- законно; при NC=4
// (BK=64) это разряды 3 и 4, и разряд 4 меняется вместе с j. Ошибка ТИХАЯ: ядро запускается,
// покрытие полное, неверны только значения.
template <int BK>
struct SwB {
  static constexpr int NC = BK / 16;
  __device__ static __forceinline__ int x_of(int row) {
    return NC == 2 ? ((row >> 3) & 1) : (((row >> 1) & 1) | ((row >> 2) & 2));
  }
  __device__ static __forceinline__ int off(int row, int c) { return row * BK + ((c ^ x_of(row)) * 16); }
};

// ------------------------------------------------------- пересылка fp16-операнда A
template <int ROWS, int BK, int NTHR>
struct LoaderH {
  static constexpr int kPerRow = BK / 8;
  static constexpr int kPer = ROWS * BK / 8 / NTHR;
  static_assert(kPer >= 1 && (ROWS * BK / 8) % NTHR == 0, "плитка A не делится на нити");
  uint4 buf[kPer];
  int row[kPer], col[kPer];
  __device__ __forceinline__ void init(int tid) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) { const int x = tid + i * NTHR; row[i] = x / kPerRow; col[i] = (x % kPerRow) * 8; }
  }
  template <bool PRED>
  __device__ __forceinline__ void ldg(const __half* g, int ld, int k0, int rows) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      if (!PRED || row[i] < rows) buf[i] = *reinterpret_cast<const uint4*>(g + (long)row[i] * ld + k0 + col[i]);
      else buf[i] = make_uint4(0, 0, 0, 0);
    }
  }
  __device__ __forceinline__ void sts(__half* s) const {
#pragma unroll
    for (int i = 0; i < kPer; ++i) *reinterpret_cast<uint4*>(s + SwH<BK>::off(row[i], col[i] / 8)) = buf[i];
  }
};

// ------------------------------------------------------- пересылка БАЙТОВОГО операнда B
// Одна 16-байтовая порция = 16 значений = ЧЕТЫРЕ шага инструкции по k.
template <int ROWS, int BK, int NTHR, int UNPACK>
struct LoaderB {
  static constexpr int kPerRow = BK / 16;
  static constexpr int kPer = ROWS * BK / 16 / NTHR;
  static_assert(kPer >= 1 && (ROWS * BK / 16) % NTHR == 0, "байтовая плитка B не делится на нити");
  uint4 buf[kPer];
  int row[kPer], colB[kPer];
  __device__ __forceinline__ void init(int tid) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) { const int x = tid + i * NTHR; row[i] = x / kPerRow; colB[i] = (x % kPerRow) * 16; }
  }
  __device__ __forceinline__ void ldg(const int8_t* g, int ld, int k0) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) buf[i] = *reinterpret_cast<const uint4*>(g + (long)row[i] * ld + k0 + colB[i]);
  }
  __device__ __forceinline__ void sts(void* s) const {
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      if (UNPACK == 0) {                                   // "A": разворот у ВЕЗУЩЕЙ стороны
        uint32_t f[8];
        uint4 w = buf[i];
        expand_i8x4(w.x, f[0], f[1]); expand_i8x4(w.y, f[2], f[3]);
        expand_i8x4(w.z, f[4], f[5]); expand_i8x4(w.w, f[6], f[7]);
        __half* sh = reinterpret_cast<__half*>(s);
        const int c0 = colB[i] / 8;                        // 16 байт = 2 чанка по 8 half
        *reinterpret_cast<uint4*>(sh + SwH<BK>::off(row[i], c0)) = make_uint4(f[0], f[1], f[2], f[3]);
        *reinterpret_cast<uint4*>(sh + SwH<BK>::off(row[i], c0 + 1)) = make_uint4(f[4], f[5], f[6], f[7]);
      } else {                                             // "B": в разделяемой лежит УПАКОВАННОЕ
        int8_t* sb = reinterpret_cast<int8_t*>(s);
        *reinterpret_cast<uint4*>(sb + SwB<BK>::off(row[i], colB[i] / 16)) = buf[i];
      }
    }
  }
};

template <int MB, int NB, int BK, int UNPACK>
__device__ __forceinline__ void warp_tile(float (&acc)[MB][NB][8], const __half* sA, int rowA,
                                          const void* sB, int rowB) {
  constexpr int NC = BK / 8;
  const __half* pA = sA + rowA * BK;
  const int xa = SwH<BK>::x_of(rowA);
  uint4 fa[2][MB], fbh[2][NB];
  uint2 fbb[2][NB];
  const __half* pBh = reinterpret_cast<const __half*>(sB) + rowB * BK;
  const int8_t* pBb = reinterpret_cast<const int8_t*>(sB) + rowB * BK;
  const int xb = UNPACK ? SwB<BK>::x_of(rowB) : SwH<BK>::x_of(rowB);
#define TEMPO_LDF(SLOT, CH)                                                                        \
  {                                                                                                \
    _Pragma("unroll") for (int i = 0; i < MB; ++i)                                                 \
        fa[SLOT][i] = *reinterpret_cast<const uint4*>(pA + (long)(i * 16) * BK + (((CH) ^ xa) * 8));\
    if (UNPACK == 0) {                                                                             \
      _Pragma("unroll") for (int j = 0; j < NB; ++j)                                               \
          fbh[SLOT][j] = *reinterpret_cast<const uint4*>(pBh + (long)(j * 16) * BK + (((CH) ^ xb) * 8)); \
    } else {                                                                                       \
      _Pragma("unroll") for (int j = 0; j < NB; ++j)                                               \
          fbb[SLOT][j] = *reinterpret_cast<const uint2*>(                                          \
              pBb + (long)(j * 16) * BK + ((((CH) >> 1) ^ xb) * 16) + (((CH) & 1) * 8));            \
    }                                                                                              \
  }
  TEMPO_LDF(0, 0)
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    const int b = c & 1;
    if (c + 1 < NC) TEMPO_LDF(b ^ 1, c + 1)
    uint32_t alo[MB][2], ahi[MB][2], blo[NB][2], bhi[NB][2];
#pragma unroll
    for (int i = 0; i < MB; ++i) { alo[i][0] = fa[b][i].x; alo[i][1] = fa[b][i].y; ahi[i][0] = fa[b][i].z; ahi[i][1] = fa[b][i].w; }
    if (UNPACK == 0) {
#pragma unroll
      for (int j = 0; j < NB; ++j) { blo[j][0] = fbh[b][j].x; blo[j][1] = fbh[b][j].y; bhi[j][0] = fbh[b][j].z; bhi[j][1] = fbh[b][j].w; }
    } else {
#pragma unroll
      for (int j = 0; j < NB; ++j) { expand_i8x4(fbb[b][j].x, blo[j][0], blo[j][1]); expand_i8x4(fbb[b][j].y, bhi[j][0], bhi[j][1]); }
    }
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j) hmma884(acc[i][j], alo[i], blo[j]);
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j) hmma884(acc[i][j], ahi[i], bhi[j]);
  }
#undef TEMPO_LDF
}

template <int GROUP>
__device__ __forceinline__ void block_swizzle(int bid, int nm, int nn, int& bm, int& bn) {
  if (GROUP <= 1) { bm = bid / nn; bn = bid % nn; return; }
  const int per = GROUP * nn, grp = bid / per, first = grp * GROUP;
  const int gs = min(GROUP, nm - first);
  bm = first + (bid % per) % gs;
  bn = ((bid % per) / gs);
}

// rowsum[m] = sum_k A[m][k] -- считается один раз, переиспользуется всеми матрицами на той же A.
__global__ void k_rowsum(const __half* A, float* s, int M, int K) {
  const int m = blockIdx.x;
  if (m >= M) return;
  float v = 0;
  for (int k = threadIdx.x; k < K; k += blockDim.x) v += __half2float(A[(long)m * K + k]);
  __shared__ float r[32];
  for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  if ((threadIdx.x & 31) == 0) r[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x == 0) { float t = 0; for (int i = 0; i < (int)blockDim.x / 32; ++i) t += r[i]; s[m] = t; }
}

template <int BM, int BN, int BK, int WM, int WN, int UNPACK, int GROUP, bool PRED, int MINB>
__global__ __launch_bounds__(WM* WN * 32, MINB) void k_gemm8(const __half* __restrict__ A,
                                                             const int8_t* __restrict__ B,
                                                             const float* __restrict__ rowsum,
                                                             __half* __restrict__ C,
                                                             int M, int N, int K) {
  constexpr int NTHR = WM * WN * 32;
  constexpr int MB = BM / (16 * WM), NB = BN / (16 * WN);
  constexpr int SA = 2 * BM * BK;                       // half
  constexpr int SBSZ = UNPACK ? (2 * BN * BK) : (2 * BN * BK * 2);   // байт
  extern __shared__ __half smem[];
  __half* sA = smem;
  void* sB = reinterpret_cast<void*>(reinterpret_cast<char*>(smem) + SA * 2);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int bmi, bni;
  block_swizzle<GROUP>(blockIdx.x, (M + BM - 1) / BM, N / BN, bmi, bni);
  const int bm = bmi * BM, bn = bni * BN;
  const int rows = PRED ? min(BM, M - bm) : BM;
  LoaderH<BM, BK, NTHR> la; la.init(tid);
  LoaderB<BN, BK, NTHR, UNPACK> lb; lb.init(tid);
  const __half* gA = A + (long)bm * K;
  const int8_t* gB = B + (long)bn * K;
  const int NT = K / BK;
  const int bStride = UNPACK ? BN * BK : BN * BK * 2;     // байт на один буфер B

  la.template ldg<PRED>(gA, K, 0, rows); lb.ldg(gB, K, 0);
  la.sts(sA); lb.sts(sB);
  __syncthreads();
  if (NT > 1) { la.template ldg<PRED>(gA, K, BK, rows); lb.ldg(gB, K, BK); }

  const int wm = warp / WN, wn = warp % WN;
  const int rowA = wm * MB * 16 + a_row(lane), rowB = wn * NB * 16 + b_col(lane);
  float acc[MB][NB][8];
  // ЗАТРАВКА РАНГ-1: снимает смещение разворота 1152 ценой ноль (иначе -- операция на элемент).
  const int r0 = bm + wm * MB * 16;
#pragma unroll
  for (int i = 0; i < MB; ++i) {
    float s0 = 0.f, s1 = 0.f;
    const int m0 = r0 + i * 16 + ((lane & 1) | ((lane & 16) >> 2) | (lane & 8));
    if (m0 < M) s0 = rowsum[m0];
    if (m0 + 2 < M) s1 = rowsum[m0 + 2];
#pragma unroll
    for (int j = 0; j < NB; ++j)
#pragma unroll
      for (int rg = 0; rg < 8; ++rg) acc[i][j][rg] = -kBiasSigned * (((rg >> 1) & 1) ? s1 : s0);
  }
  int cur = 0;
#pragma unroll 1
  for (int t = 0; t < NT; ++t) {
    warp_tile<MB, NB, BK, UNPACK>(acc, sA + cur * BM * BK, rowA,
                                  reinterpret_cast<char*>(sB) + cur * bStride, rowB);
    if (t + 1 < NT) {
      la.sts(sA + (cur ^ 1) * BM * BK);
      lb.sts(reinterpret_cast<char*>(sB) + (cur ^ 1) * bStride);
      __syncthreads();
      if (t + 2 < NT) { la.template ldg<PRED>(gA, K, (t + 2) * BK, rows); lb.ldg(gB, K, (t + 2) * BK); }
      cur ^= 1;
    }
  }
  const int c0 = bn + wn * NB * 16;
#pragma unroll
  for (int i = 0; i < MB; ++i)
#pragma unroll
    for (int j = 0; j < NB; ++j)
#pragma unroll
      for (int r = 0; r < 8; r += 2) {
        const int m = r0 + i * 16 + acc_row(lane, r), n = c0 + j * 16 + acc_col(lane, r);
        if (m < M && n + 1 < N) *reinterpret_cast<__half2*>(C + (long)m * N + n) = __floats2half2_rn(acc[i][j][r], acc[i][j][r + 1]);
        else if (m < M && n < N) C[(long)m * N + n] = __float2half(acc[i][j][r]);
      }
}

template <int BM, int BN, int BK, int UNPACK>
struct Smem8 { static constexpr int value = 2 * BM * BK * 2 + (UNPACK ? 2 * BN * BK : 2 * BN * BK * 2); };

}  // namespace gen8
}  // namespace tempo
