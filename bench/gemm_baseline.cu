// ЧЕСТНЫЙ ЭТАЛОН ЛИНЕЙНОЙ ЧАСТИ: три плеча на одних и тех же боевых формах Gemma-4-12B.
//
// ТРИ РАЗНЫХ СРАВНЕНИЯ, КОТОРЫЕ НЕЛЬЗЯ ПУТАТЬ (ради этого файл и написан):
//   (1) cuBLAS-fp16      -- ЖИВОЙ СОПЕРНИК. Абсолютная планка продукта.
//   (2) НАИВНЫЙ вход     -- ../inputs/naive_gemm_fp16.cu, собранный тем же nvcc -O3.
//                           Собственная метрика компилятора: «во сколько раз выход обгоняет вход».
//   (3) рукописный мейнлуп Volta -- то, чем заявка «0.67-0.82x к cuBLAS» была замерена
//                           (боевое дерево, tools/volta_gemm_bench.cu). Здесь она проверяется НА
//                           БОЕВЫХ ФОРМАХ, а не на квадратах 4096^3, на которых была получена.
//
// ЧТО ИСПРАВЛЕНО ОТНОСИТЕЛЬНО ИСХОДНОЙ ЗАЯВКИ. volta_gemm_bench требует M % BM == 0 и мерит только
// квадратные/крупные формы. Боевые M это 1, 8, 32 -- то есть ДЕКОД, где cuBLAS слабее всего. Поэтому
// здесь: (а) плитка предицирована по M (грид с округлением вверх, загрузка вне M даёт ноль);
// (б) добавлены УЗКИЕ по M конфигурации (BM = 16/32/64), которых в исходном стенде не было вовсе --
// без них сравнение при M<=32 было бы соломенным чучелом со стороны НАШЕГО плеча.
// Цена предикации мерится отдельно: на формах, где M % BM == 0, гоняются оба варианта.
//
// ФЛОПы ВСЕГДА ПОЛЕЗНЫЕ: 2*M*N*K. Если плитка считает 128 строк ради одной, штраф гранулярности
// виден в числе, а не спрятан в знаменателе.
//
// Сборка:
//   nvcc -O3 -std=c++17 -arch=sm_70 -lcublas \
//        -I <боевое>/fa2_src/fmha_kernel/gemm -o gemm_baseline gemm_baseline.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <array>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "volta_hmma.h"

#define NAIVE_GEMM_AS_HEADER 1
// НЕДОДЕЛАННЫЙ ХВОСТ ПЕРЕНОСА: наивный вход переехал в kernels/naive/, а эта строка осталась
// указывать на inputs/, которого в дереве НЕТ. Следствие было дороже опечатки: этот бинарь
// даёт колонки «было» и «xвход», то есть медиану x102 из README.md и report.md ПОЛУЧИТЬ ИЗ
// ДЕРЕВА БЫЛО НЕЛЬЗЯ -- он не собирался вовсе.
#include "../kernels/naive/gemm_fp16.cu"

using namespace fa2_sm70;
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)
#define CB(x) do { cublasStatus_t e=(x); if(e){printf("cuBLAS %s @%d: %d\n",#x,__LINE__,(int)e);exit(1);} } while(0)

// ------------------------------------------------------------------ данные и гейт корректности
// Псевдослучайное заполнение НА УСТРОЙСТВЕ. Постоянный memset (как в исходном стенде) делает гейт
// корректности слепым: ядро, считающее часть своей плитки, на постоянных данных даёт правдоподобный
// ответ. Хеш даёт разные значения в каждой ячейке, поэтому недосчёт виден.
__global__ void k_fill(__half* p, long n, unsigned seed) {
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    unsigned h = (unsigned)i * 2654435761u + seed;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    p[i] = __float2half(((float)(h & 0xFFFF) / 65535.f - 0.5f) * 0.25f);
  }
}
__global__ void k_rel(const __half* a, const __half* b, long n, double* out) {
  double d = 0, r = 0;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    const double x = __half2float(a[i]), y = __half2float(b[i]);
    d += (x - y) * (x - y); r += y * y;
  }
  atomicAdd(out, d); atomicAdd(out + 1, r);
}
static double rel_l2(const __half* a, const __half* b, long n) {
  double* d; CK(cudaMalloc(&d, 2 * sizeof(double))); CK(cudaMemset(d, 0, 2 * sizeof(double)));
  k_rel<<<512, 256>>>(a, b, n, d); CK(cudaDeviceSynchronize());
  double h[2]; CK(cudaMemcpy(h, d, 2 * sizeof(double), cudaMemcpyDeviceToHost)); cudaFree(d);
  return h[1] > 0 ? sqrt(h[0] / h[1]) : (h[0] > 0 ? 1.0 : 0.0);
}

// ------------------------------------------------------------------ рукописный мейнлуп (боевое дерево)
// Одна 16-байтовая порция глобальной памяти на поток. PRED: строки за границей M дают ноль.
template <int ROWS, int BK, int NTHR>
struct TileLoader {
  static constexpr int kChunks = ROWS * BK / 8;
  static constexpr int kPer = kChunks / NTHR;
  static constexpr int kPerRow = BK / 8;
  static_assert(kPer >= 1, "нитей больше, чем 16-байтовых порций в плитке");
  uint4 buf[kPer];
  int row[kPer], col[kPer];
  __device__ __forceinline__ void init(int tid) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) { const int idx = tid + i * NTHR; row[i] = idx / kPerRow; col[i] = (idx % kPerRow) * 8; }
  }
  template <bool PRED>
  __device__ __forceinline__ void ldg(const __half* g, int ld, int k0, int rows) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      if (!PRED || row[i] < rows) buf[i] = *reinterpret_cast<const uint4*>(g + (long)row[i] * ld + k0 + col[i]);
      else buf[i] = make_uint4(0, 0, 0, 0);
    }
  }
  template <int BK2>
  __device__ __forceinline__ void sts_sw(__half* s) const {
#pragma unroll
    for (int i = 0; i < kPer; ++i)
      *reinterpret_cast<uint4*>(s + SwizzleTile<BK2>::off(row[i], col[i] / 8)) = buf[i];
  }
};

// Переупорядочение блоков под L2: подряд идущие блоки образуют компактный столбец GROUP x всё-N.
template <int GROUP>
__device__ __forceinline__ void swizzle(int bid, int nm, int nn, int& bm, int& bn) {
  const int per = GROUP * nn;
  const int grp = bid / per, first = grp * GROUP;
  const int gs = min(GROUP, nm - first);
  bm = first + (bid % per) % gs;
  bn = ((bid % per) / gs);
}

template <int BM, int BN, int BK, int MB, int NB, int WM, int WN, bool PRED>
__global__ __launch_bounds__(WM* WN * 32) void k_gemm_sw(const __half* __restrict__ A,
                                                         const __half* __restrict__ B,
                                                         __half* __restrict__ C, int M, int N, int K) {
  constexpr int NTHR = WM * WN * 32;
  extern __shared__ __half smem[];
  __half* sA = smem;
  __half* sB = smem + 2 * BM * BK;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int bmi, bni; swizzle<8>(blockIdx.x, (M + BM - 1) / BM, N / BN, bmi, bni);
  const int bm = bmi * BM, bn = bni * BN;
  const int rows = PRED ? min(BM, M - bm) : BM;
  TileLoader<BM, BK, NTHR> la; la.init(tid);
  TileLoader<BN, BK, NTHR> lb; lb.init(tid);
  const __half* gA = A + (long)bm * K;
  const __half* gB = B + (long)bn * K;
  la.template ldg<PRED>(gA, K, 0, rows); lb.template ldg<false>(gB, K, 0, BN);
  la.template sts_sw<BK>(sA); lb.template sts_sw<BK>(sB);
  __syncthreads();
  if (K > BK) { la.template ldg<PRED>(gA, K, BK, rows); lb.template ldg<false>(gB, K, BK, BN); }
  const int wm = warp / WN, wn = warp % WN;
  const int rowA = wm * MB * 16 + hmma_a_row(lane), rowB = wn * NB * 16 + hmma_b_col(lane);
  float acc[MB][NB][8];
  hmma_clear<MB, NB>(acc);
  int cur = 0;
  const int NT = K / BK;
  for (int kt = 0; kt < NT; ++kt) {
    hmma_warp_tile_sw<MB, NB, BK>(acc, sA + cur * BM * BK, rowA, sB + cur * BN * BK, rowB);
    if (kt + 1 < NT) {
      la.template sts_sw<BK>(sA + (cur ^ 1) * BM * BK);
      lb.template sts_sw<BK>(sB + (cur ^ 1) * BN * BK);
      __syncthreads();
      if (kt + 2 < NT) { la.template ldg<PRED>(gA, K, (kt + 2) * BK, rows); lb.template ldg<false>(gB, K, (kt + 2) * BK, BN); }
      cur ^= 1;
    }
  }
  const int r0 = bm + wm * MB * 16, c0 = bn + wn * NB * 16;
#pragma unroll
  for (int i = 0; i < MB; ++i)
#pragma unroll
    for (int j = 0; j < NB; ++j)
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int m = r0 + i * 16 + hmma_acc_row(lane, r), n = c0 + j * 16 + hmma_acc_col(lane, r);
        if (m < M && n < N) C[(long)m * N + n] = __float2half(acc[i][j][r]);
      }
}

struct Timer {
  cudaEvent_t e0, e1;
  Timer() { cudaEventCreate(&e0); cudaEventCreate(&e1); }
  ~Timer() { cudaEventDestroy(e0); cudaEventDestroy(e1); }
  // Прогрев ВНЕ секундомера обязателен: просадка по мощности наступает не мгновенно, и на холодном
  // окне карта успевает отработать на ещё не просевшей частоте. Замерено (bench/data_power.cu):
  // установившаяся частота на тяжёлых формах 1297-1387 МГц против 1530 на первых миллисекундах,
  // то есть до 19 % завышения, если мерить короткими холодными окнами.
  template <class F> double run(F&& f, int iters) {
    for (int i = 0; i < (iters + 2) / 3; ++i) f();
    CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) f();
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); return ms / iters;
  }
};

template <int BM, int BN, int BK, int MB, int NB, int WM, int WN, bool PRED>
struct VoltaCfg {
  static constexpr size_t smem = (size_t)2 * (BM + BN) * BK * sizeof(__half);
  static void launch(const __half* A, const __half* B, __half* C, int M, int N, int K) {
    auto kern = k_gemm_sw<BM, BN, BK, MB, NB, WM, WN, PRED>;
    static bool once = false;
    if (!once) { CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); once = true; }
    const int nm = PRED ? (M + BM - 1) / BM : M / BM;
    kern<<<dim3(nm * (N / BN)), dim3(WM * WN * 32), smem>>>(A, B, C, M, N, K);
  }
  static bool ok(int M, int N, int K) {
    return (N % BN == 0) && (K % BK == 0) && (PRED || (M % BM == 0)) && smem <= 96 * 1024;
  }
};

// ------------------------------------------------------------------ описание плеча
struct Arm {
  const char* tag;
  bool (*ok)(int, int, int);
  void (*launch)(const __half*, const __half*, __half*, int, int, int);
};

#define VCFG(name, BM, BN, BK, MB, NB, WM, WN, PRED) \
  Arm{name, &VoltaCfg<BM, BN, BK, MB, NB, WM, WN, PRED>::ok, &VoltaCfg<BM, BN, BK, MB, NB, WM, WN, PRED>::launch}

int main(int argc, char** argv) {
  int dev = 0, rounds = 5;
  bool do_naive = true, do_volta = true;
  double naive_budget_ms = 4000.0;   // потолок на ОДИН запуск наивного ядра
  std::vector<int> Ms;
  std::vector<std::array<int, 2>> shapes;   // {K, N}
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--dev")) dev = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--rounds")) rounds = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--no-naive")) do_naive = false;
    else if (!strcmp(argv[i], "--no-volta")) do_volta = false;
    else if (!strcmp(argv[i], "--naive-budget")) naive_budget_ms = atof(argv[++i]);
    else if (!strcmp(argv[i], "--m")) { char* s = argv[++i]; for (char* t = strtok(s, ","); t; t = strtok(nullptr, ",")) Ms.push_back(atoi(t)); }
    else if (!strcmp(argv[i], "--shape")) { int K, N; sscanf(argv[++i], "%d:%d", &K, &N); shapes.push_back({K, N}); }
  }
  if (Ms.empty()) Ms = {1, 8, 32, 128, 512, 2048, 8192};
  if (shapes.empty()) shapes = {{3840, 4096}, {3840, 2048}, {4096, 3840}, {3840, 15360}, {15360, 3840}};
  CK(cudaSetDevice(dev));
  cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));

  // Лестница конфигураций рукописного мейнлупа. Узкие по M (16/32/64) добавлены здесь: без них
  // при M<=32 наше плечо теряло бы 4-128x на гранулярности плитки, и сравнение было бы нечестным
  // В НАШУ НЕВЫГОДУ. Порядок в списке роли не играет -- берётся лучшая ПРОШЕДШАЯ ГЕЙТ.
  const Arm volta_cfgs[] = {
      VCFG("v16x128k64/4w",  16, 128, 64, 1, 2, 1, 4, true),
      VCFG("v32x128k64/8w",  32, 128, 64, 1, 2, 2, 4, true),
      VCFG("v64x128k32/8w",  64, 128, 32, 2, 2, 2, 4, true),
      VCFG("v64x128k64/8w",  64, 128, 64, 2, 2, 2, 4, true),
      VCFG("v128x128k32/4w", 128, 128, 32, 4, 4, 2, 2, true),
      VCFG("v128x128k32/8w", 128, 128, 32, 2, 4, 4, 2, true),
      VCFG("v128x128k64/4w", 128, 128, 64, 4, 4, 2, 2, true),
      VCFG("v256x128k32/8w", 256, 128, 32, 4, 4, 4, 2, true),
  };
  // Тот же лучший размер БЕЗ предикации -- цена предикации, а не догадка о ней.
  const Arm volta_nopred[] = {
      VCFG("np128x128k32/4w", 128, 128, 32, 4, 4, 2, 2, false),
      VCFG("np128x128k64/4w", 128, 128, 64, 4, 4, 2, 2, false),
      VCFG("np256x128k32/8w", 256, 128, 32, 4, 4, 4, 2, false),
  };

  for (auto& s : shapes) {
    const int K = s[0], N = s[1];
    for (int M : Ms) {
      __half *dA, *dB, *dC, *dR;
      CK(cudaMalloc(&dA, (size_t)M * K * 2));
      CK(cudaMalloc(&dB, (size_t)N * K * 2));
      CK(cudaMalloc(&dC, (size_t)M * N * 2));
      CK(cudaMalloc(&dR, (size_t)M * N * 2));
      k_fill<<<512, 256>>>(dA, (long)M * K, 11u);
      k_fill<<<512, 256>>>(dB, (long)N * K, 77u);
      CK(cudaDeviceSynchronize());
      const double flop = 2.0 * M * N * K;
      Timer T;

      // --- (1) cuBLAS-fp16, выход fp16: ровно то, что делает F.linear -------------------------
      // alpha/beta ОБЯЗАНЫ быть float: тип задаётся computeType (CUBLAS_COMPUTE_32F), а НЕ типом
      // данных. Полуслова здесь читаются как float и дают alpha=0 -> эталон из нулей, и тогда ВСЕ
      // гейты корректности рапортуют rel=1.0 разом. Симптом «все плечи неверны сразу» = неверен
      // эталон, а не плечи.
      const float al = 1.f, be = 0.f;
      auto cublas_run = [&] {
        CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &al, dB, CUDA_R_16F, K,
                        dA, CUDA_R_16F, K, &be, dR, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      };
      cublas_run(); CK(cudaDeviceSynchronize());          // dR = эталон значений

      // подбор числа повторов: ~120 мс на плечо
      double t0 = T.run(cublas_run, 3);
      int it_cb = std::max(3, std::min(800, (int)(400.0 / std::max(t0, 1e-3))));

      // --- (3) рукописный мейнлуп: гейт корректности РАНЬШЕ секундомера ------------------------
      struct Cand { const char* tag; double ms; double rel; };
      std::vector<Cand> vres;
      if (do_volta) {
        for (const Arm& a : volta_cfgs) {
          if (!a.ok(M, N, K)) continue;
          CK(cudaMemset(dC, 0x7f, (size_t)M * N * 2));
          a.launch(dA, dB, dC, M, N, K);
          if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); continue; }
          const double rel = rel_l2(dC, dR, (long)M * N);
          if (!(rel < 5e-3)) { printf("SKIP %s M%d N%d K%d rel=%.2e\n", a.tag, M, N, K, rel); continue; }
          auto run = [&] { a.launch(dA, dB, dC, M, N, K); };
          const double tw = T.run(run, 3);
          const int it = std::max(3, std::min(800, (int)(400.0 / std::max(tw, 1e-3))));
          vres.push_back({a.tag, T.run(run, it), rel});
        }
      }
      // цена предикации: тот же размер плитки без неё
      std::vector<Cand> npres;
      if (do_volta) {
        for (const Arm& a : volta_nopred) {
          if (!a.ok(M, N, K)) continue;
          CK(cudaMemset(dC, 0x7f, (size_t)M * N * 2));
          a.launch(dA, dB, dC, M, N, K);
          if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); continue; }
          const double rel = rel_l2(dC, dR, (long)M * N);
          if (!(rel < 5e-3)) continue;
          auto run = [&] { a.launch(dA, dB, dC, M, N, K); };
          const double tw = T.run(run, 3);
          const int it = std::max(3, std::min(800, (int)(400.0 / std::max(tw, 1e-3))));
          npres.push_back({a.tag, T.run(run, it), rel});
        }
      }
      const Cand* best = nullptr;
      for (auto& c : vres) if (!best || c.ms < best->ms) best = &c;
      const Cand* bestnp = nullptr;
      for (auto& c : npres) if (!bestnp || c.ms < bestnp->ms) bestnp = &c;

      // --- (2) наивный вход --------------------------------------------------------------------
      double naive_ms = -1, naive_rel = -1;
      if (do_naive) {
        CK(cudaMemset(dC, 0x7f, (size_t)M * N * 2));
        auto run = [&] { naive_gemm_fp16_launch(dA, dB, dC, M, N, K); };
        run(); CK(cudaDeviceSynchronize());
        naive_rel = rel_l2(dC, dR, (long)M * N);
        const double t1 = T.run(run, 1);
        if (t1 <= naive_budget_ms) {
          const int it = std::max(1, std::min(200, (int)(400.0 / std::max(t1, 1e-3))));
          naive_ms = T.run(run, it);
        } else {
          naive_ms = t1;   // одна итерация -- всё, что можно себе позволить
        }
      }

      // --- парные раунды: плечи ЧЕРЕДУЮТСЯ ВНУТРИ раунда, медиана ОТНОШЕНИЙ -------------------
      std::vector<double> r_volta, r_naive, cb_ms;
      for (int r = 0; r < rounds; ++r) {
        const double c = T.run(cublas_run, it_cb);
        cb_ms.push_back(c);
        if (best) {
          const Arm* a = nullptr;
          for (const Arm& x : volta_cfgs) if (!strcmp(x.tag, best->tag)) a = &x;
          auto run = [&] { a->launch(dA, dB, dC, M, N, K); };
          const double tw = T.run(run, std::max(3, std::min(800, (int)(400.0 / std::max(best->ms, 1e-3)))));
          r_volta.push_back(c / tw);            // >1 = наше плечо БЫСТРЕЕ cuBLAS
        }
        if (do_naive && naive_ms > 0 && naive_ms < naive_budget_ms) {
          auto run = [&] { naive_gemm_fp16_launch(dA, dB, dC, M, N, K); };
          const double tn = T.run(run, std::max(1, std::min(200, (int)(400.0 / std::max(naive_ms, 1e-3)))));
          r_naive.push_back(c / tn);
        }
      }
      auto med = [](std::vector<double> v) -> double {
        if (v.empty()) return -1;
        std::sort(v.begin(), v.end());
        return v.size() & 1 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
      };
      const double cb = med(cb_ms);
      printf("JSON {\"K\":%d,\"N\":%d,\"M\":%d,\"flop\":%.0f,"
             "\"cublas_ms\":%.6f,\"cublas_tflops\":%.4f,"
             "\"volta_cfg\":\"%s\",\"volta_ms\":%.6f,\"volta_tflops\":%.4f,\"volta_rel\":%.3e,\"volta_ratio_med\":%.4f,"
             "\"nopred_cfg\":\"%s\",\"nopred_ms\":%.6f,"
             "\"naive_ms\":%.6f,\"naive_tflops\":%.4f,\"naive_rel\":%.3e,\"naive_ratio_med\":%.4f}\n",
             K, N, M, flop,
             cb, flop / (cb * 1e-3) / 1e12,
             best ? best->tag : "-", best ? best->ms : -1.0, best ? flop / (best->ms * 1e-3) / 1e12 : -1.0,
             best ? best->rel : -1.0, med(r_volta),
             bestnp ? bestnp->tag : "-", bestnp ? bestnp->ms : -1.0,
             naive_ms, naive_ms > 0 ? flop / (naive_ms * 1e-3) / 1e12 : -1.0, naive_rel, med(r_naive));
      fflush(stdout);
      cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dR);
    }
  }
  cublasDestroy(h);
  return 0;
}
