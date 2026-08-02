// SPDX-License-Identifier: LicenseRef-TRL-1.0
// Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
// Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
//
// ЗАМЕРНЫЙ СТЕНД ОДНОЙ СТАДИИ КОНВЕЙЕРА (P6..P9): собрать -> ГЕЙТ КОРРЕКТНОСТИ -> секундомер ->
// выбрать. Список гиперформ приходит из configs.inc, который порождает отсекатель (P3/P4).
//
// ДИСЦИПЛИНА, БЕЗ КОТОРОЙ ЧИСЛО НЕ ЧИСЛО:
//  * гейт корректности РАНЬШЕ секундомера, и он двойной: (а) относительная L2 к cuBLAS,
//    (б) ПОКРЫТИЕ -- ни одна ячейка не осталась с ядовитым значением. Заливка выхода 0x7f7f даёт
//    непокрытой ячейке ~65472, и одна такая на 31 млн поднимает rel до ~12. Сверка ЗНАЧЕНИЙ без
//    покрытия уже один раз пропустила ядро, считавшее ЧЕТВЕРТЬ плитки.
//  * ломка симметрии: третий, независимый от cuBLAS, эталон на РАЗРЕЖЕННОЙ выборке ячеек, считанный
//    в fp64. Две сверки, читающие один и тот же промежуточный буфер, соглашаются, будучи обе неверны.
//  * парные отношения: плечи ЧЕРЕДУЮТСЯ ВНУТРИ раунда, берётся медиана ОТНОШЕНИЙ (не отношение медиан).
//  * ФЛОПы всегда полезные 2*M*N*K: штраф гранулярности плитки виден в числе, а не спрятан.
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

#include "kernel.cuh"

using namespace tempo::gen;
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)
#define CB(x) do { cublasStatus_t e=(x); if(e){printf("cuBLAS %s @%d: %d\n",#x,__LINE__,(int)e);exit(1);} } while(0)

// ------------------------------------------------------------------ данные, эталоны, гейт
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
// ПОКРЫТИЕ: сколько ячеек так и остались ядовитыми (0x7f7f).
__global__ void k_poison(const __half* a, long n, unsigned long long* out) {
  const unsigned short P = 0x7f7f;
  unsigned long long c = 0;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
    if (*reinterpret_cast<const unsigned short*>(a + i) == P) ++c;
  atomicAdd(out, c);
}
// НЕЗАВИСИМЫЙ эталон: NS разреженных ячеек, накопление в double, БЕЗ участия cuBLAS.
__global__ void k_spot(const __half* A, const __half* B, const __half* C, int M, int N, int K,
                       int NS, double* out) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= NS) return;
  unsigned h = (unsigned)s * 2654435761u + 7u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  const int m = (int)(h % (unsigned)M);
  h ^= h >> 11; h *= 3266489917u; h ^= h >> 15;
  const int n = (int)(h % (unsigned)N);
  double acc = 0;
  for (int k = 0; k < K; ++k) acc += (double)__half2float(A[(long)m * K + k]) * (double)__half2float(B[(long)n * K + k]);
  const double got = __half2float(C[(long)m * N + n]);
  atomicAdd(out, (got - acc) * (got - acc));
  atomicAdd(out + 1, acc * acc);
}
static double rel_l2(const __half* a, const __half* b, long n) {
  double* d; CK(cudaMalloc(&d, 2 * sizeof(double))); CK(cudaMemset(d, 0, 2 * sizeof(double)));
  k_rel<<<512, 256>>>(a, b, n, d); CK(cudaDeviceSynchronize());
  double h[2]; CK(cudaMemcpy(h, d, 2 * sizeof(double), cudaMemcpyDeviceToHost)); cudaFree(d);
  return h[1] > 0 ? sqrt(h[0] / h[1]) : (h[0] > 0 ? 1.0 : 0.0);
}
static unsigned long long poison_left(const __half* a, long n) {
  unsigned long long* d; CK(cudaMalloc(&d, 8)); CK(cudaMemset(d, 0, 8));
  k_poison<<<512, 256>>>(a, n, d); CK(cudaDeviceSynchronize());
  unsigned long long h; CK(cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost)); cudaFree(d);
  return h;
}
static double spot_rel(const __half* A, const __half* B, const __half* C, int M, int N, int K, int NS) {
  double* d; CK(cudaMalloc(&d, 2 * sizeof(double))); CK(cudaMemset(d, 0, 2 * sizeof(double)));
  k_spot<<<(NS + 63) / 64, 64>>>(A, B, C, M, N, K, NS, d); CK(cudaDeviceSynchronize());
  double h[2]; CK(cudaMemcpy(h, d, 2 * sizeof(double), cudaMemcpyDeviceToHost)); cudaFree(d);
  return h[1] > 0 ? sqrt(h[0] / h[1]) : (h[0] > 0 ? 1.0 : 0.0);
}

struct Timer {
  cudaEvent_t e0, e1;
  Timer() { cudaEventCreate(&e0); cudaEventCreate(&e1); }
  ~Timer() { cudaEventDestroy(e0); cudaEventDestroy(e1); }
  template <class F> double run(F&& f, int iters) {
    for (int i = 0; i < (iters + 2) / 3; ++i) f();
    CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) f();
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); return ms / iters;
  }
};

// ------------------------------------------------------------------ обёртка гиперформы
template <int BM, int BN, int BK, int WM, int WN, int STAGES, int GSTAGE, int FPREF,
          int GROUP, int EPI, int SWZ, bool PRED, int MINB>
struct Cfg {
  static constexpr int smem = SmemBytes<BM, BN, BK, WM, WN, STAGES, EPI>::value;
  static void launch(const __half* A, const __half* B, __half* C, int M, int N, int K) {
    auto kern = k_gemm<BM, BN, BK, WM, WN, STAGES, GSTAGE, FPREF, GROUP, EPI, SWZ, PRED, MINB>;
    static bool once = false;
    if (!once) { CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); once = true; }
    const int nm = PRED ? (M + BM - 1) / BM : M / BM;
    kern<<<dim3(nm * (N / BN)), dim3(WM * WN * 32), smem>>>(A, B, C, M, N, K);
  }
  static bool ok(int M, int N, int K) {
    return (N % BN == 0) && (K % BK == 0) && (PRED || (M % BM == 0)) && smem <= 96 * 1024 && K / BK >= STAGES;
  }
  static int regs() {
    cudaFuncAttributes at;
    cudaFuncGetAttributes(&at, k_gemm<BM, BN, BK, WM, WN, STAGES, GSTAGE, FPREF, GROUP, EPI, SWZ, PRED, MINB>);
    return at.numRegs;
  }
  static int frame() {
    cudaFuncAttributes at;
    cudaFuncGetAttributes(&at, k_gemm<BM, BN, BK, WM, WN, STAGES, GSTAGE, FPREF, GROUP, EPI, SWZ, PRED, MINB>);
    return (int)at.localSizeBytes;
  }
};

struct Arm {
  const char* tag;
  bool (*ok)(int, int, int);
  void (*launch)(const __half*, const __half*, __half*, int, int, int);
  int (*regs)();
  int (*frame)();
  int smem;
};

#define CFG(tag, BM, BN, BK, WM, WN, ST, GS, FP, GR, EP, SW, PR, MB)                    \
  Arm{tag, &Cfg<BM, BN, BK, WM, WN, ST, GS, FP, GR, EP, SW, PR, MB>::ok,                \
      &Cfg<BM, BN, BK, WM, WN, ST, GS, FP, GR, EP, SW, PR, MB>::launch,                 \
      &Cfg<BM, BN, BK, WM, WN, ST, GS, FP, GR, EP, SW, PR, MB>::regs,                   \
      &Cfg<BM, BN, BK, WM, WN, ST, GS, FP, GR, EP, SW, PR, MB>::frame,                  \
      Cfg<BM, BN, BK, WM, WN, ST, GS, FP, GR, EP, SW, PR, MB>::smem}

static const Arm kArms[] = {
#include "configs.inc"
};
static constexpr int kNArms = (int)(sizeof(kArms) / sizeof(kArms[0]));

int main(int argc, char** argv) {
  int dev = 0, rounds = 5, nspot = 2048;
  double gate = 3e-3;
  std::vector<int> Ms;
  std::vector<std::array<int, 2>> shapes;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--dev")) dev = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--rounds")) rounds = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--gate")) gate = atof(argv[++i]);
    else if (!strcmp(argv[i], "--spot")) nspot = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--m")) { char* s = argv[++i]; for (char* t = strtok(s, ","); t; t = strtok(nullptr, ",")) Ms.push_back(atoi(t)); }
    else if (!strcmp(argv[i], "--shape")) { int K, N; sscanf(argv[++i], "%d:%d", &K, &N); shapes.push_back({K, N}); }
  }
  if (Ms.empty()) Ms = {2048};
  if (shapes.empty()) shapes = {{3840, 15360}};
  CK(cudaSetDevice(dev));
  cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));

  // паспорт сборки печатается ОДИН раз: регистры / кадр стека / smem по каждой гиперформе
  for (int a = 0; a < kNArms; ++a)
    printf("BUILD {\"tag\":\"%s\",\"regs\":%d,\"frame\":%d,\"smem\":%d}\n",
           kArms[a].tag, kArms[a].regs(), kArms[a].frame(), kArms[a].smem);
  fflush(stdout);

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
      const float al = 1.f, be = 0.f;
      auto cublas_run = [&] {
        CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &al, dB, CUDA_R_16F, K,
                        dA, CUDA_R_16F, K, &be, dR, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      };
      cublas_run(); CK(cudaDeviceSynchronize());
      // ломка симметрии: сам эталон проверен третьим путём
      const double cb_spot = spot_rel(dA, dB, dR, M, N, K, nspot);
      double t0 = T.run(cublas_run, 3);
      const int it_cb = std::max(3, std::min(2000, (int)(300.0 / std::max(t0, 1e-3))));

      struct Cand { int idx; double ms; double rel; double srel; };
      std::vector<Cand> res;
      for (int a = 0; a < kNArms; ++a) {
        const Arm& A_ = kArms[a];
        if (!A_.ok(M, N, K)) continue;
        CK(cudaMemset(dC, 0x7f, (size_t)M * N * 2));
        A_.launch(dA, dB, dC, M, N, K);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { cudaGetLastError(); printf("SKIP %s launch %s\n", A_.tag, cudaGetErrorString(e)); continue; }
        const unsigned long long left = poison_left(dC, (long)M * N);
        const double rel = rel_l2(dC, dR, (long)M * N);
        const double sr = spot_rel(dA, dB, dC, M, N, K, nspot);
        if (left != 0 || !(rel < gate) || !(sr < gate)) {
          printf("FAILGATE {\"tag\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"uncovered\":%llu,\"rel\":%.3e,\"spot\":%.3e}\n",
                 A_.tag, M, N, K, left, rel, sr);
          continue;
        }
        auto run = [&] { A_.launch(dA, dB, dC, M, N, K); };
        const double tw = T.run(run, 3);
        const int it = std::max(3, std::min(2000, (int)(300.0 / std::max(tw, 1e-3))));
        res.push_back({a, T.run(run, it), rel, sr});
      }
      // парные раунды против cuBLAS, чередование ВНУТРИ раунда
      std::vector<std::vector<double>> ratios(res.size());
      std::vector<double> cbms;
      for (int r = 0; r < rounds; ++r) {
        const double c = T.run(cublas_run, it_cb);
        cbms.push_back(c);
        for (size_t i = 0; i < res.size(); ++i) {
          const Arm& A_ = kArms[res[i].idx];
          auto run = [&] { A_.launch(dA, dB, dC, M, N, K); };
          const int it = std::max(3, std::min(2000, (int)(300.0 / std::max(res[i].ms, 1e-3))));
          const double t = T.run(run, it);
          ratios[i].push_back(c / t);
        }
      }
      auto med = [](std::vector<double> v) -> double {
        if (v.empty()) return -1;
        std::sort(v.begin(), v.end());
        return v.size() & 1 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
      };
      const double cb = med(cbms);
      printf("BASE {\"K\":%d,\"N\":%d,\"M\":%d,\"cublas_ms\":%.6f,\"cublas_tflops\":%.4f,\"cublas_spot_rel\":%.3e}\n",
             K, N, M, cb, flop / (cb * 1e-3) / 1e12, cb_spot);
      for (size_t i = 0; i < res.size(); ++i) {
        std::vector<double> rs = ratios[i];
        std::sort(rs.begin(), rs.end());
        printf("CAND {\"K\":%d,\"N\":%d,\"M\":%d,\"tag\":\"%s\",\"ms\":%.6f,\"tflops\":%.4f,"
               "\"ratio_med\":%.4f,\"ratio_min\":%.4f,\"ratio_max\":%.4f,\"rel\":%.3e,\"spot\":%.3e}\n",
               K, N, M, kArms[res[i].idx].tag, res[i].ms, flop / (res[i].ms * 1e-3) / 1e12,
               med(ratios[i]), rs.front(), rs.back(), res[i].rel, res[i].srel);
      }
      fflush(stdout);
      cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dR);
    }
  }
  cublasDestroy(h);
  return 0;
}
