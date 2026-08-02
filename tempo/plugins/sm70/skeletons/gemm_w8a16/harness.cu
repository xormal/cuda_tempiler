// SPDX-License-Identifier: LicenseRef-TRL-1.0
// Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
// Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
//
// СТЕНД W8A16. ПЛАНКИ ЗДЕСЬ -- НЕ cuBLAS (в int8 на тензорных ядрах sm_70 у него нет пути вовсе,
// и сравнение с ним было бы тривиально верным и потому бессодержательным):
//   (1) НАШ ЖЕ fp16-выход того же конвейера на той же форме -- показывает, что байт покупает и чего
//       не покупает;
//   (2) доля тензорного пика 125.3 ТФЛОП/с;
//   (3) размещение разворота: A (везущие) против B (считающие) -- вопрос заказчика, решаемый ЗАМЕРОМ.
// cuBLAS печатается ТОЛЬКО как якорь шкалы, заявок по нему не делается.
//
// ГЕЙТ КОРРЕКТНОСТИ. Эталон -- cuBLAS-fp16 на ТОЧНО деквантованном весе (int8 -> fp16 без потерь:
// целые до 2048 представимы в fp16 точно). Значит расхождение может идти только от НАШЕЙ арифметики,
// а не от квантования, и порог тот же, что у fp16-плеча.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <array>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "kernel.cuh"
#define TEMPO_FP16_SKELETON 1
#include "../gemm_hmma884/kernel.cuh"

using namespace tempo::gen8;
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)
#define CB(x) do { cublasStatus_t e=(x); if(e){printf("cuBLAS %s @%d: %d\n",#x,__LINE__,(int)e);exit(1);} } while(0)

__global__ void k_fillh(__half* p, long n, unsigned seed) {
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    unsigned h = (unsigned)i * 2654435761u + seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    p[i] = __float2half(((float)(h & 0xFFFF) / 65535.f - 0.5f) * 0.25f);
  }
}
__global__ void k_filli8(int8_t* p, __half* ph, long n, unsigned seed) {
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    unsigned h = (unsigned)i * 2654435761u + seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    const int v = (int)(h % 255u) - 127;
    p[i] = (int8_t)v;
    ph[i] = __float2half((float)v);   // ТОЧНАЯ деквантизация: |v| <= 127 < 2048
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
__global__ void k_poison(const __half* a, long n, unsigned long long* o) {
  unsigned long long c = 0;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
    if (*reinterpret_cast<const unsigned short*>(a + i) == 0x7f7f) ++c;
  atomicAdd(o, c);
}
static double rel_l2(const __half* a, const __half* b, long n) {
  double* d; CK(cudaMalloc(&d, 16)); CK(cudaMemset(d, 0, 16));
  k_rel<<<512, 256>>>(a, b, n, d); CK(cudaDeviceSynchronize());
  double h[2]; CK(cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost)); cudaFree(d);
  return h[1] > 0 ? sqrt(h[0] / h[1]) : (h[0] > 0 ? 1.0 : 0.0);
}
static unsigned long long poison_left(const __half* a, long n) {
  unsigned long long* d; CK(cudaMalloc(&d, 8)); CK(cudaMemset(d, 0, 8));
  k_poison<<<512, 256>>>(a, n, d); CK(cudaDeviceSynchronize());
  unsigned long long h; CK(cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost)); cudaFree(d);
  return h;
}
struct Timer {
  cudaEvent_t e0, e1;
  Timer() { cudaEventCreate(&e0); cudaEventCreate(&e1); }
  template <class F> double run(F&& f, int it) {
    for (int i = 0; i < (it + 2) / 3; ++i) f();
    CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
    for (int i = 0; i < it; ++i) f();
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); return ms / it;
  }
};

template <int BM, int BN, int BK, int WM, int WN, int UNPACK, int GROUP, bool PRED, int MINB>
struct C8 {
  static constexpr int smem = Smem8<BM, BN, BK, UNPACK>::value;
  static void launch(const __half* A, const int8_t* B, const float* rs, __half* C, int M, int N, int K) {
    auto k = k_gemm8<BM, BN, BK, WM, WN, UNPACK, GROUP, PRED, MINB>;
    static bool once = false;
    if (!once) { CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); once = true; }
    const int nm = PRED ? (M + BM - 1) / BM : M / BM;
    k<<<dim3(nm * (N / BN)), dim3(WM * WN * 32), smem>>>(A, B, rs, C, M, N, K);
  }
  static bool ok(int M, int N, int K) { return (N % BN == 0) && (K % BK == 0) && smem <= 96 * 1024; }
  static int regs() { cudaFuncAttributes a; cudaFuncGetAttributes(&a, k_gemm8<BM, BN, BK, WM, WN, UNPACK, GROUP, PRED, MINB>); return a.numRegs; }
  static int frame() { cudaFuncAttributes a; cudaFuncGetAttributes(&a, k_gemm8<BM, BN, BK, WM, WN, UNPACK, GROUP, PRED, MINB>); return (int)a.localSizeBytes; }
};
struct Arm8 { const char* tag; bool (*ok)(int, int, int); void (*launch)(const __half*, const int8_t*, const float*, __half*, int, int, int); int (*regs)(); int (*frame)(); int smem; };
#define C8F(t, BM, BN, BK, WM, WN, U, G, P, MB) \
  Arm8{t, &C8<BM,BN,BK,WM,WN,U,G,P,MB>::ok, &C8<BM,BN,BK,WM,WN,U,G,P,MB>::launch, \
       &C8<BM,BN,BK,WM,WN,U,G,P,MB>::regs, &C8<BM,BN,BK,WM,WN,U,G,P,MB>::frame, C8<BM,BN,BK,WM,WN,U,G,P,MB>::smem}

// плечо fp16 из скелета №1 -- ТОТ ЖЕ конвейер, тот же выбор, только формат другой
template <int BM, int BN, int BK, int WM, int WN, int FP, int GROUP, int MINB>
struct CH {
  static constexpr int smem = tempo::gen::SmemBytes<BM, BN, BK, WM, WN, 2, 0>::value;
  static void launch(const __half* A, const __half* B, __half* C, int M, int N, int K) {
    auto k = tempo::gen::k_gemm<BM, BN, BK, WM, WN, 2, 1, FP, GROUP, 0, 2, true, MINB>;
    static bool once = false;
    if (!once) { CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); once = true; }
    k<<<dim3(((M + BM - 1) / BM) * (N / BN)), dim3(WM * WN * 32), smem>>>(A, B, C, M, N, K);
  }
};

int main(int argc, char** argv) {
  int dev = 0, rounds = 5;
  std::vector<int> Ms;
  std::vector<std::array<int, 2>> shapes;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--dev")) dev = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--rounds")) rounds = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--m")) { char* s = argv[++i]; for (char* t = strtok(s, ","); t; t = strtok(nullptr, ",")) Ms.push_back(atoi(t)); }
    else if (!strcmp(argv[i], "--shape")) { int K, N; sscanf(argv[++i], "%d:%d", &K, &N); shapes.push_back({K, N}); }
  }
  if (Ms.empty()) Ms = {1, 8, 32, 128, 512, 2048};
  if (shapes.empty()) shapes = {{3840, 15360}};
  CK(cudaSetDevice(dev));
  cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));

  const Arm8 arms[] = {
      C8F("A_unpack@store_128x256", 128, 256, 32, 2, 4, 0, 8, true, 1),
      C8F("B_unpack@load_128x256",  128, 256, 32, 2, 4, 1, 8, true, 1),
      C8F("A_unpack@store_128x128", 128, 128, 32, 2, 2, 0, 8, true, 2),
      C8F("B_unpack@load_128x128",  128, 128, 32, 2, 2, 1, 8, true, 2),
      C8F("A_unpack@store_32x128",   32, 128, 32, 1, 4, 0, 8, true, 2),
      C8F("B_unpack@load_32x128",    32, 128, 32, 1, 4, 1, 8, true, 2),
      C8F("A_unpack@store_16x256k64",   16, 256, 64, 1, 4, 0, 8, true, 2),
      C8F("B_unpack@load_16x256k64",    16, 256, 64, 1, 4, 1, 8, true, 2),
  };
  const int NA = (int)(sizeof(arms) / sizeof(arms[0]));
  for (int a = 0; a < NA; ++a)
    printf("BUILD {\"tag\":\"%s\",\"regs\":%d,\"frame\":%d,\"smem\":%d}\n", arms[a].tag, arms[a].regs(), arms[a].frame(), arms[a].smem);

  for (auto& s : shapes) {
    const int K = s[0], N = s[1];
    for (int M : Ms) {
      __half *dA, *dBh, *dC, *dR; int8_t* dB; float* dRS;
      CK(cudaMalloc(&dA, (size_t)M * K * 2)); CK(cudaMalloc(&dB, (size_t)N * K));
      CK(cudaMalloc(&dBh, (size_t)N * K * 2)); CK(cudaMalloc(&dC, (size_t)M * N * 2));
      CK(cudaMalloc(&dR, (size_t)M * N * 2)); CK(cudaMalloc(&dRS, (size_t)M * 4));
      k_fillh<<<512, 256>>>(dA, (long)M * K, 11u);
      k_filli8<<<512, 256>>>(dB, dBh, (long)N * K, 77u);
      k_rowsum<<<M, 256>>>(dA, dRS, M, K);
      CK(cudaDeviceSynchronize());
      const double flop = 2.0 * M * N * K;
      Timer T;
      const float al = 1.f, be = 0.f;
      auto cub = [&] { CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &al, dBh, CUDA_R_16F, K, dA, CUDA_R_16F, K, &be, dR, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); };
      cub(); CK(cudaDeviceSynchronize());
      const double tcb = T.run(cub, std::max(3, std::min(2000, (int)(300.0 / 1.0))));
      // плечо fp16 нашего же конвейера (то же ядро, что выиграло в ходе 1)
      auto f16 = [&] {
        if (M >= 128) CH<128, 256, 32, 2, 4, 2, 8, 1>::launch(dA, dBh, dC, M, N, K);
        else CH<32, 128, 32, 1, 4, 1, 8, 2>::launch(dA, dBh, dC, M, N, K);
      };
      CK(cudaMemset(dC, 0x7f, (size_t)M * N * 2)); f16(); CK(cudaDeviceSynchronize());
      const double rel16 = rel_l2(dC, dR, (long)M * N);
      const double t16 = T.run(f16, std::max(3, std::min(2000, (int)(300.0 / std::max(T.run(f16, 3), 1e-3)))));
      printf("BASE {\"K\":%d,\"N\":%d,\"M\":%d,\"cublas_ms\":%.6f,\"cublas_tflops\":%.3f,"
             "\"fp16_ms\":%.6f,\"fp16_tflops\":%.3f,\"fp16_rel\":%.2e}\n",
             K, N, M, tcb, flop / (tcb * 1e-3) / 1e12, t16, flop / (t16 * 1e-3) / 1e12, rel16);
      for (int a = 0; a < NA; ++a) {
        if (!arms[a].ok(M, N, K)) continue;
        CK(cudaMemset(dC, 0x7f, (size_t)M * N * 2));
        arms[a].launch(dA, dB, dRS, dC, M, N, K);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { cudaGetLastError(); printf("SKIP %s %s\n", arms[a].tag, cudaGetErrorString(e)); continue; }
        const unsigned long long left = poison_left(dC, (long)M * N);
        const double rel = rel_l2(dC, dR, (long)M * N);
        if (left || !(rel < 5e-3)) { printf("FAILGATE {\"tag\":\"%s\",\"M\":%d,\"uncovered\":%llu,\"rel\":%.3e}\n", arms[a].tag, M, left, rel); continue; }
        auto run = [&] { arms[a].launch(dA, dB, dRS, dC, M, N, K); };
        const double tw = T.run(run, 3);
        const int it = std::max(3, std::min(2000, (int)(300.0 / std::max(tw, 1e-3))));
        std::vector<double> r16, rcb;
        double t = T.run(run, it);
        for (int r = 0; r < rounds; ++r) {
          const double x = T.run(run, it), y = T.run(f16, it), z = T.run(cub, it);
          r16.push_back(y / x); rcb.push_back(z / x); t = std::min(t, x);
        }
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        printf("CAND {\"K\":%d,\"N\":%d,\"M\":%d,\"tag\":\"%s\",\"ms\":%.6f,\"tflops\":%.3f,"
               "\"vs_fp16\":%.4f,\"vs_cublas\":%.4f,\"rel\":%.2e}\n",
               K, N, M, arms[a].tag, t, flop / (t * 1e-3) / 1e12, med(r16), med(rcb), rel);
      }
      fflush(stdout);
      cudaFree(dA); cudaFree(dB); cudaFree(dBh); cudaFree(dC); cudaFree(dR); cudaFree(dRS);
    }
  }
  return 0;
}
