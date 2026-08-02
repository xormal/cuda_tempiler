// SPDX-License-Identifier: LicenseRef-TRL-1.0
// Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
// Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
//
// ФАЛЬСИФИКАТОР ПОСТАВЛЯЕМОГО ПУТИ. Собирается ТОЛЬКО из файлов поставки: launch.cuh +
// launch.cu + dispatch.inc + kernel.cuh. Ни одной строки испытательной обвязки.
//
// ЗАЧЕМ ОН ЕСТЬ. Гейт поставки сверял dispatch.inc с select.py и полноту паспорта, но ЯДРО
// НЕ ЗАПУСКАЛ -- и поэтому пропустил отказ, при котором отгружаемый путь не исполнялся
// вовсе (запуск с НУЛЁМ динамической разделяемой при `extern __shared__` в ядре ->
// `an illegal memory access` -> мёртвый контекст на ПЕРВОЙ же боевой форме).
// ПРАВИЛО: гейт, который не ЗАПУСКАЕТ отгружаемое, проверяет не поставку.
//
// Что проверяется: применимость, код возврата запуска, СИНХРОНИЗАЦИЯ (без неё отказ ядра
// не виден вовсе) и значения против эталона на CPU. Заливка выхода ЯДОВИТАЯ (0x7f7f): ядро,
// которое не записало ячейку, обязано провалить сверку, а не унаследовать ноль.
//
// Сборка и запуск -- через `test_gate.py --with-card` (ему же принадлежат правила
// дисциплины: свободная карта не нужна, потому что здесь сверяются ЗНАЧЕНИЯ, а не время).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "launch.cuh"

#define CK(x)                                                              \
  do {                                                                     \
    cudaError_t e_ = (x);                                                  \
    if (e_ != cudaSuccess) printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); \
  } while (0)

int main() {
  struct Shape {
    int M, N, K;
    const char* role;
  };
  // Боевые формы линейной части Gemma-4-12B: обе полосы лестницы и обе ветви по N.
  const Shape sh[] = {
      {1, 4096, 3840, "q"},        {8, 4096, 3840, "q"},
      {32, 15360, 3840, "gate,up"}, {128, 4096, 3840, "q"},
      {512, 15360, 3840, "gate,up"}, {2048, 3840, 15360, "down"},
  };
  int fail = 0, pass = 0;
  for (const Shape& s : sh) {
    const size_t na = (size_t)s.M * s.K, nb = (size_t)s.N * s.K, nc = (size_t)s.M * s.N;
    __half *A = nullptr, *B = nullptr, *C = nullptr;
    CK(cudaMalloc(&A, na * 2));
    CK(cudaMalloc(&B, nb * 2));
    CK(cudaMalloc(&C, nc * 2));
    std::vector<__half> ha(na), hb(nb);
    for (size_t i = 0; i < na; ++i) ha[i] = __float2half(((int)(i % 7) - 3) * 0.125f);
    for (size_t i = 0; i < nb; ++i) hb[i] = __float2half(((int)(i % 5) - 2) * 0.125f);
    CK(cudaMemcpy(A, ha.data(), na * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(B, hb.data(), nb * 2, cudaMemcpyHostToDevice));
    CK(cudaMemset(C, 0x7f, nc * 2));  // ЯДОВИТАЯ ЗАЛИВКА

    tempo::gen::GemmParams p{A, B, C, s.M, s.N, s.K, s.K, s.K, s.N};
    const bool app = tempo::gen::applicable(p);
    const cudaError_t e = tempo::gen::launch(p, 0);
    const cudaError_t se = cudaDeviceSynchronize();

    double err = 0.0;
    int bad = 0;
    if (se == cudaSuccess && e == cudaSuccess) {
      std::vector<__half> hc(nc);
      CK(cudaMemcpy(hc.data(), C, nc * 2, cudaMemcpyDeviceToHost));
      for (int t = 0; t < 64; ++t) {
        const int m = (t * 37) % s.M, n = (t * 53) % s.N;
        double ref = 0.0;
        for (int k = 0; k < s.K; ++k)
          ref += (double)__half2float(ha[(size_t)m * s.K + k]) *
                 (double)__half2float(hb[(size_t)n * s.K + k]);
        const double got = __half2float(hc[(size_t)m * s.N + n]);
        const double d = fabs(got - ref) / (fabs(ref) + 1e-3);
        if (d > 3e-3) ++bad;
        if (d > err) err = d;
      }
    }
    printf("%-8s M=%-5d N=%-6d K=%-6d applicable=%d launch=%-26s sync=%-34s maxrel=%.3e bad=%d\n",
           s.role, s.M, s.N, s.K, (int)app, cudaGetErrorString(e), cudaGetErrorString(se), err,
           bad);
    if (se == cudaSuccess && e == cudaSuccess && bad == 0)
      ++pass;
    else
      ++fail;
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    if (se != cudaSuccess) {
      printf("  КОНТЕКСТ МЁРТВ -- остальные формы не проверяются\n");
      break;
    }
  }
  printf("ИТОГ ПОСТАВЛЯЕМОГО ПУТИ: PASS %d / FAIL %d\n", pass, fail);
  return fail ? 1 : 0;
}
