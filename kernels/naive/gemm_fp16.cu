// SPDX-License-Identifier: LicenseRef-TRL-1.0
// Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
// Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
// ОБРАЗЕЦ ВХОДА ТЕМПОЛЯТОРА №1: плотное fp16-умножение, написанное прямо.
//
// Это НЕ бенчмарк и НЕ кандидат. Это НИЖНЯЯ ОТМЕТКА -- то, что подаётся на вход компилятору
// минимального временного отпечатка, и от чего считается его собственная метрика «во сколько раз
// выход обгоняет вход».
//
// ПРАВИЛО НАПИСАНИЯ (соблюдено буквально): так, как написал бы человек, ЗНАЮЩИЙ АЛГОРИТМ и
// НЕ ЗНАЮЩИЙ МАШИНЫ. Поэтому здесь НЕТ ничего из перечисленного:
//   * плиток в разделяемой памяти и двойной буферизации,
//   * тензорных инструкций (HMMA), фрагментов, перестановок раскладки,
//   * векторных загрузок, дополнения строк, переупорядочения блоков под L2,
//   * разбиения по K, регистровых накопителей шире одного элемента.
// Есть ровно определение: C[m][n] = sum_k A[m][k] * B[n][k].
//
// ЕДИНСТВЕННОЕ, ЧТО ЗДЕСЬ ОТ МАШИНЫ, -- НАКОПИТЕЛЬ float, а не __half. Это НЕ оптимизация, а
// корректность: при K = 3840..15360 накопление в fp16 теряет знаковые разряды и даёт другой ответ.
// Ставить сюда fp16-накопитель значило бы сравнивать со сломанным входом, то есть завышать выигрыш.
//
// ФОРМА СООТВЕТСТВУЕТ БОЕВОЙ ЛИНЕЙНОЙ ЧАСТИ ДОСЛОВНО:
//   torch.nn.functional.linear(x[M,K], W[N,K]) -> y[M,N],  оба операнда строчно-мажорные,
//   то есть C = A * B^T. Именно так лежат веса Gemma-4-12B, и именно этот вызов идёт в cuBLAS.
//
// Сборка отдельно (даёт время в мс на форму):
//   nvcc -O3 -std=c++17 -arch=sm_70 -o naive naive_gemm_fp16.cu && ./naive 8192 4096 3840
// Как единица трансляции внутри харнесса: -DNAIVE_GEMM_AS_HEADER (тогда main не компилируется).
// ============================================================================================
// TEMPO-OP: gemm
//   dtype: a=fp16 b=fp16 c=fp16 acc=fp32
//   layout: a=k b=k c=n
//   shape: M=* N=* K=*
//   tol_rel_l2: 1e-3
//   coverage: atomic_stamp
//   entry: naive_gemm_fp16
//   signature: (const __half* A, const __half* B, __half* C, int M, int N, int K)
// TEMPO-OP-END
//
// ЭТОТ БЛОК -- ЕДИНСТВЕННЫЙ ИСТОЧНИК СПЕЦИФИКАЦИИ, и он живёт ЗДЕСЬ, а не в отдельном файле
// рядом. Отдельный файл -- это второе место, и оно неминуемо разъедется с первым: сигнатура
// поменяется, описание останется. Поэтому `tempo recognize` читает блок И СВЕРЯЕТ его с
// сигнатурой пуска: если объявленной точки входа в тексте нет, распознавание ОТКАЗЫВАЕТ.
//
// `M=*` означает «задаётся при запуске»: форма приходит из боевого набора (bench/shapes/*.json),
// а не вшивается во вход.
//
// СПЕЦИФИКАЦИЯ -- ЭТО НЕ ПЕРЕВОД ИСХОДНИКА. Продукт ПОРОЖДАЕТ ядро по объявленной операции, а
// не транслирует то, что здесь написано: парсера C++ в версии 1 нет и не будет. Незнакомое
// ядро получает отказ, а не плохой код.
// ============================================================================================

#include <cuda_fp16.h>

#ifndef NAIVE_GEMM_TILE
#define NAIVE_GEMM_TILE 16          // 16x16 = 256 нитей: первое, что приходит в голову
#endif

// Одна нить -- один элемент выхода. Прямая запись формулы.
__global__ void naive_gemm_fp16(const __half* A, const __half* B, __half* C, int M, int N, int K) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k)
    acc += __half2float(A[(long)m * K + k]) * __half2float(B[(long)n * K + k]);
  C[(long)m * N + n] = __float2half(acc);
}

// Запуск в одну строку -- чтобы харнесс вызывал ровно то же, что и отдельная сборка.
inline void naive_gemm_fp16_launch(const __half* A, const __half* B, __half* C,
                                   int M, int N, int K, cudaStream_t s = 0) {
  constexpr int T = NAIVE_GEMM_TILE;
  const dim3 blk(T, T), grd((N + T - 1) / T, (M + T - 1) / T);
  naive_gemm_fp16<<<grd, blk, 0, s>>>(A, B, C, M, N, K);
}

#ifndef NAIVE_GEMM_AS_HEADER
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv) {
  const int M = argc > 1 ? atoi(argv[1]) : 4096;
  const int N = argc > 2 ? atoi(argv[2]) : 4096;
  const int K = argc > 3 ? atoi(argv[3]) : 4096;
  __half *A, *B, *C;
  cudaMalloc(&A, (size_t)M * K * 2);
  cudaMalloc(&B, (size_t)N * K * 2);
  cudaMalloc(&C, (size_t)M * N * 2);
  cudaMemset(A, 0x11, (size_t)M * K * 2);
  cudaMemset(B, 0x11, (size_t)N * K * 2);
  naive_gemm_fp16_launch(A, B, C, M, N, K);
  cudaDeviceSynchronize();
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  naive_gemm_fp16_launch(A, B, C, M, N, K);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  printf("M=%d N=%d K=%d  %.3f ms  %.2f GFLOP/s\n", M, N, K, ms,
         2.0 * M * N * K / (ms * 1e-3) / 1e9);
  return 0;
}
#endif
