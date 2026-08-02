// SPDX-License-Identifier: LicenseRef-TRL-1.0
// Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
// Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
//
// ГРАНИЦА ЯДРА И ФРЕЙМВОРКА.  Здесь нет ни одного типа фреймворка -- только POD и указатели.
// Это не эстетика: заранее собранный `.so` прибивает ABI (компилятор + стандартная библиотека
// + версия фреймворка), и ШЕСТЬ ОТКАЗОВ ОБВЯЗКИ ЗА СУТКИ в этом проекте были ровно там.
// Первичная форма поставки -- ИСХОДНИК, который хост-дерево собирает своим тулчейном.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernel.cuh"

namespace tempo {
namespace gen {

// Плоский POD: всё, что нужно ядру, и ничего сверх того.
struct GemmParams {
  const __half* A;   // [M][K], k-мажорный
  const __half* B;   // [N][K], k-мажорный  (умножение C = A * B^T)
  __half* C;         // [M][N]
  int M, N, K;
  int lda, ldb, ldc;  // шаги строк В ЭЛЕМЕНТАХ; проверять кратность 16 БАЙТАМ, а не элементам
};

// Один вариант = одна инстанциация шаблона. Таблица вариантов -- в dispatch.inc,
// она порождается из select.py и НЕ пишется руками.
template <int BM, int BN, int BK, int WM, int WN, int STAGES, int GSTAGE, int FPREF, int GROUP,
          int EPI, int SWZ, bool PRED, int MINB>
inline cudaError_t launch_one(const GemmParams& p, cudaStream_t s) {
  constexpr int THREADS = WM * WN * 32;
  const int nm = (p.M + BM - 1) / BM;
  const int nn = p.N / BN;
  const dim3 grid(nm * nn), blk(THREADS);
  k_gemm<BM, BN, BK, WM, WN, STAGES, GSTAGE, FPREF, GROUP, EPI, SWZ, PRED, MINB>
      <<<grid, blk, 0, s>>>(p.A, p.B, p.C, p.M, p.N, p.K);
  return cudaGetLastError();
}

// ПРЕДИКАТ ПРИМЕНИМОСТИ. Возвращает false -- значит форма НЕ ПОКРЫТА, и вызывающий обязан
// уйти на свой штатный путь. Покрытие НИКОГДА не полное; продукт без отката ломает чужой
// сервер на первой нестандартной форме.
inline bool applicable(const GemmParams& p) {
  if (p.M <= 0 || p.N <= 0 || p.K <= 0) return false;
  if (p.lda != p.K || p.ldb != p.K || p.ldc != p.N) return false;  // только плотная раскладка
  if ((p.K % 8) != 0 || (p.N % 8) != 0) return false;              // 16-байтовое выравнивание
  if ((reinterpret_cast<uintptr_t>(p.A) % 16) != 0) return false;
  if ((reinterpret_cast<uintptr_t>(p.B) % 16) != 0) return false;
  if ((reinterpret_cast<uintptr_t>(p.C) % 16) != 0) return false;
  return true;
}

// Диспетчер по форме. Тело -- сгенерированная таблица; если ни одна строка не подошла,
// возвращается cudaErrorInvalidConfiguration, и это ЗАКОННЫЙ ответ «уходи в откат».
cudaError_t launch(const GemmParams& p, cudaStream_t s);

}  // namespace gen
}  // namespace tempo
