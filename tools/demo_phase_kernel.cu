// -*- coding: utf-8 -*-
// ЭТАЛОННОЕ ЯДРО ДЛЯ САМОПРОВЕРКИ ФАЗОВОГО ПРОФИЛИРОВЩИКА.
//
// Это НЕ боевое ядро. Оно повторяет СКЕЛЕТ нашего форварда (подача -> рандеву -> первое
// умножение -> софтмакс -> второе умножение -> эпилог) в минимальном виде, чтобы на нём
// проверялось само устройство инструмента: разбор разметки, сборка вариантов, привязка SASS к
// фазам и -- главное -- РАБОТА ПЛОМБЫ. Пять фаз названы так же, как в замере §3b, чтобы таблицы
// читались рядом.
//
// Фаза 5 (эпилог) размечена НАРОЧНО: в боевой таблице §3b её нет, и она -- один из кандидатов на
// невязку 22.3 %. Здесь она нужна как проверка, что инструмент видит фазу, которую человек в
// прошлый раз не назвал.
//
// Сборка (без GPU, только компиляция):
//   nvcc -arch=sm_70 -lineinfo -cubin -O3 -DFMHA_STRIP_MASK=0 -o demo.cubin demo_phase_kernel.cu

#include <cuda_fp16.h>
#include "fmha_phase.h"

#ifndef DEMO_SEAL
#define DEMO_SEAL 1     // 0 = собрать БЕЗ пломбы, чтобы показать обвал стрип-варианта
#endif

#if DEMO_SEAL
#define DSEAL(...) FMHA_SEAL(__VA_ARGS__)
#define DSEAL_ARR(a, n) FMHA_SEAL_ARR(a, n)
#else
#define DSEAL(...) ((void)0)
#define DSEAL_ARR(a, n) ((void)0)
#endif

#define DEMO_BK 32
#define DEMO_D 32
#define DEMO_ACC 8

extern "C" __global__ __launch_bounds__(128, 1) void demo_fwd(
    const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
    __half* __restrict__ O, float* __restrict__ Lse, int Sk, float scale) {
  __shared__ __half sK[DEMO_BK * DEMO_D];
  __shared__ __half sV[DEMO_BK * DEMO_D];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float o[DEMO_ACC];
#pragma unroll
  for (int i = 0; i < DEMO_ACC; ++i) o[i] = 0.f;
  float m_run = -1e30f, l_run = 0.f;

  __half q[DEMO_ACC];
#pragma unroll
  for (int i = 0; i < DEMO_ACC; ++i) q[i] = Q[threadIdx.x * DEMO_ACC + i];

  for (int kb = 0; kb < Sk; kb += DEMO_BK) {
    // Накопитель ПЛИТКИ -- обнуляется каждый оборот, как в боевом ядре. Именно поэтому снятие
    // первого умножения делает acc КОНСТАНТНЫМ НУЛЁМ (а не «прошлым значением»), и без пломбы
    // компилятор сворачивает весь софтмакс следом.
    float acc[DEMO_ACC];
#pragma unroll
    for (int i = 0; i < DEMO_ACC; ++i) acc[i] = 0.f;

    // ---- ПОДАЧА: везущие тащат плитку K/V в разделяемую -------------------------------------
    FMHA_PHASE(feed, 3) {
      for (int t = threadIdx.x; t < DEMO_BK * DEMO_D; t += blockDim.x) {
        sK[t] = K[(long)kb * DEMO_D + t];
        sV[t] = V[(long)kb * DEMO_D + t];
      }
    }
    // ПОДСТАНОВКА. Снять подачу ЦЕЛИКОМ нельзя: чтение неинициализированной разделяемой памяти --
    // неопределённое поведение, и компилятор вправе свернуть ОБА умножения (инструмент это и
    // поймал: 520->392 и 512->290). Поэтому снимается ровно то, что меряется -- ОБРАЩЕНИЕ В
    // ГЛОБАЛЬНУЮ, -- а разделяемая остаётся ОПРЕДЕЛЁННОЙ дешёвой раскладкой.
    FMHA_PHASE_ELSE(feed, 3) {
      for (int t = threadIdx.x; t < DEMO_BK * DEMO_D; t += blockDim.x) {
        sK[t] = __ushort_as_half((unsigned short)(t * 2654435761u >> 17));
        sV[t] = __ushort_as_half((unsigned short)(t * 40503u >> 3));
      }
    }
    // Рандеву стоит СНАРУЖИ фазы подачи: иначе снятие подачи снимало бы и защёлку, и замер
    // приписывал бы подаче чужую цену (см. комментарий про барьеры в fmha_phase.h).
    FMHA_PHASE(rendez, 4) { __syncthreads(); }

    // ---- ПЕРВОЕ УМНОЖЕНИЕ Q*K^T --------------------------------------------------------------
    FMHA_PHASE(gemm1, 0) {
#pragma unroll
      for (int i = 0; i < DEMO_ACC; ++i) {
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < DEMO_D; ++k)
          s = fmaf(__half2float(q[i]), __half2float(sK[k * DEMO_BK + ((lane + i) & 31)]), s);
        acc[i] = s * scale;
      }
    }
    // ПОДСТАНОВКА: РАЗЛИЧНЫЕ значения, а не общий ноль. Пломба делает значение непрозрачным, но
    // НЕ делает его уникальным: восемь одинаковых непрозрачных acc[i] дают восемь ОДИНАКОВЫХ
    // выражений exp2f(acc[i]-m), и CSE склеивает их в одно (замерено: 9 MUFU.EX2 -> 2 ПРИ
    // ПОСТАВЛЕННОЙ ПЛОМБЕ). Различные константы + пломба -- склеивать нечего.
    FMHA_PHASE_ELSE(gemm1, 0) {
#pragma unroll
      for (int i = 0; i < DEMO_ACC; ++i) acc[i] = __int_as_float(0x3f800000 + i);
    }
    // ПЛОМБА: без неё acc[] при снятой фазе -- компилируемая константа, и весь софтмакс со
    // вторым умножением сворачивается. Замер тогда припишет первому умножению чужую работу.
    DSEAL_ARR(acc, DEMO_ACC);

    // ---- СОФТМАКС ----------------------------------------------------------------------------
    FMHA_PHASE(softmax, 1) {
      float m = m_run;
#pragma unroll
      for (int i = 0; i < DEMO_ACC; ++i) m = fmaxf(m, acc[i]);
#pragma unroll
      for (int off = 16; off; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
      const float corr = exp2f(m_run - m);
      float l = 0.f;
#pragma unroll
      for (int i = 0; i < DEMO_ACC; ++i) { acc[i] = exp2f(acc[i] - m); l += acc[i]; }
#pragma unroll
      for (int off = 16; off; off >>= 1) l += __shfl_xor_sync(0xffffffffu, l, off);
      l_run = l_run * corr + l;
      m_run = m;
#pragma unroll
      for (int i = 0; i < DEMO_ACC; ++i) o[i] *= corr;
    }
    DSEAL_ARR(acc, DEMO_ACC);
    DSEAL(m_run, l_run);

    // ---- ВТОРОЕ УМНОЖЕНИЕ P*V ----------------------------------------------------------------
    FMHA_PHASE(gemm2, 2) {
#pragma unroll
      for (int i = 0; i < DEMO_ACC; ++i) {
        float s = o[i];
#pragma unroll
        for (int k = 0; k < DEMO_BK; ++k)
          s = fmaf(acc[(k >> 2) & (DEMO_ACC - 1)], __half2float(sV[k * DEMO_D + ((lane + i) & 31)]), s);
        o[i] = s;
      }
    }
    DSEAL_ARR(o, DEMO_ACC);

    FMHA_PHASE(rendez, 4) { __syncthreads(); }
  }

  // ---- ЭПИЛОГ: фаза, которой в таблице §3b НЕТ ------------------------------------------------
  FMHA_PHASE(epilogue, 5) {
    const float inv = 1.f / l_run;
#pragma unroll
    for (int i = 0; i < DEMO_ACC; ++i) O[threadIdx.x * DEMO_ACC + i] = __float2half(o[i] * inv);
    if (lane == 0) Lse[blockIdx.x * 4 + warp] = m_run + log2f(l_run);
  }
  // ПОДСТАНОВКА ЧЕРЕЗ СТОК. Эпилог -- ЕДИНСТВЕННЫЙ потребитель всего, что ядро посчитало. Снять
  // его без стока = сделать мёртвым ВСЁ ядро: инструмент показал 1888 -> 40 команд, то есть
  // "доля эпилога" вышла бы 98 %. FMHA_SINK оставляет запись в графе (компилятор не знает
  // значения fmha_phase_never), но не исполняет её.
  FMHA_PHASE_ELSE(epilogue, 5) {
#pragma unroll
    for (int i = 0; i < DEMO_ACC; ++i) FMHA_SINK(O, threadIdx.x * DEMO_ACC + i, __float2half(o[i]));
    FMHA_SINK(Lse, blockIdx.x * 4 + warp, m_run + l_run);
  }
}
