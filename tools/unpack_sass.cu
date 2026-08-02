// Стоимость РАСПАКОВКИ веса в инструкциях НА ЭЛЕМЕНТ -- по SASS, а не по исходнику.
//
// Роль, которую эмулируем: ВЕЗУЩИЙ варп. Он читает упакованный вес из глобальной, разворачивает
// в fp16 В РЕГИСТРАХ и кладёт в разделяемую. Считающие варпы не меняются вовсе, поэтому вся цена
// новых форматов -- здесь. Разделяемая память от формата НЕ зависит (всегда fp16).
//
// Метод счёта -- РАЗНОСТНЫЙ: одно и то же ядро собирается на NELEM=32 и NELEM=64 элементов на
// поток; (I64 - I32)/32 = инструкций на элемент. Пролог/эпилог/адресация сокращаются.
//
// Компиляция: nvcc -arch=sm_70 -cubin -DNELEM=32 ; cuobjdump -sass
#include <cuda_fp16.h>
#include <cstdint>

#ifndef NELEM
#define NELEM 32
#endif

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

extern __shared__ uint32_t sm[];   // разделяемая: ВСЕГДА fp16, по 2 значения в слове

// -----------------------------------------------------------------------------------------
// F0. fp16 как есть -- базовый везущий (ничего не разворачивает)
// -----------------------------------------------------------------------------------------
__global__ void f0_fp16(const uint32_t* __restrict__ src, uint32_t* dst) {
  const int t = threadIdx.x;
  const uint32_t* p = src + t * (NELEM / 2);
#pragma unroll
  for (int i = 0; i < NELEM / 2; ++i) sm[t * (NELEM / 2) + i] = p[i];
  __syncthreads();
  if (t == 0) dst[blockIdx.x] = sm[blockIdx.y];
}

// -----------------------------------------------------------------------------------------
// F1. int8 -> fp16 через 0x6400|u: ДВЕ prmt на четыре значения (наш отгруженный приём)
// -----------------------------------------------------------------------------------------
__global__ void f1_int8(const uint32_t* __restrict__ src, uint32_t* dst) {
  const int t = threadIdx.x;
  const uint32_t* p = src + t * (NELEM / 4);
  const uint32_t magic = 0x64646464u;
#pragma unroll
  for (int i = 0; i < NELEM / 4; ++i) {
    uint32_t w = p[i];
    sm[t * (NELEM / 2) + 2 * i + 0] = prmt(w, magic, 0x4140u);
    sm[t * (NELEM / 2) + 2 * i + 1] = prmt(w, magic, 0x4342u);
  }
  __syncthreads();
  if (t == 0) dst[blockIdx.x] = sm[blockIdx.y];
}

// -----------------------------------------------------------------------------------------
// F1p. то же + ПОКАЗАТЕЛЬ НА ГРУППУ (масштаб -- степень двойки): магия 0x64+(D<<2) уезжает в
// РЕГИСТР-источник prmt, то есть масштаб стоит НОЛЬ инструкций (volta_prims.h §4oo).
// -----------------------------------------------------------------------------------------
__global__ void f1p_int8_pow2(const uint32_t* __restrict__ src, uint32_t* dst, int D) {
  const int t = threadIdx.x;
  const uint32_t* p = src + t * (NELEM / 4);
  const uint32_t byte = 0x64u + (uint32_t)(D << 2);
  const uint32_t magic = byte * 0x01010101u;          // одна IMAD на ГРУППУ, не на элемент
#pragma unroll
  for (int i = 0; i < NELEM / 4; ++i) {
    uint32_t w = p[i];
    sm[t * (NELEM / 2) + 2 * i + 0] = prmt(w, magic, 0x4140u);
    sm[t * (NELEM / 2) + 2 * i + 1] = prmt(w, magic, 0x4342u);
  }
  __syncthreads();
  if (t == 0) dst[blockIdx.x] = sm[blockIdx.y];
}

// -----------------------------------------------------------------------------------------
// F2. int10 ПРОГРЕССИВНЫЙ: план A = СТАРШИЕ 8 бит (сам по себе законный int8),
//     план B = младшие 2. Сборка u10 = (u8<<2)|b2 -- бит-уровневая, prmt не хватает.
// -----------------------------------------------------------------------------------------
__global__ void f2_int10_progressive(const uint32_t* __restrict__ a,
                                     const uint32_t* __restrict__ b, uint32_t* dst) {
  const int t = threadIdx.x;
  const uint32_t* pa = a + t * (NELEM / 4);
  const uint32_t* pb = b + t * (NELEM / 16);
#pragma unroll
  for (int i = 0; i < NELEM / 4; ++i) {
    uint32_t w = pa[i];
    uint32_t bb = (pb[i / 4] >> ((i & 3) * 8)) & 0xFFu;   // байт с четырьмя полями по 2 бита
    // два значения в слове: младшие 10 бит каждой половины = (u8<<2)|b2
    uint32_t lo = ((w & 0x00FF00FFu) << 2) | 0x64006400u;
    uint32_t hi = ((w & 0xFF00FF00u) >> 6) | 0x64006400u;
    uint32_t c0 = (bb & 0x3u) | ((bb & 0xCu) << 14);
    uint32_t c1 = ((bb >> 4) & 0x3u) | ((bb & 0xC0u) << 10);
    sm[t * (NELEM / 2) + 2 * i + 0] = lo | c0;
    sm[t * (NELEM / 2) + 2 * i + 1] = hi | c1;
  }
  __syncthreads();
  if (t == 0) dst[blockIdx.x] = sm[blockIdx.y];
}

// -----------------------------------------------------------------------------------------
// F3. int10 ДЕШЁВЫЙ, НО НЕ ПРОГРЕССИВНЫЙ: план A = МЛАДШИЕ 8 бит, план B = старшие 2.
//     Тогда старшие 2 бита садятся в СТАРШИЙ байт вместе с магией (0x64+b2) -- значит сборка
//     остаётся байтовой (prmt), но план A в одиночку БЕССМЫСЛЕН (обрезан диапазон, не точность).
// -----------------------------------------------------------------------------------------
__global__ void f3_int10_bytewise(const uint32_t* __restrict__ a,
                                  const uint32_t* __restrict__ b, uint32_t* dst) {
  const int t = threadIdx.x;
  const uint32_t* pa = a + t * (NELEM / 4);
  const uint32_t* pb = b + t * (NELEM / 16);
#pragma unroll
  for (int i = 0; i < NELEM / 4; ++i) {
    uint32_t w = pa[i];
    uint32_t bb = (pb[i / 4] >> ((i & 3) * 8)) & 0xFFu;
    // разложить четыре двухбитных поля по БАЙТАМ и прибавить к магии
    uint32_t sp = ((bb & 0x03u)) | ((bb & 0x0Cu) << 6) | ((bb & 0x30u) << 12) | ((bb & 0xC0u) << 18);
    uint32_t magic = 0x64646464u + sp;
    sm[t * (NELEM / 2) + 2 * i + 0] = prmt(w, magic, 0x4140u);
    sm[t * (NELEM / 2) + 2 * i + 1] = prmt(w, magic, 0x4342u);
  }
  __syncthreads();
  if (t == 0) dst[blockIdx.x] = sm[blockIdx.y];
}

// -----------------------------------------------------------------------------------------
// F4. int4: 8 значений в слове, поле 4 бита -> сдвиг + LOP3 (маска и ИЛИ сливаются)
// -----------------------------------------------------------------------------------------
__global__ void f4_int4(const uint32_t* __restrict__ src, uint32_t* dst) {
  const int t = threadIdx.x;
  const uint32_t* p = src + t * (NELEM / 8);
#pragma unroll
  for (int i = 0; i < NELEM / 8; ++i) {
    uint32_t w = p[i];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      sm[t * (NELEM / 2) + 4 * i + j] = ((w >> (4 * j)) & 0x000F000Fu) | 0x64006400u;
  }
  __syncthreads();
  if (t == 0) dst[blockIdx.x] = sm[blockIdx.y];
}

// -----------------------------------------------------------------------------------------
// F5. int3: 10 значений в 32-битном слове (3.2 бита на вес с учётом отхода)
// -----------------------------------------------------------------------------------------
__global__ void f5_int3(const uint32_t* __restrict__ src, uint32_t* dst) {
  const int t = threadIdx.x;
  const uint32_t* p = src + t * (NELEM / 10 + 1);
#pragma unroll
  for (int i = 0; i < NELEM / 10; ++i) {
    uint32_t w = p[i];
#pragma unroll
    for (int j = 0; j < 5; ++j) {
      uint32_t v0 = (w >> (6 * j)) & 0x7u;
      uint32_t v1 = (w >> (6 * j + 3)) & 0x7u;
      sm[t * (NELEM / 2) + 5 * i + j] = 0x64006400u | v0 | (v1 << 16);
    }
  }
  __syncthreads();
  if (t == 0) dst[blockIdx.x] = sm[blockIdx.y];
}

// -----------------------------------------------------------------------------------------
// F6. int6: 5 значений в слове (6.4 бита с учётом отхода)
// -----------------------------------------------------------------------------------------
__global__ void f6_int6(const uint32_t* __restrict__ src, uint32_t* dst) {
  const int t = threadIdx.x;
  const uint32_t* p = src + t * (NELEM / 5 + 1);
#pragma unroll
  for (int i = 0; i < NELEM / 5; ++i) {
    uint32_t w = p[i];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      uint32_t v0 = (w >> (12 * j)) & 0x3Fu;
      uint32_t v1 = (w >> (12 * j + 6)) & 0x3Fu;
      sm[t * (NELEM / 2) + 2 * i + j] = 0x64006400u | v0 | (v1 << 16);
    }
  }
  __syncthreads();
  if (t == 0) dst[blockIdx.x] = sm[blockIdx.y];
}

// -----------------------------------------------------------------------------------------
// F7. НЕРАВНОМЕРНЫЙ СЛОВАРЬ (nf4/Ллойд): индекс -- САМИ ДАННЫЕ, значит адрес расходящийся.
//     Инструкций мало, но каждая LDS -- обращение к ТОМУ ЖЕ ресурсу, который в этом ядре
//     ЗАМЕРЕННО связывает (пропускная разделяемой). Счёт инструкций тут НЕ вся цена.
// -----------------------------------------------------------------------------------------
__global__ void f7_lut4(const uint32_t* __restrict__ src, uint32_t* dst,
                        const __half* __restrict__ cb) {
  const int t = threadIdx.x;
  __shared__ __half tab[16];
  if (t < 16) tab[t] = cb[t];
  __syncthreads();
  const uint32_t* p = src + t * (NELEM / 8);
#pragma unroll
  for (int i = 0; i < NELEM / 8; ++i) {
    uint32_t w = p[i];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t a = __half_as_ushort(tab[(w >> (8 * j)) & 0xFu]);
      uint32_t b = __half_as_ushort(tab[(w >> (8 * j + 4)) & 0xFu]);
      sm[t * (NELEM / 2) + 4 * i + j] = a | (b << 16);
    }
  }
  __syncthreads();
  if (t == 0) dst[blockIdx.x] = sm[blockIdx.y];
}

// -----------------------------------------------------------------------------------------
// F8. int5: 6 значений в слове (5.33 бита с учётом отхода)
// -----------------------------------------------------------------------------------------
__global__ void f8_int5(const uint32_t* __restrict__ src, uint32_t* dst) {
  const int t = threadIdx.x;
  const uint32_t* p = src + t * (NELEM / 6 + 1);
#pragma unroll
  for (int i = 0; i < NELEM / 6; ++i) {
    uint32_t w = p[i];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      uint32_t v0 = (w >> (10 * j)) & 0x1Fu;
      uint32_t v1 = (w >> (10 * j + 5)) & 0x1Fu;
      sm[t * (NELEM / 2) + 3 * i + j] = 0x64006400u | v0 | (v1 << 16);
    }
  }
  __syncthreads();
  if (t == 0) dst[blockIdx.x] = sm[blockIdx.y];
}

// -----------------------------------------------------------------------------------------
// F9. int9: 3 значения в слове (10.67 бита с учётом отхода -- дороже int10 по 3 в слове!)
// -----------------------------------------------------------------------------------------
__global__ void f9_int9(const uint32_t* __restrict__ src, uint32_t* dst) {
  const int t = threadIdx.x;
  const uint32_t* p = src + t * (NELEM / 3 + 1);
#pragma unroll
  for (int i = 0; i < NELEM / 3; ++i) {
    uint32_t w = p[i];
    uint32_t v0 = (w >> 0) & 0x1FFu;
    uint32_t v1 = (w >> 9) & 0x1FFu;
    uint32_t v2 = (w >> 18) & 0x1FFu;
    sm[t * (NELEM / 2) + 2 * i + 0] = 0x64006400u | v0 | (v1 << 16);
    sm[t * (NELEM / 2) + 2 * i + 1] = 0x64006400u | v2;
  }
  __syncthreads();
  if (t == 0) dst[blockIdx.x] = sm[blockIdx.y];
}
