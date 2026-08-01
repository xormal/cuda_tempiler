// -*- coding: utf-8 -*-
// ФАЗОВАЯ РАЗМЕТКА BACKWARD (attention_kernel_backward_batched_impl) -- ОСНАСТКА, НЕ БОЕВОЙ КОД.
//
// ЗАЧЕМ. Разложение форварда даёт в сумме 77.7 %, а 22.3 не объяснены, и РАЗДЕЛИТЬ остаток на
// ПЕРЕКРЫТИЕ и НЕНАЗВАННОЕ одним числом (DIAG/nosc) нельзя: одним числом не снять ДВЕ фазы разом.
// Здесь фаза снимается БИТОМ маски FMHA_STRIP_MASK, поэтому доступны пары (e_ij = s_ij - s_i - s_j)
// и вариант "снять все" (s_all), то есть тождество
//     1 = SUM(s_i) + (s_all - SUM(s_i)) + (1 - s_all).
// Формат разметки и spec-файла -- tools/fmha_phase.h + tools/phaseprof.py в ..
//
// ---------------------------------------------------------------------------------------------
// ГЛАВНОЕ ПРАВИЛО ЭТОГО ФАЙЛА: НУЛЕВАЯ ЦЕНА НА БОЕВОМ ПУТИ.
// Боевая сборка НЕ задаёт ни FMHA_STRIP_MASK, ни FMHA_BWD_PHASE_SEAL. Тогда:
//   * FMHA_PHASE(...)      -> `if constexpr (true)`      (тело как было, только своя область);
//   * FMHA_SEAL/SEAL_ARR   -> `((void)0)`                (ни одной команды);
//   * FMHA_SINK            -> `((void)0)`                (нет и __constant__-символа);
//   * шаблонные ключи фаз у fused_qk_gradk -> оба true   (тот же инстанс).
// Проверяется компиляционным A/B (cuobjdump -sass побайтово + -Xptxas -v: регистры/кадр/разлив).
// Ядро d=128 стоит РОВНО на 255 регистрах при нулевом разливе -- любой лишний живой регистр
// столкнёт его в разлив, поэтому A/B здесь условие работоспособности, а не формальность.
//
// ---------------------------------------------------------------------------------------------
// КАК ПЕРЕДАЁТСЯ МАСКА (и почему она ни с чем не сталкивается)
// Backward НЕ имеет рантайм-диспетчера вариантов: коды `nosc`/`DIAG` (var 11..15 -> DIAG 1..5,
// var 20 = PHSPLIT, var 30 = KSP) живут в attn_fwd_cutlass.cu и относятся к ФОРВАРДУ. Варианты
// backward выбираются макросами сборки (-DFMHA_BWD_R7, -DFMHA_BWD_COAL_FO, ...). Поэтому маска --
// тоже макрос сборки, `-DFMHA_STRIP_MASK=<биты>`, и с числовым пространством nosc не пересекается
// в принципе (разные единицы трансляции, разный механизм).
// Занятые биты: 0..10 (см. FMHA_PH_* ниже). Свободны 11..31.
//
// ---------------------------------------------------------------------------------------------
// ПЛОМБА И СТОК: ЧТО ОТ ЧЕГО ЗАЩИЩАЮТ
//   ПЛОМБА (DSEAL/DSEAL_ARR) стоит на ВЫХОДЕ фазы: снятая фаза не задаёт свой выход, компилятор
//   видит константу и сворачивает ВНИЗ по течению (у нас: обнулённый S -> exp2 в константу ->
//   всё второе умножение). prmt.b32 d,a,a,0x3210 -- одна команда, побитово тождественная.
//   СТОК (DSINK_ARR) стоит в ПОДСТАНОВКЕ (FMHA_PHASE_ELSE) и защищает ВВЕРХ по течению: у снятой
//   фазы пропадает единственный потребитель её входа, и компилятор выбрасывает ЧУЖУЮ фазу.
//   Пример из этого ядра: снять DSSTORE -> у dS нет потребителя -> исчезает и DS, и DOIVJ.
// Обе включаются только в профилирующей сборке.
#pragma once

// Профилирующая сборка опознаётся по любому из двух ключей командной строки.
#if defined(FMHA_STRIP_MASK) || defined(FMHA_BWD_PHASE_SEAL)
#define FMHA_BWD_PHASE_ACTIVE 1
#else
#define FMHA_BWD_PHASE_ACTIVE 0
#endif

// Пломбы/стоки по умолчанию включаются ровно тогда, когда маска НЕ нулевая: сборка с маской 0
// обязана быть побайтово боевой, а сборка со снятой фазой без пломбы -- ложью. Инструмент задаёт
// -DFMHA_BWD_PHASE_SEAL=1 ВСЕМ вариантам (включая базу), чтобы пломба сокращалась из отношения
// времён, и -DFMHA_BWD_PHASE_SEAL=0 -- чтобы измерить её цену.
#if !defined(FMHA_BWD_PHASE_SEAL)
#if defined(FMHA_STRIP_MASK) && ((FMHA_STRIP_MASK) != 0)
#define FMHA_BWD_PHASE_SEAL 1
#else
#define FMHA_BWD_PHASE_SEAL 0
#endif
#endif

#if FMHA_BWD_PHASE_ACTIVE
#include "fmha_phase.h"
#else
// Боевые заглушки с теми же именами: ни одной команды и ни одного символа.
#ifndef FMHA_STRIP_MASK
#define FMHA_STRIP_MASK 0u
#endif
#define FMHA_PHASE(name, id) if constexpr (true)
#define FMHA_PHASE_ELSE(name, id) else
#define FMHA_PHASE_ON(id) true
#define FMHA_SEAL(...) ((void)0)
#define FMHA_SEAL_ARR(a, n) ((void)0)
#define FMHA_SINK(ptr, idx, val) ((void)0)
#endif

// ---- НОМЕРА ФАЗ ------------------------------------------------------------------------------
// Порядок = порядок в расписании плитки. Имена совпадают с именами в FMHA_PHASE.
#define FMHA_PH_QK 0      // S = K @ Q^T  (главный мейнлуп, включая подачу K/Q)
#define FMHA_PH_SOFTMAX 1 // scale, причинная маска, exp(S-LSE), укладка P^T в разделяемую
#define FMHA_PH_GRADV 2   // dV += P^T @ dO
#define FMHA_PH_DOIVJ 3   // dP = dO @ V^T
#define FMHA_PH_DS 4      // dS = (dP - Delta) * P * scale  -- поэлементно в регистрах
#define FMHA_PH_DSSTORE 5 // материализация dS/dS^T в разделяемую (tmpT / r7_sT / accumToSmem)
#define FMHA_PH_GRADQ 6   // dQ-умножение (R7 whole-B k^T@S^T либо колоночный lock-RMW dS@K)
#define FMHA_PH_DQFOLD 7  // свёртка dQ наружу (fragment-order RED / запись в площадку + эпилог)
#define FMHA_PH_GRADK 8   // dK += dS^T @ Q (слитно в QK / отложенно / в плитке)
#define FMHA_PH_KVOUT 9   // круг накопителей dK/dV: writeFragsToGmem + accumulateInGmem/store
#define FMHA_PH_DELTA 10  // предпроход Delta = (dO*O).sum(-1)

// ---- НАШИ ОБЁРТКИ ----------------------------------------------------------------------------
// (phaseprof ищет пломбу по ЛЮБОМУ имени, содержащему SEAL, поэтому имена ниже он видит.)
#if FMHA_BWD_PHASE_SEAL
// Явное приведение к типу приёмника: накопители у нас fp32, а приёмники стока -- half_t, и неявного
// присваивания между ними нет (сборка гейтится __CUDA_NO_HALF_CONVERSIONS__).
template <class T, class U>
__device__ __forceinline__ T fmha_sink_cast(T*, U v) {
  return T(v);
}
#define DSEAL(...) FMHA_SEAL(__VA_ARGS__)
#define DSEAL_ARR(a, n) FMHA_SEAL_ARR(a, n)
// СТОК массива: заставляет значения дожить до этой точки, не порождая исполняемой записи
// (fmha_phase_never лежит в постоянной памяти, хост его не пишет, но свернуть его компилятор
// не вправе). Ставится ТОЛЬКО в подстановке снятой фазы.
// (_Pragma, а не CUTLASS_PRAGMA_UNROLL: тот раскрывается в `#pragma`, что внутри макроса
//  недопустимо -- "# not expected here".)
#define DSINK_ARR(ptr, a, n)                                                   \
  do {                                                                         \
    _Pragma("unroll")                                                          \
    for (int _si = 0; _si < (n); ++_si) {                                      \
      FMHA_SINK(ptr, _si, ::fmha_sink_cast(ptr, (a)[_si]));                    \
    }                                                                          \
  } while (0)
#else
#define DSEAL(...) ((void)0)
#define DSEAL_ARR(a, n) ((void)0)
#define DSINK_ARR(ptr, a, n) ((void)0)
#endif
