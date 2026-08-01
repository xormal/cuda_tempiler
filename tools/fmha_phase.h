// -*- coding: utf-8 -*-
// РАЗМЕТКА ФАЗ ЯДРА ДЛЯ ФАЗОВОГО ПРОФИЛИРОВЩИКА (tools/phaseprof.py).
//
// ЗАЧЕМ. Разложение ядра по фазам у нас делалось РУКАМИ: программист писал #ifdef, собирал
// ЗАВЕДОМО НЕВЕРНОЕ ядро со снятой фазой, мерил время, считал долю. Так получены 35.1 / 19.7 /
// 15.2 / 5.7 / 2.0 % у байтового форварда. Стоит человеко-часов на фазу -> делается раз в полгода.
// Здесь разметка сводится к одной строке, а сборку вариантов, замер и (главное) СВЕДЕНИЕ БАЛАНСА
// делает инструмент.
//
// -------------------------------------------------------------------------------------------
// КАК РАЗМЕЧАТЬ
//
//     #include "fmha_phase.h"
//
//     FMHA_PHASE(gemm1, 0) {          // id 0 -> бит 0 маски FMHA_STRIP_MASK
//         mma.accumulate(...);
//     }
//     FMHA_SEAL(acc[0], acc[1], acc[2], acc[3]);   // ПЛОМБА: см. ниже, без неё замер врёт
//
// Боевая сборка (FMHA_STRIP_MASK не задан) разворачивает тело как есть -- ноль накладных.
// Сборка `-DFMHA_STRIP_MASK=1` выбрасывает фазу с id 0. `-DFMHA_STRIP_MASK=5` -- фазы 0 и 2
// СРАЗУ ОБЕ (это нужно для замера ПЕРЕКРЫТИЯ пары, см. README, раздел про невязку).
//
// -------------------------------------------------------------------------------------------
// ПЛОМБА (FMHA_SEAL) -- ГЛАВНАЯ ТОНКОСТЬ, БЕЗ КОТОРОЙ ИНСТРУМЕНТ ЛЖЁТ
//
// Снятая фаза перестаёт задавать свои выходы. Компилятор видит константу (или неинициализированное
// значение) и выбрасывает ВСЁ, что от неё зависело, -- то есть куски ДРУГИХ фаз. Замер тогда
// приписывает снятой фазе чужую работу, и доля завышена. Это не теория: у любого softmax-подобного
// кода снятие первого умножения обнуляет S -> максимум сворачивается в константу -> exp2 сворачивается
// -> второе умножение получает константный P.
//
// FMHA_SEAL(x...) отмывает значения тождественной перестановкой байт `prmt.b32 d,a,a,0x3210`
// (селектор 0x3210 берёт байты 0,1,2,3 первого источника, то есть возвращает его как есть).
// Стоит РОВНО ОДНУ команду SASS на значение и НЕ меняет бит: проверено на карте на 65536 словах,
// включая +-0, денормали, inf, NaN и sNaN -- расхождений 0 (data/phase_seal_sass.txt).
//
// >>> ГЛАВНАЯ ЛОВУШКА, НА КОТОРОЙ ЭТОТ ЗАГОЛОВОК УЖЕ ОДИН РАЗ СОВРАЛ <<<
// Общепринятая пустая вставка `asm volatile("" : "+f"(x))` ЗДЕСЬ НЕ РАБОТАЕТ. NVVM её уважает --
// в PTX видно, что зависимый `ex2.approx.f32` СОХРАНЁН, -- а ptxas сворачивает сквозь неё, потому
// что тело пустое и разрывать в SASS нечего. Замерено: ядро с пустой пломбой и без пломбы дают
// БАЙТ В БАЙТ одинаковый SASS (16 команд, значение свёрнуто в константу `MOV R5, 0x403504f2`).
// Отсюда правило, которое стоит помнить шире этого заголовка: **проверка барьера по PTX даёт
// ЛОЖНОЕ ПОДТВЕРЖДЕНИЕ; барьер проверяется только по SASS.** Пломба обязана нести настоящую
// команду -- поэтому prmt, а не "".
//
// Чего пломба НЕ умеет:
//   * значение, которое надо сохранить ЦЕЛЫМ массивом в разделяемой памяти -- пломбируйте по
//     элементам через FMHA_SEAL_ARR или ставьте FMHA_SINK;
//   * побочные эффекты (запись в глобальную), которые сами по себе никем не читаются, --
//     для них FMHA_SINK(ptr, idx, val), он ЗАСТАВЛЯЕТ запись состояться;
//   * барьеры: __syncthreads() снятой фазы исчезает вместе с ней. Если фаза внутри цикла с
//     рандеву, снимайте ТЕЛО, а рандеву оставляйте СНАРУЖИ FMHA_PHASE -- иначе меряется не фаза,
//     а фаза плюс защёлка.
//
// -------------------------------------------------------------------------------------------
// ЧТО СЧИТАЕТСЯ ПРАВИЛЬНОЙ РАЗМЕТКОЙ (проверяет `phaseprof.py scan`)
//   * id уникальны и идут подряд с нуля (иначе маска пары строится неверно);
//   * у каждой фазы есть хотя бы одна пломба в той же функции, иначе -- в раздел НЕ РАЗОБРАНО;
//   * фазы не вложены друг в друга (вложенную инструмент отнесёт к ВНУТРЕННЕЙ и скажет об этом).

#pragma once

#ifndef FMHA_STRIP_MASK
#define FMHA_STRIP_MASK 0u
#endif

// Максимум фаз на ядро = разрядность маски.
#define FMHA_PHASE_MAX_ID 31

#if defined(__cplusplus) && (__cplusplus >= 201703L || defined(__CUDACC__))
// if constexpr: ветка не порождает кода даже при -O0 и не даёт предупреждений о недостижимости.
#define FMHA_PHASE_IF(cond) if constexpr (cond)
#else
#define FMHA_PHASE_IF(cond) if (cond)
#endif

// ФАЗА. Тело в фигурных скобках сразу за макросом.
#define FMHA_PHASE(name, id) FMHA_PHASE_IF(!((FMHA_STRIP_MASK) & (1u << (id))))

// Подстановка на случай снятой фазы (необязательна; чаще достаточно пломбы).
#define FMHA_PHASE_ELSE(name, id) else

// Явный вопрос "фаза на месте?" для условий вне блочной разметки.
#define FMHA_PHASE_ON(id) (!((FMHA_STRIP_MASK) & (1u << (id))))

#if defined(__CUDACC__)

// ---- ПЛОМБЫ -------------------------------------------------------------------------------
// Одна команда SASS на 32-битное слово, тождественная побитово (проверено на карте).
#define FMHA_SEAL_W32(w) asm volatile("prmt.b32 %0, %0, %0, 0x3210;" : "+r"(w))

__device__ __forceinline__ void fmha_seal_bits(unsigned& w) { FMHA_SEAL_W32(w); }

__device__ __forceinline__ void fmha_seal_one(unsigned& x) { FMHA_SEAL_W32(x); }
__device__ __forceinline__ void fmha_seal_one(float& x) {
  FMHA_SEAL_W32(*reinterpret_cast<unsigned*>(&x));
}
__device__ __forceinline__ void fmha_seal_one(int& x) {
  FMHA_SEAL_W32(*reinterpret_cast<unsigned*>(&x));
}
__device__ __forceinline__ void fmha_seal_one(__half2& x) {
  FMHA_SEAL_W32(*reinterpret_cast<unsigned*>(&x));
}
// 64-битные -- ДВЕ команды, по слову на половину: prmt 32-разрядный, а `mov.b64 %0,%0` ptxas
// сворачивает ровно так же, как пустую вставку.
__device__ __forceinline__ void fmha_seal_one(double& x) {
  unsigned* w = reinterpret_cast<unsigned*>(&x);
  FMHA_SEAL_W32(w[0]);
  FMHA_SEAL_W32(w[1]);
}
__device__ __forceinline__ void fmha_seal_one(long long& x) {
  unsigned* w = reinterpret_cast<unsigned*>(&x);
  FMHA_SEAL_W32(w[0]);
  FMHA_SEAL_W32(w[1]);
}
__device__ __forceinline__ void fmha_seal_one(unsigned long long& x) {
  fmha_seal_one(*reinterpret_cast<long long*>(&x));
}
// 16-битное значение расширяем до слова: prmt.b16 в ISA нет. Стоит 1 команду плюс, возможно,
// пару перемещений -- для 16-битных пломба НЕ бесплатна, и инструмент это покажет в цене пломбы.
__device__ __forceinline__ void fmha_seal_one(__half& x) {
  unsigned t = 0;
  *reinterpret_cast<unsigned short*>(&t) = *reinterpret_cast<unsigned short*>(&x);
  FMHA_SEAL_W32(t);
  *reinterpret_cast<unsigned short*>(&x) = *reinterpret_cast<unsigned short*>(&t);
}
__device__ __forceinline__ void fmha_seal_one(unsigned short& x) {
  fmha_seal_one(*reinterpret_cast<__half*>(&x));
}
__device__ __forceinline__ void fmha_seal_one(short& x) {
  fmha_seal_one(*reinterpret_cast<__half*>(&x));
}

__device__ __forceinline__ void fmha_seal_all() {}
template <class T, class... R>
__device__ __forceinline__ void fmha_seal_all(T& a, R&... rest) {
  fmha_seal_one(a);
  fmha_seal_all(rest...);
}

#define FMHA_SEAL(...) ::fmha_seal_all(__VA_ARGS__)

#define FMHA_SEAL_ARR(a, n)                     \
  do {                                          \
    _Pragma("unroll") for (int _i = 0; _i < (n); ++_i) ::fmha_seal_one((a)[_i]); \
  } while (0)

// СТОК: заставляет запись состояться, даже если её никто не читает. Стоит одну команду сравнения
// и (никогда не исполняемую) запись -- применять там, где пломба неприменима.
extern "C" {
__device__ __constant__ int fmha_phase_never;   // хост его не пишет, значение 0
}
#define FMHA_SINK(ptr, idx, val)               \
  do {                                          \
    if (fmha_phase_never) (ptr)[idx] = (val);   \
  } while (0)

#else   // не CUDA -- заглушки, чтобы заголовок читался хостовым компилятором
#define FMHA_SEAL(...) ((void)0)
#define FMHA_SEAL_ARR(a, n) ((void)0)
#define FMHA_SINK(ptr, idx, val) ((void)0)
#endif
