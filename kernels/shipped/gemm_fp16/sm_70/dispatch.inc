// SPDX-License-Identifier: LicenseRef-TRL-1.0
// Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
// Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
//
// ПОРОЖДЕНО из select.py -- РУКАМИ НЕ ПРАВИТЬ. Правится select.py, потом перегенерация:
//   python3 kernels/shipped/gemm_fp16/sm_70/gen_dispatch.py > dispatch.inc
//
// ЛЕСТНИЦА ПО M -- НЕ УДОБСТВО, А ЗАМЕР: лучшая ЕДИНАЯ гиперформа на все 35 боевых точек
// даёт геосреднее 0.746 к сопернику, выбор по полосе M -- 0.885. Разница 0.139 больше
// всего, что удалось выиграть ВНУТРИ одной гиперформы.
//
// TEMPO_FALLBACK: любая форма, не покрытая строками ниже, обязана уйти на штатный путь
// вызывающего. Возврат cudaErrorInvalidConfiguration -- ЗАКОННЫЙ ответ, а не сбой.

  if (p.M <= 16 && (p.N % 128) == 0 && (p.K % 64) == 0)
    return launch_one<16,128,64,1,4,2,1,2,8,0,2,true,2>(p, s);  // m16x128k64
  if (p.M <= 64 && (p.N % 128) == 0 && (p.K % 64) == 0)
    return launch_one<32,128,64,2,4,2,1,2,8,0,2,true,2>(p, s);  // m32x128k64
  if (p.M <= 256 && (p.N % 128) == 0 && (p.K % 64) == 0)
    return launch_one<64,128,64,2,4,2,1,2,8,0,2,true,2>(p, s);  // m64x128k64
  if ((p.N % 256) == 0 && (p.K % 32) == 0)
    return launch_one<128,256,32,2,4,2,1,2,8,0,2,true,1>(p, s);  // m128x256k32  (M > 256, N кратно 256)
  if ((p.N % 128) == 0 && (p.K % 32) == 0)
    return launch_one<128,128,32,2,2,2,1,2,8,0,2,true,2>(p, s);  // m128x128k32  (M > 256, узкое N)

  return cudaErrorInvalidConfiguration;  // TEMPO_FALLBACK
