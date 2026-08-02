# SPDX-License-Identifier: LicenseRef-TRL-1.0
# Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
# Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
"""Диспетчер поставки: (op, формы, dtype, arch) -> идентификатор ядра либо None.

None означает "штатный путь" (F.linear -> cuBLAS). Диспетчер зовётся ОДИН раз при загрузке.

ПОЧЕМУ ДИСПЕТЧЕР ОБЯЗАТЕЛЕН, А НЕ УДОБЕН (замерено на 35 боевых точках):
  лучшая ЕДИНАЯ гиперформа на все точки .... геосреднее 0.746 к cuBLAS
  выбор по полосе M (эта таблица) .......... геосреднее 0.885
Разница 0.139 -- это и есть цена отказа от диспетчера; она больше всего, что удалось выиграть
внутри одной гиперформы.
"""

# FPREF СТОИТ ПОСТРОЧНО, А НЕ В ОБЩИХ. Пока он был общим (=2), отгружалось НЕ ТО, ЧТО МЕРИЛИ:
# замеренные строки скелета (`configs.inc`, тот же макрос CFG, которым снимали сетку) дают
# FPREF=1 у двух нижних ступеней лестницы, а отгружалось 2. На этих двух ступенях стоят ТРИ
# из семи заявленных побед (gate,up M=1 1.218, M=8 1.211, M=32 1.266), то есть паспорт
# описывал не тот бинарь. Значения приведены к ЗАМЕРУ; числа отчёта требуют ПЕРЕМЕРА
# (отдельный шаг, нужна карта -- см. docs/NOT_YET.md).
#
# (M_max, tag, BM, BN, BK, WM, WN, MINB, FPREF) -- порог по M включительно
LADDER = [
    (16, "m16x128k64", 16, 128, 64, 1, 4, 2, 1),
    (64, "m32x128k64", 32, 128, 64, 2, 4, 2, 1),
    (256, "m64x128k64", 64, 128, 64, 2, 4, 2, 2),
]
WIDE = ("m128x256k32", 128, 256, 32, 2, 4, 1, 2)  # M > 256 и N % 256 == 0
NARROW = ("m128x128k32", 128, 128, 32, 2, 2, 2, 2)  # M > 256, иначе

# Общие для всех: STAGES=2, GSTAGE=1, GROUP=8, EPI=0, SWZ=2 (фазово-инъективный), PRED=1
COMMON = dict(STAGES=2, GSTAGE=1, GROUP=8, EPI=0, SWZ=2, PRED=1)


def select(op, M, N, K, dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", arch="sm_70"):
    if op != "gemm" or arch != "sm_70":
        return None
    if (dtype_a, dtype_b, dtype_c) != ("fp16", "fp16", "fp16"):
        return None
    for m_max, tag, BM, BN, BK, WM, WN, MINB, FPREF in LADDER:
        if M <= m_max and N % BN == 0 and K % BK == 0:
            return dict(
                tag=tag,
                BM=BM,
                BN=BN,
                BK=BK,
                WM=WM,
                WN=WN,
                MINB=MINB,
                FPREF=FPREF,
                **COMMON,
            )
    tag, BM, BN, BK, WM, WN, MINB, FPREF = WIDE if N % 256 == 0 else NARROW
    if N % BN or K % BK:
        return None
    return dict(
        tag=tag, BM=BM, BN=BN, BK=BK, WM=WM, WN=WN, MINB=MINB, FPREF=FPREF, **COMMON
    )
