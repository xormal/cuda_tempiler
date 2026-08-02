// SPDX-License-Identifier: LicenseRef-TRL-1.0
// Copyright (c) 2026 Alexander Romanov <alex.xorm@gmail.com>
// Tempo Research License v1.0 -- see LICENSE. Commercial use by agreement.
//
// ЕДИНСТВЕННАЯ ЕДИНИЦА ТРАНСЛЯЦИИ ПОСТАВКИ. Здесь и только здесь порождённая таблица
// `dispatch.inc` попадает в тело диспетчера -- то есть здесь инстанцируются ядра.
//
// ЗАЧЕМ ОТДЕЛЬНЫЙ .cu, А НЕ ВСЁ В ЗАГОЛОВКЕ: заголовок включают в несколько единиц
// трансляции, и инстанциации ядер размножились бы по каждой. Одна единица -- один набор.
//
// ЧЕМ ЭТОГО ФАЙЛА НЕ ХВАТАЛО: `launch.cuh` объявлял `tempo::gen::launch`, а определения не
// было НИГДЕ, и `dispatch.inc` не включался ни в один `.cu`. Принимающее дерево получало
// ошибку компоновки. Замеры при этом существовали -- они снимались испытательной обвязкой,
// которая инстанцирует ядра сама и поставляемый путь не трогает.
#include "launch.cuh"

namespace tempo {
namespace gen {

cudaError_t launch(const GemmParams& p, cudaStream_t s) {
#include "dispatch.inc"
}

}  // namespace gen
}  // namespace tempo
