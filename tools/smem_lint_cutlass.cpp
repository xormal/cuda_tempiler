// ЗОНД ПО НАСТОЯЩИМ РАСКЛАДКАМ CUTLASS -- ХОСТОВАЯ ПРОГРАММА, БЕЗ GPU И БЕЗ nvcc.
//
// ЗАЧЕМ ОН ОТДЕЛЬНО. Разбор исходника принципиально СЛЕП на cutlass-ядрах: индексного выражения там
// нет в тексте вовсе, адрес считает функтор раскладки (Layout::operator()). Поэтому линтер не
// угадывает свиззл, а ИНСТАНЦИРУЕТ настоящий cutlass::layout::* и спрашивает у него адреса. Все
// функторы помечены CUTLASS_HOST_DEVICE, то есть на хосте это обычные inline-функции: ни карты, ни
// nvcc, ни sudo не нужно.
//
// Печатает строки вида:  CASE <имя> <элемент_в_байтах> <ширина_на_полосу_в_байтах> ; далее 32 числа --
// смещение первого слова доступа каждой полосы В ЭЛЕМЕНТАХ (или -1, если полоса неактивна).
// Конфликтность считает python (tools/smem_lint.py), здесь только адреса.
//
// Сборка:  g++ -std=c++17 -I<cutlass>/include smem_lint_cutlass.cpp -o smem_lint_cutlass

#include <cstdio>
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"

static void emit(const char* name, int elem_bytes, int width_bytes, const int* off) {
  printf("CASE %s %d %d", name, elem_bytes, width_bytes);
  for (int i = 0; i < 32; ++i) printf(" %d", off[i]);
  printf("\n");
}

int main() {
  int off[32];

  // ---------------------------------------------------------------------------------------------
  // 1. B2bGemm<...>::accumToSmem для Volta, накопитель float (fa2_src/fmha_kernel/gemm/mma_from_smem.h
  //    строки ~1697-1775). Это ЗАПИСЬ P = softmax(QK^T) в разделяемую память между первым и вторым
  //    умножением: и в форварде (MM0::AccumulatorSharedStorage), и в бэкварде (MatmulQK).
  //    Полоса пишет Array<half,2> = ЧЕТЫРЕ БАЙТА в свиззлованную раскладку, рассчитанную на 128 бит.
  // ---------------------------------------------------------------------------------------------
  {
    using L = cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<16, 32>;
    L l = L::packed({32, 32});
    for (int lane = 0; lane < 32; ++lane) {
      int quad = lane >> 2, liq = lane & 3;
      // float-ветка: accum_m = (((quad&4)>>1)+(quad&1))*8 + (lane_in_quad&1)
      int r = (((quad & 4) >> 1) + (quad & 1)) * 8 + (liq & 1);
      int c = ((quad >> 1) & 1) * 4 * 2 + (liq & 2);   // kElementsPerPartial*kAccumulatorPatials = 8
      off[lane] = l({r, c});
    }
    emit("B2bGemm.accumToSmem.volta.f32accum", 2, 4, off);
  }

  // 2. Та же запись, но накопитель half (ветка else того же тела): полоса пишет Array<half,4> = 8 Б.
  {
    using L = cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<16, 32>;
    L l = L::packed({32, 32});
    for (int lane = 0; lane < 32; ++lane) {
      int quad = lane >> 2, liq = lane & 3;
      int r = (((quad & 4) >> 1) + (quad & 1)) * 8 + liq;
      int c = ((quad >> 1) & 1) * 4 * 2;
      off[lane] = l({r, c});
    }
    emit("B2bGemm.accumToSmem.volta.f16accum", 2, 8, off);
  }

  // ---------------------------------------------------------------------------------------------
  // 3-6. КОНТРОЛЬНЫЕ ОБХОДЫ по свиззлованным раскладкам операндов. ВАЖНАЯ ОГОВОРКА: наивный обход по
  //      строке/столбцу -- НЕ тот, которым ходит MmaVoltaTensorOpMultiplicandTileIterator. Эти случаи
  //      нужны, чтобы показать, ЧТО ИМЕННО делает свиззл, а вердикт «не трогать» выносится по ТИПУ
  //      раскладки (см. python-часть), а не по этим обходам.
  // ---------------------------------------------------------------------------------------------
  {
    using L = cutlass::layout::VoltaTensorOpMultiplicandCongruous<16>;
    L l = L::packed({64, 64});
    for (int lane = 0; lane < 32; ++lane) off[lane] = l({lane, 0});          // по столбцу
    emit("VoltaCongruous16.naive_col", 2, 16, off);
    for (int lane = 0; lane < 32; ++lane) off[lane] = l({0, lane * 8});      // по строке, по 8 полуслов
    emit("VoltaCongruous16.naive_row8", 2, 16, off);
  }
  {
    using L = cutlass::layout::VoltaTensorOpMultiplicandCrosswise<16, 32>;
    L l = L::packed({32, 32});
    for (int lane = 0; lane < 32; ++lane) off[lane] = l({lane, 0});
    emit("VoltaCrosswise16x32.naive_col", 2, 16, off);
    for (int lane = 0; lane < 32; ++lane) off[lane] = l({0, lane});
    emit("VoltaCrosswise16x32.naive_row", 2, 16, off);
  }

  // ---------------------------------------------------------------------------------------------
  // 7-8. ГОЛЫЕ RowMajor/ColumnMajor того же размера -- то, ЧТО БЫЛО БЫ без свиззла. Это контроль
  //      измерения: если бы у зонда конфликт получался всегда, он ничего бы не различал.
  // ---------------------------------------------------------------------------------------------
  {
    cutlass::layout::RowMajor l(32);
    for (int lane = 0; lane < 32; ++lane) off[lane] = l({lane, 0});
    emit("RowMajor32.col_walk", 2, 4, off);
    for (int lane = 0; lane < 32; ++lane) off[lane] = l({0, lane * 2});
    emit("RowMajor32.row_walk", 2, 4, off);
  }
  {
    cutlass::layout::RowMajor l(64);
    for (int lane = 0; lane < 32; ++lane) off[lane] = l({lane, 0});
    emit("RowMajor64.col_walk", 2, 4, off);
  }
  return 0;
}
