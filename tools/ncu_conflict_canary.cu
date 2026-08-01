// СИНТЕТИЧЕСКИЙ ЯКОРЬ для tools/ncu.py: конфликтность банков, известная ИЗ АРИФМЕТИКИ, а не из
// прошлого замера. 32 банка по слову; шаг строки в СЛОВАХ решает всё.
//
//   ядро clean:   шаг 33 слова -> столбец ложится на 33 разных банка (mod 32 сдвигается на 1)
//                 ждём: вайвфронтов 1 на команду, идеал 1, ИЗБЫТОЧНЫХ 0, N-way 1
//   ядро dirty:   шаг 32 слова -> ВЕСЬ столбец в ОДИН банк
//                 ждём: вайвфронтов 32 на команду, идеал 1, ИЗБЫТОЧНЫХ 31, N-way 32
//
// Один блок в 32 полосы, один проход: числа получаются целыми и проверяются глазами.
// Сборка: nvcc -arch=sm_70 -lineinfo -o ncu_conflict_canary ncu_conflict_canary.cu
#include <cstdio>

__global__ void tempo_bank_clean(float* o) {
  __shared__ float s[32][33];              // ШАГ 33 СЛОВА
  int t = threadIdx.x;
  s[t][0] = t;
  __syncthreads();
  float v = s[t][0];                       // столбец: банк = (t*33) % 32 = t -> все разные
  o[t] = v;
}

__global__ void tempo_bank_dirty(float* o) {
  __shared__ float s[32][32];              // ШАГ 32 СЛОВА
  int t = threadIdx.x;
  s[t][0] = t;
  __syncthreads();
  float v = s[t][0];                       // столбец: банк = (t*32) % 32 = 0 -> ВСЕ в банк 0
  o[t] = v;
}

int main() {
  float* d;
  cudaMalloc(&d, 1024);
  tempo_bank_clean<<<1, 32>>>(d);
  tempo_bank_dirty<<<1, 32>>>(d);
  cudaDeviceSynchronize();
  printf("canary %d\n", (int)cudaGetLastError());
  return 0;
}
