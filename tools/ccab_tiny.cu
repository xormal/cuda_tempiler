// микротест разбора: два ядра -- одно с объявленным бюджетом (разольёт), одно без.
__global__ void __launch_bounds__(256, 8) k_tight(const float* __restrict__ a, float* out, int n) {
  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = a[threadIdx.x + i * 32];
  for (int it = 0; it < n; ++it) {
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[i] = acc[i] * acc[(i + 7) & 63] + a[it + i];
  }
  float s = 0;
#pragma unroll
  for (int i = 0; i < 64; ++i) s += acc[i];
  out[threadIdx.x] = s;
}
__global__ void k_free(const float* __restrict__ a, float* out, int n) {
  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = a[threadIdx.x + i * 32];
  for (int it = 0; it < n; ++it) {
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[i] = acc[i] * acc[(i + 7) & 63] + a[it + i];
  }
  float s = 0;
#pragma unroll
  for (int i = 0; i < 64; ++i) s += acc[i];
  out[threadIdx.x] = s;
}
