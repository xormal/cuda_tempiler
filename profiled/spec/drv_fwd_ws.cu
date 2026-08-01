// Инстанцирующий драйвер volta_fwd_ws для СТАТИЧЕСКОГО A/B по SASS (без торча и без запуска).
// WS_D -- четырнадцатый шаблонный параметр DIAG. Боевое значение 0; семейство маски -- 64+mask.
// Плитка -- БОЕВАЯ, та же, что раскрывает диспетчер при f==8: <32,64,256,2,4,causal,64,1,1,256,8>.
#include "volta_fwd_ws.cuh"
namespace fa2_sm70 { namespace fwd_ws {
#ifndef WS_D
#define WS_D 0
#endif
#ifndef WS_CAUSAL
#define WS_CAUSAL true
#endif
template __global__ void volta_fwd_ws<32, 64, 256, 2, 4, WS_CAUSAL, 64, 1, 1, 256, 8, 0, 0, WS_D, 0, 1>(
    const __half*, const int8_t*, const float*, const int8_t*, const float*,
    __half*, float*, int, int, float, int, int,
    long, long, long, long, long, long, long, long, long, long, long, long, long,
    long long*, int*);
}}
