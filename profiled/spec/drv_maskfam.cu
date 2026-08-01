// Проверка ТОГО ЖЕ раскрытия семейства, что делает диспетчер: 32 маски x causal/не-causal.
#include "volta_fwd_ws.cuh"
using ws_kern_t = void (*)(const __half*, const int8_t*, const float*, const int8_t*, const float*,
                           __half*, float*, int, int, float, int, int, long, long, long, long,
                           long, long, long, long, long, long, long, long, long, long long*, int*);
template <int M> inline ws_kern_t ws_mask_kern(bool causal) {
    return causal ? &fa2_sm70::fwd_ws::volta_fwd_ws<32, 64, 256, 2, 4, true,  64, 1, 1, 256, 8, 0, 0, 64 + M>
                  : &fa2_sm70::fwd_ws::volta_fwd_ws<32, 64, 256, 2, 4, false, 64, 1, 1, 256, 8, 0, 0, 64 + M>;
}
template <int M> inline ws_kern_t ws_mask_pick(int m, bool causal) {
    if (m == M) return ws_mask_kern<M>(causal);
    if constexpr (M > 0) return ws_mask_pick<M - 1>(m, causal);
    else return nullptr;
}
__host__ ws_kern_t probe(int m, bool c) { return ws_mask_pick<31>(m, c); }
