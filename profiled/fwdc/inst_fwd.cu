// Standalone compile-only instantiation of the shipped cutlass forward for phase marking A/B.
#include "kernel_forward.h"

// d<=128 production tile:  <32,128,128>  -> kSingleValueIteration=true  -> kKeepOutputInRF=TRUE
using AttnD128 = AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128,
                                 false, false, DefaultToBatchHook, /*kOutputBHSD*/true>;
// d<=256 production tile:  <32,128,256>  -> kSingleValueIteration=false -> kKeepOutputInRF=FALSE
using AttnD256 = AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 256,
                                 false, false, DefaultToBatchHook, /*kOutputBHSD*/true>;

template __global__ void __launch_bounds__(AttnD128::kNumThreads, AttnD128::kMinBlocksPerSm)
    attention_kernel_batched_impl<AttnD128>(typename AttnD128::Params p);
template __global__ void __launch_bounds__(AttnD256::kNumThreads, AttnD256::kMinBlocksPerSm)
    attention_kernel_batched_impl<AttnD256>(typename AttnD256::Params p);
