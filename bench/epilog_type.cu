// Меняется ТОЛЬКО тип выхода cuBLAS. Всё остальное тождественно.
#include <cstdio>
#include <cuda_fp16.h>
#include <cublas_v2.h>
int main(){
  cublasHandle_t h; cublasCreate(&h); cublasSetMathMode(h,CUBLAS_TENSOR_OP_MATH);
  int SH[][3]={{4096,4096,4096},{8192,4096,3840},{8192,15360,3840},{8192,3840,15360}};
  for(auto&s:SH){int M=s[0],N=s[1],K=s[2];
    __half *A,*B; void*C; cudaMalloc(&A,(size_t)M*K*2);cudaMalloc(&B,(size_t)N*K*2);cudaMalloc(&C,(size_t)M*N*4);
    cudaMemset(A,0x11,(size_t)M*K*2);cudaMemset(B,0x11,(size_t)N*K*2);
    float al=1,be=0; double flop=2.0*M*N*K; cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);
    double tf[2];
    for(int t=0;t<2;++t){auto dt=t?CUDA_R_32F:CUDA_R_16F;
      auto go=[&]{cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&al,B,CUDA_R_16F,K,A,CUDA_R_16F,K,&be,C,dt,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);};
      go();cudaDeviceSynchronize();for(int i=0;i<3;++i)go();cudaDeviceSynchronize();
      cudaEventRecord(e0);for(int i=0;i<30;++i)go();cudaEventRecord(e1);cudaEventSynchronize(e1);
      float ms;cudaEventElapsedTime(&ms,e0,e1);tf[t]=flop/(ms/30*1e-3)/1e12;}
    printf("M%6d N%6d K%6d | выход fp16 %6.1f | выход fp32 %6.1f | fp32/fp16 x%.3f\n",M,N,K,tf[0],tf[1],tf[1]/tf[0]);
    cudaFree(A);cudaFree(B);cudaFree(C);}
  return 0;}
