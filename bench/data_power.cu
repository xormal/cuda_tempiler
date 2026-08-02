// ФАЛЬСИФИКАТОР ДАННЫХ: меняются ТОЛЬКО ЗНАЧЕНИЯ в операндах. Ядро, форма, тип выхода, число
// повторов -- тождественны. Частота и мощность опрашиваются ПЛОТНО (в отдельной нити, всё время
// исполнения), а не одним отсчётом: одиночный отсчёт NVML на V100 -- это не измерение.
//
// ЗАЧЕМ. Наш собственный стенд боевого дерева (tools/volta_gemm_bench.cu) заполняет операнды
// `cudaMemset(dA, 0x11, ...)`, то есть ОДНИМ И ТЕМ ЖЕ полусловом. Если время от этого зависит,
// все его абсолютные ТФЛОП/с (включая «потолок мейнлупа 120.96 = 96.8% пика» и «cuBLAS 97.0»)
// сняты не с той машины, что работает в бою.
#include <cstdio>
#include <vector>
#include <algorithm>
#include <thread>
#include <atomic>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <nvml.h>
__global__ void fill_rand(__half*p,long n,unsigned s){
  for(long i=(long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x){
    unsigned h=(unsigned)i*2654435761u+s;h^=h>>15;h*=2246822519u;h^=h>>13;
    p[i]=__float2half(((float)(h&0xFFFF)/65535.f-0.5f)*0.25f);}}
static double med(std::vector<double>v){if(v.empty())return 0;std::sort(v.begin(),v.end());return v[v.size()/2];}
int main(){
  nvmlInit_v2(); nvmlDevice_t nd; nvmlDeviceGetHandleByIndex_v2(0,&nd);
  cublasHandle_t h;cublasCreate(&h);cublasSetMathMode(h,CUBLAS_TENSOR_OP_MATH);
  int SH[][3]={{8192,15360,3840},{8192,3840,15360},{4096,4096,4096},{8192,4096,3840}};
  printf("%6s %6s %6s | %-34s | %-34s | %s\n","M","N","K",
         "КОНСТАНТА (memset 0x11)","СЛУЧАЙНЫЕ (как в бою)","конст/случ");
  for(auto&s:SH){int M=s[0],N=s[1],K=s[2];
    __half*A,*B;void*C;cudaMalloc(&A,(size_t)M*K*2);cudaMalloc(&B,(size_t)N*K*2);cudaMalloc(&C,(size_t)M*N*2);
    float al=1,be=0;double flop=2.0*M*N*K;cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);
    auto go=[&]{cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&al,B,CUDA_R_16F,K,A,CUDA_R_16F,K,&be,C,
                             CUDA_R_16F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);};
    double tf[2],fq[2],pw[2],fmin[2];
    for(int mode=0;mode<2;++mode){
      if(mode==0){cudaMemset(A,0x11,(size_t)M*K*2);cudaMemset(B,0x11,(size_t)N*K*2);}
      else{fill_rand<<<512,256>>>(A,(long)M*K,11u);fill_rand<<<512,256>>>(B,(long)N*K,77u);}
      cudaDeviceSynchronize();
      for(int i=0;i<20;++i)go();cudaDeviceSynchronize();      // прогрев ДО стабилизации мощности
      std::atomic<bool> stop{false}; std::vector<double> cs,ps;
      std::thread w([&]{unsigned c,p;while(!stop.load()){
        if(nvmlDeviceGetClockInfo(nd,NVML_CLOCK_SM,&c)==0)cs.push_back(c);
        if(nvmlDeviceGetPowerUsage(nd,&p)==0)ps.push_back(p/1000.0);}});
      cudaEventRecord(e0);for(int i=0;i<60;++i)go();cudaEventRecord(e1);cudaEventSynchronize(e1);
      stop=true;w.join();
      float ms;cudaEventElapsedTime(&ms,e0,e1);
      tf[mode]=flop/(ms/60*1e-3)/1e12; fq[mode]=med(cs); pw[mode]=med(ps);
      fmin[mode]=cs.empty()?0:*std::min_element(cs.begin(),cs.end());}
    printf("%6d %6d %6d | %6.1f ТФ %6.0f МГц(мин %4.0f) %5.0f Вт | %6.1f ТФ %6.0f МГц(мин %4.0f) %5.0f Вт | x%.3f\n",
           M,N,K,tf[0],fq[0],fmin[0],pw[0],tf[1],fq[1],fmin[1],pw[1],tf[0]/tf[1]);
    cudaFree(A);cudaFree(B);cudaFree(C);}
  return 0;}
