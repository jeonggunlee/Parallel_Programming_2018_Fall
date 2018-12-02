#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>


 // 텐서코어와 쿠다 코어의 결과를 저장한다.
// cublas를 통하여 계산하며 
// 옵션을 통하여 쿠다와 텐서코어를 선택한다.
 // Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}
 #define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}
 #define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}
 #include <mma.h>
using namespace nvcuda;
 // Must be multiples of 16 for wmma code to work
#define MATRIX_M 268435456
#define MATRIX_N 268435456
#define MATRIX_K 268435456



 __global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}


 int main(int argc, char* argv[]) {
   float *a_fp32;
   float *b_fp32;
   half *a_fp16;
   half *b_fp16;
    float *c;

   float *c_cuda;

   float *c_host_cuda;

   
   curandGenerator_t gen;
   cublasHandle_t cublasHandle;
   
   cudaEvent_t startCUDA;
   cudaEvent_t stopCUDA;
   
   
   cudaErrCheck(cudaEventCreate(&startCUDA));
   cudaErrCheck(cudaEventCreate(&stopCUDA));
   
   
   
   cublasErrCheck(cublasCreate(&cublasHandle));
   
   
   // 쿠다 코어로 변경 1번 CUBLAS_TENSOR_OP_MATH를 CUBLAS_DEFAULT_MATH로 변경해준다.
   cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH));// 
   
// 메모리 할당
   cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
   cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

    cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_cuda, MATRIX_M * MATRIX_N * sizeof(float)));


   c_host_cuda = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

    curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
   curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));

    // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K);
   convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);
    curandErrCheck(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N));
   
   curandErrCheck(curandDestroyGenerator(gen));
   
   
   cudaErrCheck(cudaMemcpy(c_cuda, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));
    float alpha = 2.0f;
   float beta = 2.0f;
    printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
   
   
   
   // Now using cuBLAS with CUDA
   printf("Running with cuBLAS with Cuda Core\n");
   cudaErrCheck(cudaEventRecord(startCUDA));
   // 쿠다 코어로 변경 2번
   // 쿠다 코어 이용시에 CUUBLAS_GEMM_DFALT_TENSOR_OP을 CUBLAS_GEMM_DEFALT 로 변경해준다. 
   cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
                &alpha,
                a_fp32, CUDA_R_32F, MATRIX_M,
                b_fp32, CUDA_R_32F, MATRIX_K,
                &beta, 
                c_cuda, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DFALT));
   cudaErrCheck(cudaEventRecord(stopCUDA));
    // Error checking
  


   cudaErrCheck(cudaMemcpy(c_host_cuda, c_cuda, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
   
 
   
   //  cublas결과와 wmma결과를 비교하는 코드이다. 
   //  c_host_cuda[i] 에는 cuda 결과가 
   //  c_host_cublas[i] 에는 cublass 결과가 담긴다.
   //  혼합 정밀도를 가진 TensorCore를 통하여 계산한 결과아
   //  단일정밀도를 가진 쿠다 코어로 계산한 결과를 비교하면 반드시 에러가 발생한다. 
    // 0.01% relative tolerance. 1e-5 absolute tolerance.

      printf("Results.\n\n");
      float cudaTime;
      cudaErrCheck(cudaEventSynchronize(stopCUDA));
      cudaErrCheck(cudaEventElapsedTime(&cudaTime, startCUDA, stopCUDA));
      
     // TFLOPS 계산 결과 출력
      printf("cuda took %fms\n", cudaTime);
      printf("[+] TFLOPS: %.2f\n", ((double)MATRIX_M * MATRIX_N * MATRIX_K * 2) / cudaTime / 1e9);
  
       printf("\nCUBALS WITH CUDA !\n\n");
   
   
 
   
   cudaErrCheck(cudaEventDestroy(startCUDA));
   cudaErrCheck(cudaEventDestroy(stopCUDA));
;
   
   cudaErrCheck(cudaFree(a_fp32));
   cudaErrCheck(cudaFree(b_fp32));
   cudaErrCheck(cudaFree(a_fp16));
   cudaErrCheck(cudaFree(b_fp16));
   cudaErrCheck(cudaFree(c));
   cudaErrCheck(cudaFree(c_cuda));

   free(c_host_cuda);
    cudaErrCheck(cudaDeviceReset());
   return 0;
}
