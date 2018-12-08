#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>



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
#define MATRIX_M 16384
#define MATRIX_N 16384
#define MATRIX_K 16384



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
   float *c_cublas;
   float *c_cuda;
    float *c_host_cublas;
   float *c_host_cuda;

   
   curandGenerator_t gen;
   cublasHandle_t cublasHandle;
   
   cudaEvent_t startCUDA;
   cudaEvent_t stopCUDA;
   
   cudaEvent_t startcublas;
   cudaEvent_t stopcublas;
   
   cudaErrCheck(cudaEventCreate(&startCUDA));
   cudaErrCheck(cudaEventCreate(&stopCUDA));
   
   cudaErrCheck(cudaEventCreate(&startcublas));
   cudaErrCheck(cudaEventCreate(&stopcublas));
   
   
   cublasErrCheck(cublasCreate(&cublasHandle));
   
   
   // 쿠다 코어로 변경 1번 CUBLAS_TENSOR_OP_MATH를 CUBLAS_DEFAULT_MATH로 변경해준다.
   cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH));// 
   
// 메모리 할당
   cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
   cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

    cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_cuda, MATRIX_M * MATRIX_N * sizeof(float)));

    c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
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
   
   cudaErrCheck(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));
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
                a_fp16, CUDA_R_16F, MATRIX_M,
                b_fp16, CUDA_R_16F, MATRIX_K,
                &beta, 
                c_cuda, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DFALT));
   cudaErrCheck(cudaEventRecord(stopCUDA));
    // Error checking
  

   cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));// 

   // Now using cuBLAS with Tensor
   printf("Running with cuBLAS with Tensor Core\n");
   cudaErrCheck(cudaEventRecord(startcublas));
   // 쿠다 코어로 변경 2번
   // 쿠다 코어 이용시에 CUUBLAS_GEMM_DFALT_TENSOR_OP을 CUBLAS_GEMM_DFALT 로 변경해준다. 
   cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
                &alpha,
                a_fp16, CUDA_R_16F, MATRIX_M,
                b_fp16, CUDA_R_16F, MATRIX_K,
                &beta, 
                c_cublas, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
   cudaErrCheck(cudaEventRecord(stopcublas));
    // Error checking
   printf("\nChecking results...\n");

   cudaErrCheck(cudaMemcpy(c_host_cuda, c_cuda, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
   cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
   
   
  
  
      printf("Results.\n\n");
      float cudaTime;
      float cublasTime;
      cudaErrCheck(cudaEventSynchronize(stopCUDA));
      cudaErrCheck(cudaEventSynchronize(stopcublas));
      cudaErrCheck(cudaEventElapsedTime(&cudaTime, startCUDA, stopCUDA));
      cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
      
     // TFLOPS 계산 결과 출력
      printf("cuda took %fms\n", cudaTime);
      printf("[+] TFLOPS: %.2f\n", ((double)MATRIX_M * MATRIX_N * MATRIX_K * 2) / cudaTime / 1e9);
      printf("tensor took %fms\n", cublasTime);
      printf("[+] TFLOPS: %.2f\n", ((double)MATRIX_M * MATRIX_N * MATRIX_K * 2) / cublasTime / 1e9);
   
       printf("\nCUBALS WITH CUDA OR TENSOR CORE CODE !\n\n");
   
   
      
   
   cudaErrCheck(cudaEventDestroy(startCUDA));
   cudaErrCheck(cudaEventDestroy(stopCUDA));
    cudaErrCheck(cudaEventDestroy(startcublas));             
   cudaErrCheck(cudaEventDestroy(stopcublas));
   
   cudaErrCheck(cudaFree(a_fp32));
   cudaErrCheck(cudaFree(b_fp32));
   cudaErrCheck(cudaFree(a_fp16));
   cudaErrCheck(cudaFree(b_fp16));
    cudaErrCheck(cudaFree(c));
   cudaErrCheck(cudaFree(c_cublas));
   cudaErrCheck(cudaFree(c_cuda));
   
   free(c_host_cublas);
   free(c_host_cuda);
    cudaErrCheck(cudaDeviceReset());
   return 0;
}
