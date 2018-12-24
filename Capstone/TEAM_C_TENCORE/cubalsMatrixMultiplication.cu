
/*
병렬프로그래밍 c_team
 10월 29일 과제
 cuda를 이용하여 행렬 A*B+C 연산 구현
 +cublas 라이브러리를 사용할 것
*/

#include <stdio.h> 
#include <stdlib.h> 
#include <cuda_runtime.h> 
#include "cublas_v2.h"

#define IDX2C(i,j,ld) (((j)*(ld))+( i ))  // 메모리 index
#define m 6 // a 는 mxk 행렬 
#define n 4 // b 는 kxn 행렬 
#define k 5 // c 는 mxn 행렬

int main(void) {
 cudaError_t cudaStat; // cudaMalloc status 
 cublasStatus_t stat; // CUBLAS functions status 
 cublasHandle_t handle; // CUBLAS context
 
 int i, j; // i는 행의 index j는 열의 index
 
 // 행열 메모리 주소를 가리킬 포인터
 float* a;
 float* b;
 float* c;
 
 // 메모리를 행렬의 크기만큼 할당한다.
 a = (float*)malloc(m*k * sizeof(float)); // host memory for a
 b = (float*)malloc(k*n * sizeof(float)); // host memory for b 
 c = (float*)malloc(m*n * sizeof(float)); // host memory for c

 // cublas 행렬곱 연산은 cloumn major이기 때문에
 // 행렬 a를 열 기준으로 값을 저장한다.
 int ind = 11; // 행렬의 1행 1열의 값
 for (j = 0; j < k; j++) {
  for (i = 0; i < m; i++) {
   a[IDX2C(i, j, m)] = (float)ind++;
  }
 }
 // 11,17,23,29,35 
 // 12,18,24,30,36 
 // 13,19,25,31,37 
 // 14,20,26,32,38
 // a 행렬 출력
 printf("a:\n");
 for (i = 0; i < m; i++) {
  for (j = 0; j < k; j++)
  {
   printf("%5.0f", a[IDX2C(i, j, m)]);
  }
  printf("\n");
 }
 // b행렬 
 ind = 11; // 행렬 1행 1열의 값 
 for (j = 0; j < n; j++) {
  for (i = 0; i < k; i++) {
   b[IDX2C(i, j, k)] = (float)ind++;
  }
 }
 //b:
 //11,16,21,26
 //12,17,22,27
 //13,18,23,28
 //14,19,24,29
 //15,20,25,30

 printf("b:\n"); 
 for (i = 0; i < k; i++) { 
  for (j = 0; j < n; j++) {
   printf("%5.0f", b[IDX2C(i, j, k)]); 
  } 
  printf("\n");
 }
 
 ind = 11; // 1행 1열의 값
 for(j=0;j<n;j++){
   for(i=0;i<m;i++){
    c[IDX2C(i,j,m)]=(float)ind++;
  } 
 }
 
 //c:
 //11,17,23,29
 //12,18,24,30
 //13,19,25,31
 //14,20,26,32
 //15,21,27,33
 //16,22,28,34
 
 printf("c:\n"); 
 for (i = 0; i < m; i++) { 
  for (j = 0; j < n; j++) { 
   printf("%5.0f", c[IDX2C(i, j, m)]); 
  } 
  printf("\n"); 
 }
 
 // gpu로 값을 넘주는 작업
 float* d_a; 
 float* d_b; 
 float* d_c;
 
 // 행렬 크기만큼 gpu 메모리 할당  
 cudaStat=cudaMalloc((void**)&d_a,m*k*sizeof(*a)); 
 cudaStat=cudaMalloc((void**)&d_b,k*n*sizeof(*b)); 
 cudaStat=cudaMalloc((void**)&d_c,m*n*sizeof(*c));

 stat = cublasCreate(&handle); // cublas 초기화
 
 // 행렬의 값을 gpu메모리로 복사
 stat = cublasSetMatrix(m, k, sizeof(*a), a, m, d_a, m);//a -> d_a 
 stat = cublasSetMatrix(k,n,sizeof(*b),b,k,d_b,k);//b -> d_b 
 stat = cublasSetMatrix(m,n,sizeof(*c),c,m,d_c,m);//c -> d_c
 // matrix -matrix multiplication: d_c = al*d_a*d_b + bet*d_c 
 // d_a -mxk matrix , d_b -kxn matrix , d_c -mxn matrix;
 
 // 사용하는 함수는 C=al*A*B+C 결과를 가진다 
 // 따라서 C가 0행렬이라면 A*B의 결과만
 // C의 값이 존재한다면 A*B+C의 결과르 가진다. 
 
 // al,bet는 행렬의 scalar 이다.
 float al = 1.0f; // al=1
 float bet=1.0f; //bet=1
 
 stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, d_a, m, d_b, k, &bet, d_c, m);
 stat = cublasGetMatrix(m, n, sizeof(*c), d_c, m, c, m); //cp d_c->c 
 
 printf("c after Sgemm :\n"); 
 for(i=0;i<m;i++){ 
  for(j=0;j<n;j++){ 
   printf("%7.0f",c[IDX2C(i,j,m)]); //print c after Sgemm 
  } 
  printf("\n"); 
 } 
 
 // gpu 메모리 헤재
 cudaFree(d_a); 
 cudaFree(d_b); 
 cudaFree(d_c); 
 
 cublasDestroy(handle); //쿠다 명령어 삭제
 
 // cpu 메모리 헤제 
 free(a); 
 free(b); 
 free(c); 
 return 0;
}
