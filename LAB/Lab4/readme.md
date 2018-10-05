## CUDA 프로그래밍 기초 실습

```
07_sumArraysOnGPU-timer.cu : 벡터함 성능 평가
08_sumMatrixOnGPU.cu : Matrix 합 GPU 코드
```

*  *  *

```07_sumArraysOnGPU-timer.cu```의 내부 코드 중 main 함수에서 kernel 함수를 부르기 전에 시간을 재고, kernel 함수를 부르고 난 후에 시간을 잰 후 그 두 시간의 차이를 계산함으로써 수행시간을 구하게 됩니다.

```C
    iStart = seconds();
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
```

자!,그럼 저기 코드 사이에 끼어 있는 ```cudaDeviceSynchronize()``` 는 몰까요 ?
GPU kernel 함수는 기본적으로 asynchronous (non-blocking) 함수로써 GPU에서 수행이 되는 와중에서 CPU에서는 수행이 진행될 수 있습니다. 이 말은 다시 말해서 GPU 함수가 끝나지 않은 상태에서 CPU에서 시간을 젤수도 있다는 뜻이지요.

따라서, 끝나는 시간을 재기 전에 받드시 kernel 함수가 끝나는 것을 확인해야합니다. 이를 위하여 ```cudaDeviceSynchronize()``` 함수를 불러 GPU에서 동작하는 kernel 함수가 완료된 것을 기다리도록 하는 것입니다.


*  *  *

```08_sumMatrixOnGPU.cu```: 행렬 합 프로그램입니다. 다양한 형태의 코딩이 가능하네요! 분석해보시기 바랍니다.

```C
// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx,
                                 int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

// grid 1D block 1D
__global__ void sumMatrixOnGPU1D(float *MatA, float *MatB, float *MatC, int nx,
                                 int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

    if (ix < nx )
        for (int iy = 0; iy < ny; iy++)
        {
            int idx = iy * nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }


}

// grid 2D block 1D
__global__ void sumMatrixOnGPUMix(float *MatA, float *MatB, float *MatC, int nx,
                                  int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}
```




