## CUDA 프로그래밍 기초 실습

다음 CUDA source 들에 대해서 살펴보고 수정하면서 실행해 보세요!

```
01_hello.cu: Kernel 함수를 이용하여 Hello World를 프린트하는 코드입니다.

02_checkDeviceInfor.cu: 사용하고 있는 시스템의 GPU/CUDA 관련 정보를 추출하여 프린트하는 코드입니다.

03_checkDimension.cu: grid index, block index 및 thread index에 대한 이해를 돕는 코드입니다.

04_checkThreadIndex.cu: grid index, block index 및 thread index에 대한 이해를 돕는 코드입니다.

05_sumArraysOnHost.c: CPU에서 수행되는 일반적인 벡터 합 코드입니다.

06_sumArraysOnGPU-small-case.cu: 수업 시간에 배운 CUDA 기반의 벡터 합 코드입니다.
```

위의 코드들에 대해서 다양한 수정을 통해서 ```block index``` 및 ```thread index```에 대한 이해를 높이도록 하세요.

*  *  *
**01_hello.cu**

아래 코드는 kernel 함수를 이용하여 Hello World를 프린트하는 함수입니다.

main 함수에서 helloFromGPU를 call할 때, ```<<<1, 10>>>```로 thread 생성 configuration을 설정하였는데, 이는 10개의 쓰레드를 가진 하나의 block을 생성함을 의미합니다. 따라서 총 10개의 쓰레드가 생성되고 10번의 "Hello World from GPU!"가 프린트 됩니다.

```C
#include "./common.h"
#include <stdio.h>

/*
 * A simple introduction to programming in CUDA. This program prints "Hello
 * World from GPU! from 10 CUDA threads running on the GPU.
 */

__global__ void helloFromGPU()
{
    printf("Hello World from GPU!\n");
}

int main(int argc, char **argv)
{
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 10>>>();
    CHECK(cudaDeviceReset());
    return 0;
}
```

*  *  *
**02_checkDeviceInfor.cu**

아래 코드는 GPU의 사양 및 사용하고 있는 CUDA SDK의 상세 정보를 프린트하는 프로그램입니다.

CUDA Sample 중 ```~/samples/1_Utilities/deviceQuery/``` 디렉토리에 있는 ```deviceQuery```도 유사한 기능을 수행합니다.

```C
#include "./common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * Display a variety of information on the first CUDA device in this system,
 * including driver version, runtime version, compute capability, bytes of
 * global memory, etc.
 */

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);
    printf("  Total amount of global memory:                 %.2f MBytes (%llu "
           "bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
           (unsigned long long)deviceProp.totalGlobalMem);
    printf("  GPU Clock rate:                                %.0f MHz (%0.2f "
           "GHz)\n", deviceProp.clockRate * 1e-3f,
           deviceProp.clockRate * 1e-6f);
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:                              %d-bit\n",
           deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize)
    {
        printf("  L2 Cache Size:                                 %d bytes\n",
               deviceProp.l2CacheSize);
    }

    printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), "
           "2D=(%d,%d), 3D=(%d,%d,%d)\n", deviceProp.maxTexture1D,
           deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
           deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
           deviceProp.maxTexture3D[2]);
    printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
           "2D=(%d,%d) x %d\n", deviceProp.maxTexture1DLayered[0],
           deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0],
           deviceProp.maxTexture2DLayered[1],
           deviceProp.maxTexture2DLayered[2]);
    printf("  Total amount of constant memory:               %lu bytes\n",
           deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %lu bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
           deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
           deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %lu bytes\n",
           deviceProp.memPitch);

    exit(EXIT_SUCCESS);
}
```


*  *  *

**03_checkDimension.cu**

아래 코드는 thread Index 와 block Index의 개념을 이해하기 위해서 구성된 코드입니다. 전체 저리해야할 데이터의 수가 ```int nElem = 6;```로 6개 있습니다. ```dim3 block(3);```를 통해서 하나의 block에 3개의 thread를 사용한다면 몇개의 block이 만들어져야 할까요 ?

답은 식 "```dim3 grid((nElem + block.x - 1) / block.x);```"를 통해서 ```(6 + 3 - 1) / 3```, 즉 2입니다. ```(6 + 3 - 1) / 3```의 값은 실상 2.666.. 이지만, 정수형으로 짤리기 때문에 2가 됩니다. 그렇다면, 수식을 좀더 간단히 ```dim3 grid(nElem / block.x);```와 같이 만들지 않은 것일까요 ?


```C
#include "./common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * Display the dimensionality of a thread block and grid from the host and
 * device.
 */

__global__ void checkIndex(void)
{
    printf("threadIdx:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx:(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);

    printf("blockDim:(%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim:(%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

}

int main(int argc, char **argv)
{
    // define total data element
    int nElem = 6;

    // define grid and block structure
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);

    // check grid and block dimension from host side
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

    // check grid and block dimension from device side
    checkIndex<<<grid, block>>>();

    // reset device before you leave
    CHECK(cudaDeviceReset());

    return(0);
}
```

*  *  *

**04_checkThreadIndex.cu**

아래 코드는 1차원 array를 2차원 형식으로 어떻게 처리하는지 보여주는 코드입니다.

첫번째 함수는 CPU에서 동작하는 코드이고 두번째 kernel 함수는 GPU에서 동작하는 코드입니다.

```C
void printMatrix(int *C, const int nx, const int ny)
{
    int *ic = C;
    printf("\nMatrix: (%d.%d)\n", nx, ny);

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            printf("%3d", ic[ix]);

        }

        ic += nx;
        printf("\n");
    }

    printf("\n");
    return;
}

__global__ void printThreadIndex(int *A, const int nx, const int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) global index"
           " %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y,
           ix, iy, idx, A[idx]);
}
```

GPU에서 동작하는 kernel 함수의 2차원 구조의 thread index 와 block index를 살펴보시기 바랍니다.

>   int ix = threadIdx.x + blockIdx.x * blockDim.x;
>
>   int iy = threadIdx.y + blockIdx.y * blockDim.y;
>
>   unsigned int idx = iy * nx + ix;

*  *  *

**05_sumArraysOnHost.c**

아래 코드는 CPU에서 수행되는 통상적인 벡터 합 프로그램입니다. 

```C
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }

}
```

*  *  *

**06_sumArraysOnGPU-small-case.cu**

아래 코드는

