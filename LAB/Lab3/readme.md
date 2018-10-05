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

아래 코드는 kernel 함수를 이용하여 Hello World를 프린트하는 함수입니다.

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

