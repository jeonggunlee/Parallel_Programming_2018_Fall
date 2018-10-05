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








