## OpenMP를 이용한 간단한 병렬화 예제입니다.

*  *  *

아주 단순한 vector addition을 수행하는 코드입니다.
Loop iteration (루프 반복)에서 의존성이 없기 때문에 병렬로 수행될 수 있는 Loop입니다.

```c
...
void main()
{
   int i, k, N=1000;
   double A[N], B[N], C[N];
   
   for (i=0; i<N; i++)
   {
       A[i] = B[i] + k*C[i];
   }
}
...
```
위 코드의 수행시간을 N을 변화시켜가면서 측정해 볼까요 ?

자, 그럼 위의 코드에 ```#pragma omp parallel for```를 적용해 봅시다.

```c
#include “omp.h”
...
void main()
{
   int i, k, N=1000;
   double A[N], B[N], C[N];
   
   #pragma omp parallel for
   for (i=0; i<N; i++)
   {
       A[i] = B[i] + k*C[i];
   }
}
...
```
성능을 측정해보면 어떻게 나올까요 ?

위 코드 역시, N을 증가시켜보면서 성능을 측정해 보세요.


```
gcc myomp.c –o myomp -fopenmp
```

![Alt text](https://github.com/jeonggunlee/Parallel_Programming_2018_Fall/blob/master/img/openmp.PNG
 "Optional title")
 
 *  *  *

아래는 pi를 계산하는 C 프로그램의 일부입니다. 이를 완성하고, 수행시간이 얼마나 걸리는지 확인해 보세요!
 ```c
 ...
double pi = 0.0;
const int iterationCount = 200000000;

clock_t startTime = clock();

for (int i = 0; i < iterationCount; i++)
{
	pi += 4 * (i % 2 ? -1 : 1) / (2.0 * i + 1.0);
}

printf("Elpase Time : %d\n", clock() - startTime);
printf("pi = %.8f\n", pi);
...
```

자! 그럼 ```OpenMP```를 사용해 볼까요 ?

```c
...
double pi = 0.0;
const int iterationCount = 200000000;
clock_t startTime = clock();

#pragma omp parallel
{
	#pragma omp for
	for (int i = 0; i < iterationCount; i++)
	{
		pi += 4 * (i % 2 ? -1 : 1) / (2.0 * i + 1.0);
	}
}

printf("Elpase Time : %d\n", clock() - startTime);
printf("pi = %.8f\n", pi);
...
```

성능이 나아졌을까요 ?
나아지지 않았다면 왜 일까요 ?


아래의 코드는 어떨까요 ?

```c
...
double pi = 0.0;
const int iterationCount = 200000000;
clock_t startTime = clock();

#pragma omp parallel
{
	#pragma omp for
	for (int i = 0; i < iterationCount; i++)
	{
		#pragma omp atomic
		pi += 4 * (i % 2 ? -1 : 1) / (2.0 * i + 1.0);
	}
}

printf("Elpase Time : %d\n", clock() - startTime);
printf("pi = %.8f\n", pi);
...
```

자, 그럼 최종적으로 다음과 같은 코드는 어떨까요 ?

```c
...
double pi = 0.0;
const int iterationCount = 200000000;
clock_t startTime = clock();

#pragma omp parallel
{
	double temp = 0.0;
	#pragma omp for
	for (int i = 0; i < iterationCount; i++)
	{
		temp += 4 * (i % 2 ? -1 : 1) / (2.0 * i + 1.0);
	}
	#pragma omp atomic
	pi += temp;

}

printf("Elpase Time : %d\n", clock() - startTime);
printf("pi = %.8f\n", pi);
...
```

위의 코드들에 대해서 성능을 평가해보고, 왜 그러한 성능 지표가 나오는지 분석해보기 바랍니다.



