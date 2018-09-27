main()
…
  // vectorAdd <<<block의수, block당thread의수  >>> (A, B, C, n)
  vectorAdd <<<10, 10>>> (A, B, C, 100)
…
}


__global void vectorAdd(int* A, int* B, int* C, int n)
{
	// core index에 해당하는 , 자신이 처리해야할 데이터 결정을 위해서
 	// 자신의 고유한 thread index를 찾아야함!!!
	// i = 고유한 thread index;
	// block index (blockIdx), thread index (threadIdx)
	// 한 block에 있는 thread의 수 (blockDim) = 10
 	i = blockIdx * blockDim + threaIdx;  // 0 … 99
	// 2번째 block의 첫번째 thread의 id ? 1*1+10 = 11
 	10    =   1        *      10        +       0   
	// 5번째 block의 7번째 thread의 id
	46   =    4  * 10 + 6
	C[i] = A[i] + B[i]

	// for( int j = i*10; j < 10*(i+1); j++ ) ------------------- (1)
	// for( int j = i*(n/100); j < (i+1)*(n/100); j++ ) ------- (2)
	// j = i
	//	C[j] = A[j] + B[j]
}
