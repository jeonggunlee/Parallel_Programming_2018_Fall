	// 순차 구구단 !
	for(i = 2; i < 10; i++)	// 2단 - 9단
	   for(j = 1; j < 10; j++)	// 
		table[ (i-2)*9 + (j-1) ] = i*j;
		// 2*1 ---> 0, 2*2 ---> 1
		// 5*5 --->27 +4 = 31




// 병렬 구구단!!!!!!!
main
	multTable<<<8, 9>>>() // blockIdx : 0 .. 7
			       // threadIdx : 0 .. 8
			       // blockDim = 9



__global void multiTable ...
// 병렬 구구단 !
	i = blockIdx;
	j = threadIdx;
	table[ i*blockDim + j] = (i+2)*(j+1);
  
  
