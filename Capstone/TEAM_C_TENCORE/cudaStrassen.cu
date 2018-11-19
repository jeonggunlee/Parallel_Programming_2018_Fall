/*
https://www.cise.ufl.edu/~sahni/papers/strassen.pdf
슈트라센 알고리즘을 쿠다로 구현한 코드입니다. 
*/

__device__ void update2(float *a, float b, float *c)
{
   for (int i = 0; i < 16; i++)
      c[i] += a[i * 4] * b;
}

__global__ void GPU8 (float *a, float *b, float *c, int n)
{// thread code to compute one column of a 16 x 128 sub-matrix of c
 // use shared memory to hold the transpose of a
 // 16 x 64 sub-matrix of 1 x 4 sub-vectors of a
    __shared__ float as[16][65];
    // registers for column of c sub-matrix
    float cr[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int nDiv64 = n/64;
    int sRow = threadIdx.y;
    int sRow4 = sRow*4;
    int sCol = threadIdx.x;
    int tid = sRow*16+sCol.x;
    int aNext = (16*blockIdx.y+sRow)*n+sCol*4;
    int bNext = 128*blockIdx.x + tid;
    int cNext = 16*blockIdx.y*n + 128*blockIdx.x + tid;
    int nTimes2 = 2*n;
    int nTimes3 = 3*n;
    int nTimes4 = 4*n;
    a += aNext;
    b += bNext;
    c += cNext;
    float4 *a4 = (float4 *)a;
    for (int i = 0; i < nDiv64; i++)
    {
        *( (float4 *)(&as[sCol][sRow4]) ) = a4[0];
        *( (float4 *)(&as[sCol][sRow4+32]) ) = a4[nTimes2];
        __syncthreads(); // wait for read to complete
        
        float br0 = b[0];
        float br1 = b[n];
        float br2 = b[nTimes2];
        float br3 = b[nTimes3];
      b += nTimes4;
        #pragma unroll
        for (int k = 0; k < 15; k++)
        {
            update2 (&as[k][0], br0, cr); br0 = b[0];
            update2 (&as[k][1], br1, cr); br1 = b[n];
            update2 (&as[k][2], br2, cr); br2 = b[nTimes2];
            update2 (&as[k][3], br3, cr); br3 = b[nTimes3];
            b+= nTimes4;
        }
        update2 (&as[15][0], br0, cr);
        update2 (&as[15][1], br1, cr);
        update2 (&as[15][2], br2, cr);
        update2 (&as[15][3], br3, cr);
        a4 += 16;
        __syncthreads(); // wait for computation to complete
    }
    for (int j = 0; j < 16; j++)
    {
      c[0] = cr[j];
       c += n; }
}


__global__ void add (float *d_A, float *d_B, float *d_C, int widthA, int widthB, int widthC)
{
    int startA = blockIdx.x*64 + threadIdx.x*2 + (blockIdx.y*8 + threadIdx.y)*widthA;
    int startB = blockIdx.x*64 + threadIdx.x*2 + (blockIdx.y*8 + threadIdx.y)*widthB;
    int startC = blockIdx.x*64 + threadIdx.x*2 + (blockIdx.y*8 + threadIdx.y)*widthC;
    float2 tempA = *(float2 *)(d_A+startA);
    float2 tempB = *(float2 *)(d_B+startB);
    tempA.x += tempB.x;
    tempA.y += tempB.y;
    *(float2 *)(d_C+startC) = tempA;
}
