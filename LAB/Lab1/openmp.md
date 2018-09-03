## OpenMP를 이용한 간단한 병렬화 예제입니다.

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
