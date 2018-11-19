/*
cuBLAS users will notice a few changes from their existing cuBLAS GEMM code:

1.The routine must be a GEMM; currently, only GEMMs support Tensor Core execution.

2.The math mode must be set to CUBLAS_TENSOR_OP_MATH. Floating point math is not associative, so 
the results of the Tensor Core math routines are not quite bit-equivalent to the results of the analogous 
non-Tensor Core math routines.  cuBLAS requires the user to “opt in” to the use of tensor cores.
3.All of k, lda, ldb, and ldc must be a multiple of eight; 
m must be a multiple of four. The Tensor Core math routines stride through input data in steps of
eight values, so the dimensions of the matrices must be multiples of eight.

4.The input and output data types for the matrices must be either half precision or single precision. (Only CUDA_R_16F is shown above, 
but CUDA_R_32F also is supported.)

*/

// First, create a cuBLAS handle:
cublasStatus_t cublasStat = cublasCreate(&handle);

// Set the math mode to allow cuBLAS to use Tensor Cores:
cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

// Allocate and initialize your matrices (only the A matrix is shown):
size_t matrixSizeA = (size_t)rowsA * colsA;
T_ELEM_IN **devPtrA = 0;

cudaMalloc((void**)&devPtrA[0], matrixSizeA * sizeof(devPtrA[0][0]));
T_ELEM_IN A  = (T_ELEM_IN *)malloc(matrixSizeA * sizeof(A[0]));

memset( A, 0xFF, matrixSizeA* sizeof(A[0]));
status1 = cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, rowsA, devPtrA[i], rowsA);

// ... allocate and initialize B and C matrices (not shown) ...

// Invoke the GEMM, ensuring k, lda, ldb, and ldc are all multiples of 8, 
// and m is a multiple of 4:
cublasStat = cublasGemmEx(handle, transa, transb, m, n, k, alpha,
                          A, CUDA_R_16F, lda,
                          B, CUDA_R_16F, ldb,
                          beta, C, CUDA_R_16F, ldc, CUDA_R_32F, algo);
