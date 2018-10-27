/*
슈트라센 알고리즘 구현한 코드입니다.
주석이 전부 빠져있습니다. 
주석을 달아주세요. 
 */
 #include "stdio.h"
#include "stdlib.h"
#include "time.h"
 void normal_product(int n, int **A, int **B,int ***C);
void strassen(int n, int**A, int **B,int ***C);
void matrix_sum(int n,int **A, int **B, int ***C);
void matrix_subtract(int n,int **A, int **B, int ***C);
 const int matrix_array = 8;	
const int thres = 3;
 void normal_product(int n, int **A, int **B, int ***C)
{
	int i, j, k;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			(*C)[i][j] = 0;
			for (k = 0; k < n; k++)
				(*C)[i][j] = (*C)[i][j] + A[i][k] * B[k][j];
		}
	}
}
void strassen(int n, int **A, int **B,int ***C)
{
	if (n < thres)
	{
		normal_product(n,A, B, C);
	}
	else
	{
		int i, j, r, c;
		
		int **A_11 = (int**)malloc(sizeof(int*)*n / 2);
		int **A_12 = (int**)malloc(sizeof(int*)*n / 2);
		int **A_21 = (int**)malloc(sizeof(int*)*n / 2);
		int **A_22 = (int**)malloc(sizeof(int*)*n / 2);
		int **B_11 = (int**)malloc(sizeof(int*)*n / 2);
		int **B_12 = (int**)malloc(sizeof(int*)*n / 2);
		int **B_21 = (int**)malloc(sizeof(int*)*n / 2);
		int **B_22 = (int**)malloc(sizeof(int*)*n / 2);
		int **C_11 = (int**)malloc(sizeof(int*)*n / 2);
		int **C_12 = (int**)malloc(sizeof(int*)*n / 2);
		int **C_21 = (int**)malloc(sizeof(int*)*n / 2);
		int **C_22 = (int**)malloc(sizeof(int*)*n / 2);
		int **temp_A = (int**)malloc(sizeof(int*)*n / 2);
		int **temp_B = (int**)malloc(sizeof(int*)*n / 2);
		int **M1 = (int**)malloc(sizeof(int*)*n / 2);
		int **M2 = (int**)malloc(sizeof(int*)*n / 2);
		int **M3 = (int**)malloc(sizeof(int*)*n / 2);
		int **M4 = (int**)malloc(sizeof(int*)*n / 2);
		int **M5 = (int**)malloc(sizeof(int*)*n / 2);
		int **M6 = (int**)malloc(sizeof(int*)*n / 2);
		int **M7 = (int**)malloc(sizeof(int*)*n / 2);
		
		for (i = 0; i < n / 2; i++)
		{
			A_11[i] = (int *)malloc(sizeof(int)*n / 2);
			A_12[i] = (int *)malloc(sizeof(int)*n / 2);
			A_21[i] = (int *)malloc(sizeof(int)*n / 2);
			A_22[i] = (int *)malloc(sizeof(int)*n / 2);
			B_11[i] = (int *)malloc(sizeof(int)*n / 2);
			B_12[i] = (int *)malloc(sizeof(int)*n / 2);
			B_21[i] = (int *)malloc(sizeof(int)*n / 2);
			B_22[i] = (int *)malloc(sizeof(int)*n / 2);
			C_11[i] = (int *)malloc(sizeof(int)*n / 2);
			C_12[i] = (int *)malloc(sizeof(int)*n / 2);
			C_21[i] = (int *)malloc(sizeof(int)*n / 2);
			C_22[i] = (int *)malloc(sizeof(int)*n / 2);
			temp_A[i] = (int *)malloc(sizeof(int)*n / 2);
			temp_B[i] = (int *)malloc(sizeof(int)*n / 2);
			M1[i] = (int *)malloc(sizeof(int)*n / 2);
			M2[i] = (int *)malloc(sizeof(int)*n / 2);
			M3[i] = (int *)malloc(sizeof(int)*n / 2);
			M4[i] = (int *)malloc(sizeof(int)*n / 2);
			M5[i] = (int *)malloc(sizeof(int)*n / 2);
			M6[i] = (int *)malloc(sizeof(int)*n / 2);
			M7[i] = (int *)malloc(sizeof(int)*n / 2);
		}
		r = 0, c = 0;
		for (i = 0; i < n; i++)
		{
			c = 0;
			
			for (j = 0; j < n; j++)
			{
				
				if (i < n / 2 && j< n/2)
				{
					A_11[i][j] = A[i][j];
					B_11[i][j] = B[i][j];
					temp_A[i][j] = 0;
					temp_B[i][j] = 0;
					C_11[i][j] = 0;
					C_12[i][j] = 0;
					C_21[i][j] = 0;
					C_22[i][j] = 0;
				}
				else if (i >= n / 2 && j < n / 2)
				{
					A_21[r][j] = A[i][j];
					B_21[r][j] = B[i][j];
					
				}
				else if (i < n / 2 && j >= n / 2)
				{
					A_12[i][c] = A[i][j];
					B_12[i][c] = B[i][j];
					
				}
				else if (i >= n/2 && j >= n/2)
				{
					A_22[r][c] = A[i][j];
					B_22[r][c] = B[i][j];
					
				}
				if (j >= n / 2) c++;
			}
			if (i >= n / 2) r++;
		}
		r = 0, c = 0;
 		matrix_sum(n/2,A_11, A_22, &temp_A);
		matrix_sum(n / 2, B_11, B_22, &temp_B);
		strassen(n / 2, temp_A, temp_B, &M1);	//M1
		
		matrix_sum(n / 2, A_21, A_22,&temp_A);
		strassen(n / 2, temp_A, B_11, &M2);
		
		matrix_subtract(n / 2, B_12, B_22, &temp_B);
		strassen(n / 2, A_11, temp_B, &M3);
 		matrix_subtract(n / 2, B_21, B_11, &temp_B);
		strassen(n / 2, A_22, temp_B, &M4);
 		matrix_sum(n / 2, A_11, A_12, &temp_A);
		strassen(n / 2, temp_A, B_22, &M5);
 		matrix_subtract(n / 2, A_21, A_11, &temp_A);
		matrix_sum(n / 2, B_11, B_12, &temp_B);
		strassen(n / 2, temp_A, temp_B, &M6);
		
		matrix_subtract(n / 2, A_12, A_22, &temp_A);
		matrix_sum(n / 2, B_21, B_22, &temp_B);
		strassen(n / 2, temp_A, temp_B, &M7);
 		matrix_sum(n / 2, M1, M4, &temp_A);
		matrix_subtract(n / 2, temp_A, M5, &temp_B);
		matrix_sum(n / 2, temp_B, M7, &C_11);
 		matrix_sum(n / 2, M3, M5, &C_12);
		
		matrix_sum(n / 2, M2, M4, &C_21);
 		matrix_sum(n / 2, M1, M3, &temp_A);
		matrix_subtract(n / 2, temp_A, M2, &temp_B);
		matrix_sum(n / 2, temp_B, M6, &C_22);
 		for (i = 0; i < n; i++)
		{
			c = 0;
			
			for (j = 0; j < n; j++)
			{
				if (i < n / 2  && j < n / 2)
				{
					(*C)[i][j] = C_11[i][j];
				}
				else if (i >= n / 2 && j < n / 2)
				{
					(*C)[i][j] = C_21[r][j];
					
				}
				else if (i < n / 2 && j >= n / 2)
				{
					(*C)[i][j] = C_12[i][c];
					
				}
				else if (i >= n / 2 && j >= n / 2)
				{
					(*C)[i][j] = C_22[r][c];
					
				}
				if (j >= n / 2) c++;
			}
			if (i >= n / 2) r++;
		}
 		
		for (i = 0; i < n / 2; i++)
		{
			free(A_11[i]); free(A_12[i]); free(A_21[i]); free(A_22[i]);
			free(B_11[i]); free(B_12[i]); free(B_21[i]); free(B_22[i]);
			free(C_11[i]); free(C_12[i]); free(C_21[i]); free(C_22[i]);
			free(temp_A[i]); free(temp_B[i]);
			free(M1[i]); free(M2[i]); free(M3[i]); free(M4[i]); free(M5[i]); free(M6[i]); free(M7[i]);
		}
		free(A_11); free(A_12); free(A_21); free(A_22);
		free(B_11); free(B_12); free(B_21); free(B_22);
		free(C_11); free(C_12); free(C_21); free(C_22);
		free(temp_A); free(temp_B);
		free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
	}
}
 void matrix_sum(int n, int **A, int **B, int ***C)
{
	int i,j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			(*C)[i][j] = A[i][j] + B[i][j];
		}
	}
}
 void matrix_subtract(int n, int **A, int **B, int ***C)
{
	int i, j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			(*C)[i][j] = A[i][j] - B[i][j];
		}
	}
}
void main()
{
	int i,j;
	int **matrix_A = (int **)malloc(sizeof(int*) * matrix_array);
	int **matrix_B = (int **)malloc(sizeof(int*) * matrix_array);
	int **matrix_C = (int **)malloc(sizeof(int*) * matrix_array);
 	for (i = 0; i < matrix_array; i++)
	{
		matrix_A[i] = (int *)malloc(sizeof(int)*matrix_array);
		matrix_B[i] = (int *)malloc(sizeof(int)*matrix_array);
		matrix_C[i] = (int *)malloc(sizeof(int)*matrix_array);
	}
 	srand((unsigned)time(NULL));
	for (i = 0; i < matrix_array; i++)
	{
		for (j = 0; j < matrix_array; j++)
		{
			matrix_A[i][j] = (20*rand() / RAND_MAX) ;
			matrix_B[i][j] = (30*rand() / RAND_MAX) ;
			matrix_C[i][j] = 0;
		}
	}
 	printf("matA : \n");
	for (i = 0; i < matrix_array; i++)
	{
		
		for (j = 0; j < matrix_array; j++)
		{
			printf("%2d ", matrix_A[i][j]);
		}
		printf("\n");
	}
	printf("matB : \n");
	for (i = 0; i < matrix_array; i++)
	{
		
		for (j = 0; j < matrix_array; j++)
		{
			printf("%2d ", matrix_B[i][j]);
		}
		printf("\n");
	}
 	strassen(matrix_array, matrix_A, matrix_B,&matrix_C);
	printf("matA * matB : \n");
	
	for (i = 0; i < matrix_array; i++)
	{
 		for (j = 0; j < matrix_array; j++)
		{
			printf("%2d ", matrix_C[i][j]);
		}
		printf("\n");
	}
 	normal_product(matrix_array, matrix_A, matrix_B, &matrix_C);
 	printf("matA * matB : \n");
 	for (i = 0; i < matrix_array; i++)
	{
 		for (j = 0; j < matrix_array; j++)
		{
			printf("%2d ", matrix_C[i][j]);
		}
		printf("\n");
	}
 	for (i = 0; i < matrix_array; i++)
	{
		free(matrix_A[i]); free(matrix_B[i]); free(matrix_C[i]);
	}
	free(matrix_A);
	free(matrix_B);
	free(matrix_C);
}
