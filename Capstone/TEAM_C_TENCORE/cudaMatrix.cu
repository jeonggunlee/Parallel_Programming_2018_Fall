
#include <stdio.h> 
#include <stdlib.h> 
#include <cuda_runtime.h> 
#include <iostream>

#define IDX2C(rowlength,i,j) ((rowlength*i)+j) // 행렬의 index 구하는 방식 

#define m 2
#define k 3
#define n 2

using namespace std;


// 쓰레드의 인덱스 만큼 행렬의 덧샘을 실행하는 함수 
__global__ void add(int *c , int *d){
    int tid=threadIdx.x;
    d[tid]+=c[tid];
}

__global__ void multi(int *a , int *b,int *d){
    int tid=threadIdx.x;

    for(int j=0; j<k ; j++){
        for(int l=0;l<k; l++){
            d[IDX2C(n,tid,j)]+=a[IDX2C(k,tid,l)]*b[IDX2C(n,l,j)];
        }    
    } 
}


int main(){
    int *a;
    int *b;
    int *c;
    int *d;
    int i,j;

    //device 메모리
    int *d_a,*d_b,*d_c,*d_d;
    
    // 행렬 메모리 할당 
    a=(int*)malloc( m*k * sizeof(int) ); 
    b=(int*)malloc( k*n * sizeof(int) ); 
    c=(int*)malloc( m*n * sizeof(int) ); 
    d=(int*)malloc( m*n * sizeof(int) ); 
    

    


    int value_a[m*k]={1,0,-3,-2,4,1};
    int value_b[k*n]={2,-1,3,0,-5,2};
    int value_c[m*n]={3,-1,-2,2};


    // a,b,c행렬의 값을 넣고 확인한다. 
    cout<<"a:\n";
    for(i=0; i<m*k; i++){
        a[i]=value_a[i];
    }

    for(i=0; i<m; i++){
        for(j=0;j<k;j++){
            cout<<(a[ IDX2C(k,i,j) ])<<" ";
        }
        cout<<endl;
    }

    cout<<"b:\n";
    for(i=0; i<k*n; i++){
        b[i]=value_b[i];
    }

    for(i=0; i<k; i++){
        for(j=0;j<n;j++){
            cout<<(b[ IDX2C(n,i,j) ])<<" ";
        }
        cout<<endl;
    }

    cout<<"c:\n";
    for(i=0; i<m*n; i++){
        c[i]=value_c[i];
    }

    for(i=0; i<m; i++){
        for(j=0;j<n;j++){
            cout<<(c[ IDX2C(n,i,j) ])<<" ";
        }
        cout<<endl;
    }

    // 결과는 0으로 초기화
    for(i=0; i<m*n; i++){
        d[i]=0;
    }

    // cuda 메모리 할당
    cudaMalloc( (void**)&d_a , m*k*sizeof(int) ) ;
    cudaMalloc( (void**)&d_b , k*n*sizeof(int) ) ;
    cudaMalloc( (void**)&d_c , m*n*sizeof(int) ) ;
    cudaMalloc( (void**)&d_d , m*n*sizeof(int) ) ;

    // device로 행렬값 전달 
    cudaMemcpy( d_a,a,m*k*sizeof(int),cudaMemcpyHostToDevice );
    cudaMemcpy( d_b,b,k*n*sizeof(int),cudaMemcpyHostToDevice );
    cudaMemcpy( d_c,c,m*n*sizeof(int),cudaMemcpyHostToDevice );
    


    //행렬의 행의 수 만큼 쓰레드 생성 
    multi<<<1,m>>>(d_a,d_b,d_d);

    //메모리 가지고 오기
    cudaMemcpy( d,d_d,m*n*sizeof(int),cudaMemcpyDeviceToHost );

    cout<<"A*B 결과 d:\n";


    for(i=0; i<m; i++){
        for(j=0;j<n;j++){
            cout<<(d[ IDX2C(n,i,j) ])<<" ";
        }
        cout<<endl;
    }


    //행렬의 원소 수 만큼 쓰레드를 생성 후 덧샘
    
    add<<<1,m*n>>>(d_c,d_d);
    
    //메모리 가지고 오기
    cudaMemcpy( d,d_d,m*n*sizeof(int),cudaMemcpyDeviceToHost );

    cout<<"A*B+C 결과 d:\n";


    for(i=0; i<m; i++){
        for(j=0;j<n;j++){
            cout<<(d[ IDX2C(n,i,j) ])<<" ";
        }
        cout<<endl;
    }
    
    //device 메모리 헤제
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);



    // host 메모리 헤제

    free(a);
    free(b);
    free(c);
    free(d);

    return 0;
}
