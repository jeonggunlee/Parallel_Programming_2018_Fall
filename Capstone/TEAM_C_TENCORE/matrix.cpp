/*
 1차 대학원생들과의 미팅 후 작성한 파일입니다.
 행렬의 연산에 대해서 알아보고 
 cpu상 병렬화를 구현하였습니다.
 
*/
//
//  main.cpp
//  matrix
//
//  Created by macRoom on 2018. 11. 2..
//  Copyright © 2018년 macRoom. All rights reserved.
//
#include <thread>
#include <iostream>

#define IDX2C(col,i,j) ((col*i)+j)  // 열*행 index + j열 index
#define m 2
#define k 3
#define n 2


using namespace std;

void add_thread(int* A, int *B , int* result , int size, int index,int nthreads){

    for(int i = index ; i< size ;i+=nthreads){
        result[i]=A[i]+B[i];
    }
    
}
void multi_thread(int* a,int *b, int *result,int a_m,int a_k,int b_n,int index , int nthreads ){
    for(int i = index ; i< a_m ;i+=nthreads){// 각 쓰레드는 다른 행으로 연산을 한다.
        for(int j=0; j<a_k ; j++){
            for(int l=0 ; l<a_k ; l++ ){
                result[IDX2C(b_n, i, j)]+=a[IDX2C(a_k, i, l)]*b[IDX2C(b_n, l, j)];
            }
        }
    }
}




int main(int argc, const char * argv[]) {
    // insert code here...
    
    int* a;
    int* b;
    int* c;
    int* result;
    
    // 행렬 메모리 할당
    a= new int[m*k];// a 는 mxk 행렬
    b= new int[k*n];// b 는 kxn 행렬
    c= new int[m*n];// c 는 mxn 행렬
    result= new int[m*n];
    
    
    
    int value_a[]={1,0,-3,-2,4,1};
    int value_b[]={2,-1,3,0,-5,2};
    int value_c[]={3,-1,-2,2};
    
    for(int i=0; i<m*k; i++){
        a[i]=value_a[i];
    }
    cout<<"a:\n";
    for(int i=0; i<m; i++){
        for(int j=0; j<k;j++){
            cout<<a[IDX2C(k, i, j)]<<" ";
        }
        cout<<endl;
    }
    //1 0 -3
    //-2 4 1
    
    cout<<"B\n";
    for(int i=0; i<k*n; i++){
        b[i]=value_b[i];
    }
    for(int i=0; i<k; i++){
        for(int j=0; j<n;j++){
            cout<<b[IDX2C(n, i, j)]<<" ";
        }
        cout<<endl;
    }
   // 2 -1
   // 3 0
    // -5 2
    
    cout<<"C:\n";
    for(int i=0; i<m*n; i++){
        c[i]=value_c[i];
    }
    for(int i=0; i<m; i++){
        for(int j=0; j<n;j++){
            cout<<c[IDX2C(n, i, j)]<<" ";
        }
        cout<<endl;
    }
   // 3 -1
   // -2 2
    
    
    cout<<"result:\n";
    for(int i=0; i<m*n; i++){
        result[i]=0;
    }
    
    for(int i=0; i<m; i++){
        for(int j=0; j<n;j++){
            cout<<result[IDX2C(n, i, j)]<<" ";
        }
        cout<<endl;
    }
    
    int nthreads = thread::hardware_concurrency();
    cout <<"cpu의 코어의 갯수: "<<nthreads <<endl;

    // 코어의 갯수 만큼 쓰레드 생성
    thread myThread[nthreads];
    
    
    for(int i=0;i<nthreads;i++){
        myThread[i] =  thread(multi_thread,a, b, result, m, k, n, i, nthreads);
    }
    
    
    // 각 쓰레드들의 종료를 확인한다.
  
    for(int i=0;i<nthreads;i++){
            myThread[i].join();
    
    }


    
    cout<<"곱샘 연산 후 result:\n";
    for(int i=0; i<m; i++){
        for(int j=0; j<n;j++){
            cout<<result[IDX2C(n, i, j)]<<" ";
        }
        cout<<endl;
    }
    
    
    
    
    
    
    
    // 덧샘 연산
    // 반복문을 통하여 쓰레드 배열의 번호를 함수의 인자로 전달한다.
    for(int i=0;i<nthreads;i++){
        myThread[i] =  thread(add_thread,result,c,result,m*n,i,nthreads);
    }
    
    // 각 쓰레드들의 종료를 확인한다.
    for(int i=0;i<nthreads;i++){
        myThread[i].join();
    }

    
    cout<<"덧샘 연산 후 result:\n";
    for(int i=0; i<m; i++){
        for(int j=0; j<n;j++){
            cout<<result[IDX2C(n, i, j)]<<" ";
        }
        cout<<endl;
    }
  
    
    return 0;
    
    
}
