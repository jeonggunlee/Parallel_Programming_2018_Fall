/*
 1차 대학원생들과의 미팅 후 작성한 파일입니다.
 행렬의 연산에 대해서 알아보고 
 cpu상 병렬화를 구현하였습니다.
 
*/

#include <iostream>
#include <vector>
#include <thread>

using namespace std;


void multi(vector<vector<int>> A, vector<vector<int>> B,vector<vector<int>> *C){
    int row=A.size();
    int col=B.size();
    
    for(int i =0 ; i<row;i++){
        for(int j=0;j<col; j++){
            for(int k=0; k<col; k++){
                (*C)[i][j]+=A[i][k]*B[k][j];
            }
        }
    }
}

/*
 행렬의 곱샘은 i행 j의 연산이 다른 n행 m열 정보를 변경하지 않는다.
 그래서 어떤 행에 대해서 연산을 수행하게 되는지를 함수 인자값으로 넘겨 받아 수행 될 수 있게 하였다.
 */

void multi_thread(vector<vector<int>> A, vector<vector<int>> B,vector<vector<int>> *C,int th){
    int i= th;
    int col=B.size();
    
    
        for(int j=0;j<col; j++){
            for(int k=0; k<col; k++){
                (*C)[i][j]+=A[i][k]*B[k][j];
            }
        }
    
}




void disp(vector<vector<int>> C){
    int row=C.size();
    int col=C[0].size();
    
    cout<<"결과"<<endl;
    for(int i=0; i<row ; i++){
        for(int j=0;j<col; j++){
            cout<<C[i][j]<<" ";
        }
        cout<<endl;
    }
}


// 덧샘
void add(vector<vector<int>> A, vector<vector<int>> *B){
    int row=A.size();
    int col=B[0].size();
    
    for(int i=0; i<row;i++){
        for(int j=0; j<col;j++){
            (*B)[i][j]+=A[i][j];
        }
    }
    
}

void add_thread(vector<vector<int>> A, vector<vector<int>> *B , int i){
    int col=B[0].size();
    
        for(int j=0; j<col;j++){
            (*B)[i][j]+=A[i][j];
        }
    
}


// main에서는 쓰레드를 사용하지 않은 함수는 빠져있습니다. 

int main() {
    
    /*
      대학생과제 파일에 있는 대로 3가지 행렬을 선언하였으며
      이후 결과를 저장할 행렬을 하나 더 생성후
      결과를 저장할 행렬을 가리키는 포인터를 생성하였다.
    */
    
    vector<vector<int>> da={{1,0,-3},{-2,4,1}};
    vector<vector<int>> db={{2,-1},{3,0},{-5,2}};
    vector<vector<int>> dc={{3,-1},{-2,2}};
    vector<vector<int>> dd={{0,0},{0,0}}; // 결과 저장

    vector<vector<int>> *d=&dd; // 행렬을 가리키느 포인터 
    
    int size=da.size();// 행렬의 행의 크기를 구한다.
    
    thread myThread[size];// 행렬의 행의 크기대로 쓰레드를 선언한다.
    
    // 반복문을 통하여 쓰레드 배열의 번호를 함수의 인자로 전달한다.
    for(int i=0;i<size;i++){
    myThread[i] =  thread(multi_thread,da, db,d, i);
    }
    
    // 각 쓰레드들의 종료를 확인한다.
    for(int i=0;i<size;i++){
        myThread[i].join();
    }
    
    // 결과를 출력한다.
    disp(dd);
    
    
    // 행렬의 사이즈를 구한다.
    size=dc.size();
    
    
    for(int i=0;i<size;i++){
        myThread[i] =  thread(add_thread,dc,d, i);
    }
    
    for(int i=0;i<size;i++){
        myThread[i].join();
    }
    
    disp(dd);
    
    return  0;
}

/*
행렬에 병렬화에 대한 생각
행렬 연산과정에서 원소 하나의 연산과정이 다른 원소 값에 영향을 미치지 않는 작업이 많으므로
행렬의 연산은 충분히 병렬화 할 수 있을 것이라고 생각합니다. 
*/

/*
어려운점
행렬의 크기 정보를 어떻게 함수에 전달하는 방법과
thread를 처음 사용해봐서 어려웠습니다.
*/
