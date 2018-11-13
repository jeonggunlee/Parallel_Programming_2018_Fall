# 병렬 프로그래밍 / 2018년 가을 (종합설계)

*  *  *

## [질문 올려주세요](https://github.com/jeonggunlee/Parallel_Programming_2018_Fall/blob/master/QnA.md)

## [알림] 11월 1일(목) 오후 5시까지 과제에 따른 중간 보고서 제출
*  *  *

### 강의 교수
 이정근 (Jeong-Gun Lee, http://www.onchip.net)
 
| Email | 전화번호 | 연구실 | 면담시간 |
| :---: | :---: | ------ | ----- |
| jeonggun.lee@gmail.com | 033-248-2312 | 공학관 1306 | 이메일 연락후 면담가능 |

### 조교
 정다운

| Email | 전화번호 | 연구실 |
| :---: | :---: | ------ |
| kozer9@gmail.com  | N.A. | 공학관 3층 데이터베이스 연구실 |

### 교과목 설명
이 과목은 **소프트웨어융합대학** 기초를 수강한 학부 상급생을 대상으로 병렬처리 환경의 소개와 병렬프로그래밍 기법에 대하여 입문수준의 지식을 제공함을 목적으로 한다. 현재 병렬 프로그래밍은 빅데이터/머신러닝 분야에서 크게 이슈가 되고 있고, 병렬 프로그래밍은 선택적인 지식에서 벗어나 컴퓨터 전문가로서 필수적인 지식으로 요구되고 있다. 특히 최근 AMD 및 NVIDIA의 고성능 GPU 프로그래밍에 대해서 심도 있게 살펴보며, 산업체 수요가 많은 *NVIDIA*의 *CUDA* 프로그래밍에 대해서 집중적으로 살펴본다.

본 강의에 사용되는 강의 자료와 실습 코드 등은 Github를 통해서 전달될 예정이며, https://github.com/jeonggunlee/Parallel_Programming_2018_Fall 를 통해 접근할 수 있다.

본강의는 기업이 제시하는 기업형 문제를 하나의 프로젝트로 할당 받아 한학기 동안 진행하는 ```종합설계형 교과목```입니다. 할당되는 프로젝트는 병렬 컴퓨팅 기반의 기업과제로써 학생들은 4명이 한팀을 이루어 프로젝트를 진행하게 됩니다. 팀의 최종 결과물(보고서, 개발소스)는 모두 ```Github```를 통하여 공개하며, 오픈소스화 하는 것이 최종 미션입니다.

## [캡스톤 상세 내용](https://github.com/jeonggunlee/Parallel_Programming_2018_Fall/blob/master/Capstone/readme.md)

*  *  *

### 강의 교재 교재
  - 교수자 PPT 자료 및 보조 문서
  - 특별히 책으로 된 교재는 없습니다. 다만, 필요에 따라 병렬 프로그래밍에 도움이 되는 문서들을 업로드할 계획입니다.
  - 필요에 따라 **동영상 강의 클립**이 제공될 수 있습니다.
  
## Schedule (스케쥴)
  - **1주:**
    -	병렬프로그래밍소개: 병렬 프로그래밍 개요 / 병렬 컴퓨터 구조
    - 동영상 시청
        - [[T타임] 가상현실부터 인공지능까지…미래기술은 ‘GPU’에 달려있다!](https://www.youtube.com/watch?v=srLim-zAAIs)
        - [GPU 기술의 세계 - 이용덕 엔비디아 지사장 / YTN 사이언스](https://www.youtube.com/watch?v=34uW5k77AOA)
  - **2주:**
    - 병렬프로그래밍 모델 소개 [PPT](https://github.com/jeonggunlee/Parallel_Programming_2018_Fall/blob/master/PPTs/Intro_PC_Under.pdf)
    -	병렬프로그래밍 = *OpenMP* 간략 소개
    - *[LAB 1](https://github.com/jeonggunlee/Parallel_Programming_2018_Fall/blob/master/LAB/Lab1/openmp.md)*: OpenMP를 활용한 매트릭스 곱셈 구현 및 평가
  - **3주:**
    -	[GPU 병렬 프로그래밍 기초 1](https://github.com/jeonggunlee/Parallel_Programming_2018_Fall/blob/master/PPTs/01_CUDA_I_2pages.pdf)
    - *LAB 2*: Amazon 클라우드를 활용한 GPU 실습 환경 구축 / Google 클라우드에서 GPU 사용법 소개
  - **4주:**
    -	GPU 병렬 프로그래밍 기초 2
    - *LAB 3*: HelloCUDA !
       - Grid-Block-Thread 구조 실험
       - DeviceQuery / Vector Addition 구현
  - **5주:**
    -	GPU 아키텍쳐: SIMD / SIMT
    - *LAB 4*: 이전 Lab 계속
       - Grid-Block-Thread 구조 실험
       - DeviceQuery / Vector Addition 구현
  - **6주:**
    -	GPU 병렬 프로그래밍 기초 3
    - *LAB 5*: Matrix Transpose 구현 (기본)
  - **7주:**
    - 종합 설계 중간 발표
  - **8주:**
    -	[GPU 병렬프로그래밍 최적화 1](https://github.com/jeonggunlee/Parallel_Programming_2018_Fall/blob/master/PPTs/02_CUDA%20II_2pages.pdf)
    - *LAB 6*: 최적화 실습
  - **9주:**
    -	[GPU 병렬프로그래밍 최적화 2](https://github.com/jeonggunlee/Parallel_Programming_2018_Fall/blob/master/PPTs/02_CUDA%20II_2pages.pdf)
    - *LAB 7*: 최적화 실습
  - **10주:**
    -	[Parallel Transpose 최적화 ](https://github.com/jeonggunlee/Parallel_Programming_2018_Fall/blob/master/PPTs/02_CUDA%20II_2pages.pdf)
    - *LAB 8*: Transpose 최적화 실습 1
  - **11주:**
    -	종합 설계 발표 1
    - *LAB 9*: 종합 설계 과제 진행
  - **12주:**
    -	종합 설계 발표 2
    - *LAB 10*: 종합 설계 과제 진행
  - **13주:**
    -	종합 설계 발표 3
    - *LAB 11*: 종합 설계 과제 진행
  - **14주:**
    -	종합 설계 최종 발표회 1
    - 종합 설계 최종 발표회 2

## 평가 방식
  - 종합설계 과제 발표 30%
  - 종합설계 과제 수행 40%
  - 실습 30%
  - 과제 copy 시에 해당 과제 0점 처리
  - 팀별 프로젝트 진행시, 각 참여 학생의 역활에 대한 평가 진행
  - 최종 종합설계 과제는 ```Github```를 통하여 제출함!

## REFERENCES (참조Sites)
  - OpenMP 동영상 강의 (KISTI) - 한국어 !~
     - [1. OpenMP 소개](https://www.youtube.com/watch?v=6rXJneScWFM)
     - [2. OpenMP 지시어](https://www.youtube.com/watch?v=_K8PTVYjDmc)
     - [3. OpenMP 환경변수](https://www.youtube.com/watch?v=LKrEWu_5dSQ&t=137s)
     - [4. OpenMP를 이용한 병렬화](https://www.youtube.com/watch?v=xuEo51976d8)
     - [5. OpenMP 고급 사용법](https://www.youtube.com/watch?v=LP3IuENi17M)
  - CUDA Sample Directory: C:\ProgramData\NVIDIA Corporation\CUDA Samples
  - CUDA 최고의강좌! 강추! Udacity [Intro to Parallel Programming](https://www.youtube.com/watch?v=F620ommtjqk&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2)
  - Udacity [High Performance Computer Architecture](https://www.youtube.com/watch?v=tawb_aeYQ2g&list=PLAwxTw4SYaPmqpjgrmf4-DGlaeV0om4iP&index=1)
  - Udacity [High Performance Computing](https://www.youtube.com/watch?v=grD5en6_IiQ&list=PLAwxTw4SYaPk8NaXIiFQXWK6VPnrtMRXC)
  - [CUDA LECTURE](https://www.youtube.com/watch?v=sxhvmTveO2A) - Oklahoma State University
  - 코딩 실습을 위한 [클라우드 설정(AWS)](https://github.com/jeonggunlee/CUDATeaching/blob/master/gpu4cloud.md) 
  - [머신러닝에 사용되는 *CNN*의 CUDA 구현](https://sites.google.com/site/5kk73gpu2013/assignment/cnn)
