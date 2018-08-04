# 병렬 프로그래밍 / 2018년 가을

### 강의 교수
 이정근 (Jeong-Gun Lee, http://www.onchip.net)
 
| Email | 전화번호 | 연구실 | 면담시간 |
| :---: | :---: | ------ | ----- |
| jeonggun.lee@gmail.com | 033-248-2312 | 공학관 1306 | 이메일 연락후 면담가능 |

### 조교
 정다운

| Email | 전화번호 | 연구실 |
| :---: | :---: | ------ |
| xxx@gmail.com | 033-248-2312 | 공학관 1306 |

### 교과목 설명
이 과목은 **소프트웨어융합대학** 기초를 수강한 학부 상급생을 대상으로 병렬처리 환경의 소개와 병렬프로그래밍 기법에 대하여 입문수준의 지식을 제공함을 목적으로 한다. 현재 병렬 프로그래밍은 빅데이터/머신러닝 분야에서 크게 이슈가 되고 있고, 병렬 프로그래밍은 선택적인 지식에서 벗어나 컴퓨터 전문가로서 필수적인 지식으로 요구되고 있다. 특히 최근 AMD 및 NVIDIA의 고성능 GPU 프로그래밍에 대해서 심도 있게 살펴보며, 산업체 수요가 많은 *NVIDIA*의 *CUDA* 프로그래밍에 대해서 집중적으로 살펴본다.

본 강의에 사용되는 강의 자료와 실습 코드 등은 Github를 통해서 전달될 예정이며, https://github.com/jeonggunlee/Parallel_Programming_2018_Fall 를 통해 접근할 수 있다.

### 강의 교재 교재
 
## Schedule (스케쥴)
  - **1주:**
    -	병렬프로그래밍소개: 병렬 프로그래밍 개요 / 병렬 컴퓨터 구조
  - **2주:**
    - 병렬프로그래밍 모델 소개
    -	병렬프로그래밍 = *OpenMP* 간략 소개
    - *LAB 1*: OpenMP를 활용한 매트릭스 곱셈 구현 및 평가
  - **3주:**
    -	GPU 병렬 프로그래밍 기초 1
    - *LAB 2*: 아마존 클라우드를 활용한 GPU 실습 환경 구축
  - **4주:**
    -	GPU 병렬 프로그래밍 기초 2
    - *LAB 3*:
  - **5주:**
    -	GPU 아키텍쳐: SIMD / SIMT
    - *LAB 4*:
  - **6주:**
    -	GPU 병렬 프로그래밍 기초 3
    - *LAB 5*:
  - **7주:**
    -	중간 고사
  - **8주:**
    -	GPU 병렬프로그래밍 최적화 1
    - *LAB 6*:
  - **9주:**
    -	GPU 병렬프로그래밍 최적화 2
    - *LAB 7*:
  - **10주:**
    -	Parallel Transpose 최적화 1
    - *LAB 8*:
  - **11주:**
    -	Parallel Transpose 최적화 2
    - *LAB 9*:
  - **12주:**
    -	Parallel Reduction 최적화 1
    - *LAB 10*:
  - **13주:**
    -	Parallel Reduction 최적화 2
    - *LAB 11*:
  - **14주:**
    -	병렬 CUDA 벡터곱 최적화 소개
    - *LAB 12*:

## 평가 방식
  - 중간고사 25%, 기말고사 25%
  - 프로젝트 25%, 실습 25%
  - 과제 copy 시에 해당 과제 0점 처리
  - 팀별 프로젝트 진행시, 각 참여 학생의 역활에 대한 평가 진행

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
