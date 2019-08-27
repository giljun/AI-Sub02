# 분류 알고리즘을 활용한 자연어 처리

## 프로젝트 개요

- 본 프로젝트에서는 자연어 처리 실습을 하기 위한 머신 러닝 기반의 Classifier의 개념을 익히고 실행하여 주어진 영화 평점과 리뷰 자료들을 분석합니다.

- 이전에 구현한 Linear regression 모델의 기반 지식으로 Logistic regression과 Naive bayes classifier를 통하여 선형적 특성을 갖지 않는 데이터를 분석합니다.

- 또한 자연어 처리 과정에서 Pre-processing 과정의 성능과 그 중요성을 실습을 통하여 알아볼 수 있습니다.

- 프로젝트 설명

  - 문장 데이터를 training 및 test 데이터를 사용하기에 자연어 처리 과정을 통하여 머신러닝 알고리즘에 사용할 수 있게 변환합니다.
  - 영화 평점 데이터의 label은 긍정적인 평점인 1값과 부정적인 평점 0값으로 이루어져 있기에 지도학습 중에서 분류(classification) 알고리즘인 Logistic regression과 Naive bayes classifier를 사용하여 학습합니다.
  - 새로운 영화 댓글을 입력했을 때 그 댓글이 긍정적인지 부정적인지를 예측해주는 slack app을 구현하는 것을 목표로 합니다.

  

## 머신 러닝 파트

- 머신 러닝 파트에서는 영화 댓글에 따른 평점 분석기를 구현
- 다음 그림과 같이 머신 러닝의 구조는 5가지 파트로 분류할 수 있습니다.
- 데이터를 읽기
  - ratings_train.txt, ratings_test.txt 파일을 읽고 트레이닝 데이터와 테스트 데이터로 저장합니다.
  - 데이터 자체가 트레이닝용과 테스트용으로 나누어져 있기에 추가적으로 데이터를 나누는 작업을 할 필요는 없습니다.
  - ratings_train.txt파일은 영화 댓글과 그에 따른 label 값인 0 or 1이 저장되어 있습니다.
  - 첫 번째 열을 제외한 document, label 열을 읽습니다.
- 문장 데이터 자연어 처리하기
  - 문장 데이터를 처리하기 위해서 자연어 처리 과정을 거쳐 컴퓨터가 처리할 수 있는 데이터로 변환합니다.
  - 문장을 형태소 별로 나누기 위하여 KoNLPy를 사용하여 tokenizing을 합니다.
  - 이후, 사용하게 될 classification 기법의 학습 데이터로 사용하기 적합한 one-hot 임베딩을 적용합니다.
  - One-hot 임베딩은 학습에 사용된 모든 token의 종류들 중에 트레이닝 문장에서 사용되는 token에 1을 붙이고 나머지를 0으로 채우는 방식입니다.
  - 따라서 각 문장별로 token들에 해당되는 index 값이 0, 1을 갖게 되는 트레이닝 데이터를 얻을 수 있습니다.
  - 이를 구현하기 위해선 모든 token들의 정보가 필요하고 이는 word_indices라는 사전 데이터로 저장합니다.
  - 마지막 전 처리 과정으로 one-hot 임베딩 과정으로 만들어진 행렬의 크기가 매우 크기 때문에 머신 러닝 모델을 학습할 시 복잡도 문제를 덜고자 sparse matrix를 사용합니다.
  - sparse matrix는 유의미한 데이터보다 무의미한 데이터 즉 0값이 많은 비중을 차지하는 행렬의 경우 그 외의 값을 갖는 데이터의 인덱스 정보만을 기억하여 메모리를 효율적으로 사용하는 기법입니다.
  - 전 처리 과정 : tokenizing → word_indices → one hot embedding → sparse matrix
- 알고리즘을 불러와 학습
  - 사용할 데이터가 label이 존재하고 불 연속적인 값을 가지고 있기에 분류 기법을 사용합니다.
  - 분류 기법 중 선형으로 데이터를 근사화 하는 기법으로 logistic regression과 naive bayes classifier 모델을 사용하고 데이터를 학습합니다.
- 학습된 결과물로 테스트 데이터를 통하여 정확도 측정
  - 학습된 결과물로 clf, clf2, word_indices를 얻을 수 있으며, 각각 logistic regression모델, Naive bayes classifier 모델, token 사전 데이터를 의미합니다.
  - 이후, 테스트 데이터를 사용하여 정확도를 확인합니다. Scikit-learn에서 multiclass classification의 정확도를 구하기 위해서 제공하는 accuracy_score를 사용합니다.
  - 참고
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
- 학습된 결과물을 저장
  - 학습된 Logistic regression과 Naive bayes classifier 모델의 class 정보를 파이썬의 pickle  기능을 통하여 model.clf에 저장합니다. Flask 서버 코드에서는 이 정보를 불러와 사용하여 학습의 반복을 피할 수 있습니다.



## slack app 구현 파트

- 로컬에서 프로그래밍한 머신 러닝 알고리즘을 slack app을 사용하기 위해서 python slack client를 기반으로 한 flask 서버를 구동합니다. 
- Flask 서버에서는 slack app에서 전달받은 영화 댓글 정보를 받고 머신 러닝에서 학습된 정보를 가지고 있는 model.clf파일을 통하여 긍정적인 댓글인지 부정적인 댓글인지 예측하고 전달합니다.
- 추가적으로 2주차에서는 database를 사용하여 새로운 데이터들을 저장하고 이를 활용하여 새로운 학습 데이터로서 활용할 수 있습니다.
- 이를 구현하기 위하여 SQLite를 사용하여 app.db 파일을 생성합니다.
- 이번 프로젝트에서는 간단하게 slack에서 입력하는 영화 댓글의 문장을 app.db에 저장하는 것을 목표로 합니다.
- 파이썬에서 구현 가능한 sqlite3 라이브러리를 활용하여 db 테이블을 갖춘 app.db파일을 생성하고 flask 서버 코드에서 slack으로부터 입력 받은 데이터를 app.db파일에 저장하게 합니다.
- 이후, 이를 활용하여 입력하는 댓글의 평점 또한 받게 하여 수 많은 사람들이 이 영화 평점 분석기를 이용했을 시 얻어지는 database 정보를 가지고 새롭게 트레이닝 셋을 업데이트할 수 있습니다.
- 이번 프로젝트를 진행하면서 기본적인 Machine learning classifier 들에 대해서 이해하고 사용할 수 있으며 Pre-processing을 통하여 자연어 데이터를 효율적으로 처리하는 방법을 습득합니다.
- 최종 프로젝트에서는 영화 평점 및 댓글을 분석하여 평점이 좋은 영화와 나쁜 영화를 분류함으로 실제 환경에서 자연어 처리 기술이 어떻게 사용되는지 알 수 있습니다.
- 또한 새로운 데이터를 활용하기 위하여 database 파트를 구성하여 slack app 을 통하여 얻어진 데이터들을 업데이트합니다.



## Naive bayes classifier algorithm  구현 파트

- 이 과정을 심화 과정으로 Naive bayes classifier algorithm의 심도 있는 이해를 위하여 그 동작 과정을 구현합니다.
- 알고리즘은 Naive_Bayes_Classifier 클래스를 구현하여 작동되며 다음 6가지 함수로 구성되어 있습니다.
  - log_likelihoods_naivebayes() :
    - Feature 데이터를 받아 label(class) 값에 해당되는 likelihood 값들을 naive 방식으로 구하고 그 값이 로그 값을 리턴합니다.
  - class_posteriors () :
    - Feature 데이터를 받아 label(class) 값에 해당되는 posterior 값들을 구하고 그 값의 로그 값을 리턴합니다.
  - classify ():
    - Feature 데이터에 해당되는 posterior 값들( class 개수 )을 불러와 비교하여 더 높은 확률을 갖는 class를 리턴합니다.
  - train () :
    - 트레이닝 데이터를 받아 학습합니다. 학습 후, 각 class에 해당하는 prior 값과 likelihood 값을 업데이트합니다.
  - predict () :
    - 예측하고 싶은 문장의 전 처리된 행렬을 받아 예측된 class를 리턴합니다.
  - score () :
    - 테스트를 데이터로 받아 예측된 데이터(predict 함수)와 테스트 데이터의 label 값을 비교하여 정확도를 계산합니다.



## Logistic regression algorithm 구현 파트

- 이 과정을 심화 과정으로 Logistic regression algorithm의 심도 있는 이해를 위하여 그 동작 과정을 구현합니다.
- 알고리즘은 Logistic_Regression_Classifier 클래스를 구현하여 작동되며 다음 7가지 함수로 구성되어 있습니다.
  - sigmoid () :
    - 인풋 값의 sigmoid 함수 값을 리턴합니다.
  - prediction () :
    - 가중치 값인 beta 값들을 받아서 예측 값 P(class=1|train data)을 계산합니다.
    - 예측 값 계산은 데이터와 가중치 간의 선형합의 logistic function 값으로 얻을 수 있습니다.
  - gradient_beta () :
    - 가중치 값인 beta 값에 해당되는 gradient 값을 계산하고 learning rate를 곱하여 출력합니다.
    - Gradient 값은 데이터 포인터에 대한 기울기를 의미하고 손실함수에 대한 가중치 값의 미분 형태로 얻을 수 있습니다.
  - train () :
    - 트레이닝 데이터를 받아 학습합니다.
    - 학습 후, sigmoid 함수로 근사하는 최적의 가중치 값을 업데이트합니다.
  - predict () :
    - 예측하고 싶은 문장의 전 처리된 행렬을 받아 예측된 class를 리턴합니다.
  - score() : 
    - 테스트 데이터를 받아 예측된 데이터(predict 함수)와 테스트 데이터의 label값을 비교하여 정확도를 계산합니다.



## 기술 스택

### KoNLPy 라이브러리

- KoNLPy는 한국어 정보처리를 위한 파이썬 오픈 소스 패키지입니다. 
- 기존의 영어를 위한 자연어 처리 패키지는 다수 존재하였고 이 패키지들은 한글을 지원하지 않거나 지원하지만 높은 수준의 처리가 불가능했습니다.
- KoNLPy는 여러가지 형태소 분석과 품사 판별 기능을 가지고 있으며 데이터 크롤링, 웹프로그래밍에 필요한 기능 또한 제공합니다.
- 본 프로젝트에서는 한글을 기반으로 한 자연어 처리를 구현하기에 KoNLPy가 사용됩니다.

### SQLite

- SQLite는 C 기반의 데이터베이스 라이브러리로 간단하지만 매우 높은 성능을 가지고 있습니다.
- 또한 데이터베이스가 파일로 저장되기 때문에 git 등에 커밋할 수 있고 이를 통해 환경에 제약 없이 개발할 수 있습니다.
- 본 과제에서는 SQLite를 이용해 간단한 로컬 데이터베이스를 만들고 데이터베이스 안에서 머신 러닝 자료를 관리합니다.



## 프로젝트 목표

### 1. Classification을 사용하여 영화 댓글 평점 분석 챗봇 구현하기

1. 영화 댓글, 평점 데이터를 읽고 트레이닝 데이터와 테스트 데이터로 저장하기
2. 한글 문장 데이터 tokenizing하기
3. token 정보가 저장된 사전을 만들고 one-hot 임베딩하기
4. 임베딩 된 데이터를 sparse matrix로 저장하기
5. scikit-learn에서 Logistic regression과 Naive bayes classifier 모델을 가져와 학습하기
6. 정확도 평가를 위한 accuracy 값 구하기
7. 테스트 데이터의 영화 댓글에 따른 긍정, 부정 리뷰 분류하기
8. Pickle을 사용하여 학습된 classification 모델이 저장된 model.clf 파일 저장하기
9. Python slack client를 사용하여 flask 서버 app.py 구현하기
10. Slack app에서 영화 댓글을 입력하고 긍정 or  부정 리뷰 출력하기
11. SQLite를 사용해서 slack에 입력 받을 데이터를 저장할 app.db 만들기
12. app.db에 새로운 데이터 업데이트 하기



### 2. Naive bayes classifier algorithm 구현 ( 심화 과정 )

1. Naive_Bayes_Classifier() 클래스 구현하기
2. train() 함수를 사용하여 트레이닝 데이터 학습하기
3. predict() 함수를 사용하여 테스트 데이터의 영화 댓글에 따른 긍정, 부정 리뷰 분류하기
4. score() 함수를 사용하여 테스트 데이터 정확도 측정하기



### 3. Logistic regression algorithm 구현 ( 심화 과정 )

1. Logistic_Regression_Classifier() 클래스 구현하기
2. train() 함수를 사용하여 트레이닝 데이터 학습하기
3. predict() 함수를 사용하여 테스트 데이터의 영화 댓글에 따른 긍정, 부정 리뷰 분류하기
4. score() 함수를 사용하여 테스트 데이터 정확도 측정하기