# Deepest Season 4 Challenge Q2
## Judging the book by its cover: 논문 제목 & 저자만 보고 accept 여부 맞추기 (Sang-gil Lee)

## Introduction
딥러닝이 과연 논문 리뷰어를 대체할 수 있을까요? 논문 제목과 저자만 보고 accept or reject을 예측할 수 있는 모델이 존재할 수 있을까요?

이 챌린지는 “당연히 안될 것 같은” 이 질문으로부터 출발한, (hopefully) 재미있기를 바라는 task 입니다.

## Test set: ICLR 2018 list
ICLR 2018의 accept & reject 전체 list를 test set으로 삼기로 했습니다. Openreview.net 에서 가져왔으며, Oral + Poster 는 Accept, Invite to workshop + Reject는 Reject로 설정하여 binary classification problem 으로 구성하였습니다.

ICLR 2018 리스트는 training set에 추가될 수 없습니다.

## Training set: ICLR 2017 + anything you want!
Training data의 starter kit으로 ICLR 2017의 accept & reject list를 2018과 같은 형식으로 분류하여 제작해 놓았습니다. 추가 데이터는 여러분이 원하는 대로, 원하는 방식으로 얼마든지 추가해도 좋습니다. 단, ICLR 2018 리스트는 상기했듯이 추가할 수 없습니다.

“ICLR 2018 reject되고 ICML 2018 accept 된 논문들도 있지 않나요?”

여기서 이 task가 ill-posed problem인 것을 알 수 있습니다;;; 이러한 논문들을 training set에 어떻게 포함시킬 지는 여러분의 판단에 맡기겠습니다.

## Baseline: the dystopia (test accuracy 63.32%)
베이스라인은 가장 단순한 형태의 RNN + MLP 입니다. 논문 제목과 저자를 word tokenization한 후, 제목 token과 저자 token을 독립된 LSTM에 각각 적용한 뒤 마지막 step의 hidden state를 논문 제목과 저자의 summary로 가정하였습니다. 이 summary를 concat하여 MLP 를 적용하여 softmax를 통해 paper decision 을 하였습니다 (0: accept, 1: reject).

불행히도 베이스라인 모델은 모든 논문을 reject하기로 결정했습니다. (전부 output=1) ICLR 2018 test set accuracy는 모든 논문 reject시 63.32% 입니다. (accept논문 336개, reject 논문 580개)

## The task
test set의 정보를 직접 학습에 반영하는 cheating을 하지 않는 선에서 (ex: 논문 저자에 Joshua Bengio가 포함된다면 무조건 accept 등…), 수단과 방법을 가리지 않고 test set accuracy가 baseline 이상을 기록하는 모델을 설계하는 것입니다.

단순한 딥러닝 모델 설계만으로는 부족할 것입니다. 확률 모델링 기법, 클래식 머신러닝 기법 등 여러분이 알고 있는 모든 지식을 동원하여 이 “말도 안되는” task가 baseline을 넘어설 수 있는지 알아봅시다!

## The prize

1. 가장 높은 test accuracy를 기록한 팀

2. 가장 다양한 시도를 해본 팀

Task의 특성을 반영하여 두가지 평가 기준을 도입하였습니다. 따라서 결과물 제출 시에는 여러분이 시도했던 모든 방법들, 왜 그 방법을 시도해 보았는지에 대한 이유, (망한) 결과들을 간단하게 서술해서 2번 prize도 노려보도록 합시다!

만약 아무도 baseline test accuracy를 넘지 못한다면 2번 prize를 2팀 선정할 예정입니다.
