# 학사 졸업논문  
# SENet에 Multihead_self_attention을 추가한 구조 모델링  

## INDEX  
1. 서론   
    1) 연구의 필요성   
Vision Transformer는 자연어 처리에서 많이 사용되었던 Transformer를 이미지 처리에 도입한 것이다. ViT의 핵심 아이디어인 multi-head attention의 특성을 탐구해 보고자, ResNet에 Self-Attention Block을 추가해 Self-attention의 특징과 Convolutional Network에 이를 적용했을 때의 성능차이에 대한 연구를 해 보고자 한다. ILSVRC 2017 우승 모델인 SENet의 핵심 아이디어 squeeze and excitation이 Image에 대하여 attention을 적용한 기법이다. ResNet에 SE-Block만 추가한 구조가 되는데, 이 SE-Block을 변형시켜 attention에 self-attention을 더한 모델을 연구해 보고자 한다.   
    2) 관련 연구   
  squeeze and excitation: SENet에는 SE-Block이 존재하는데, 그 안에서는 squeeze and excitation이 일어난다. squeeze 단계에서는 global average pooling을 사용해 채널마다 하나의 값을 추출한다. excitation 단계에서는 이 값들을 사용해 각 채널의 중요도를 계산하고, 중요한 채널을 강조하고 중요하지 않은 채널을 억제한다. 채널별로 입력에 대한 정보를 학습하여 중요한 채널에 높은 가중치를 부여하는 방식으로, 채널의 중요성을 동적으로 학습한다. 이를 통해, 입력의 특성을 더욱 잘 추출하여 성능을 향상한다.
  attention: 입력 시퀀스 내에서 서로 다른 위치 간의 상관관계를 고려하여 가중치를 계산하는 방법이다. 이를 통해 입력 시퀀스 내에서 어떤 위치들이 중요한지를 결정할 수 있다. SE-Block에서는 squeeze 연산(Global Average Pooling)으로 특성 맵의 채널별로 평균값을 계산하고, 이를 이용하여 채널별로 중요도를 계산하는 것이다. 그리고 이 중요도는 각 채널의 가중치로 사용되어, 더 중요한 채널은 더 큰 가중치를 가지게 된다. 이렇게 계산된 가중치를 이용하여 특성 맵의 정보를 조절하게 된다. 따라서 squeeze 연산은 특성 맵 내에서 특정 채널이나 위치에 집중하는 attention 기법의 일종으로 볼 수 있다.
  self-attention: 입력 시퀀스 내의 각 위치를 서로 연결하여 가중치를 계산하고, 이를 기반으로 입력 시퀀스 내의 각 위치가 다른 위치에 미치는 영향력을 결정하는 방법이다. 입력 데이터를 쿼리(query), 키(key), 밸류(value)로 변환한 뒤, 쿼리와 모든 키 간의 유사도를 계산한다. 이 유사도에 softmax 함수를 적용하여 attention weight를 계산한다. attention weight와 value의 행렬을 곱한 값이 출력된다. 본 논문에서는 기존 SE-Block을 변형시켜 Squeeze & Excitation을 구현한 뒤, 그 값을 활용해 self-attention을 한다. 이는 입력 데이터 내에서 특성 간의 의존성을 파악하거나, 전체 데이터에서 중요한 정보를 추출하는 데 유용하다.
    3) 연구계획   
기존의 SEBlock[2]을 수정해서 SENet_self를 구축해 새로운 네트워크를 만들고자 한다. SEBlock의 excitation이 끝나고 난 뒤, 값을 Flatten 시키고 MSA를 시도한다. 나온 값은 다시 처음 input 값과 차원을 같게 만들어 준다. 이렇게 새로 만든 SENet_self 네트워크를 다른 네트워크(ResNet, SENet)와 비교해 보고자 한다.


2. 본문   
    1) 방법   
본래 SEBlock에 구현되어있던 SEBlock에 MSA(emb_size, num_heads)를 추가한다.      
먼저 excitation의 output을 MSA에 입력시키기 위해 차원을 변경해 준다.
x = x.flatten(2).transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
다음으로 x를 multiheadattention 에 넣어준다.
x = self.multihead_attn(x, x, x)[0] # query, key, value 벡터로 x가 같게 들어감
Query: \Q_i = XQ_i^Q / Key:  / Value: 
, , 는 각 head 에 해당하는 Query, Key, Value에 대한 가중치 행렬.
이후, 각 head에서 계산된 Query와 Key의 내적값을 로 나눠주고, Softmax 함수를 적용해 각 head에서 계산된 가중치를 구한다.

이렇게 구한 각 head에서 계산된 값들을 다시 합쳐서 출력값을 구한다.
    2) Experiment   
    3) 분석 & 비교   
3. 결론   
참고문헌  

