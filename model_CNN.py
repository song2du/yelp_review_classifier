import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CnnClassifier(nn.Module):
    def __init__(self, embedding_dim, vocab_size, num_channels, num_classes, dropout_p, **kwargs):
        # 모델 초기화 및 첫번째 층과 두번째 층 출력 크기 정의
        super(CnnClassifier, self).__init__() # nn.Module에서 상속 받아 초기화

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=0)

        self.convnet = nn.Sequential( # 특징 추출기 역할
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=num_channels, kernel_size=3), # 원본 데이터에서 패턴 학습
            nn.ReLU(), # 비선형성 부과
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3, stride=2), # 학습된 피처맵을 압축하는 역할 
            nn.ReLU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, # 압축한 피처맵에서 특징 학습
                      kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1) # 각 채널마다 특징 벡터의 길이를 1로 줄임
        )
        self.fc = nn.Linear(num_channels, num_classes) # 최종 분류기 역할
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x_in, apply_softmax=False):
        # 정방향 계산
        x_in = x_in.long() 
        x_embedded = self.embedding(x_in).permute(0, 2, 1)
        features = self.convnet(x_embedded).squeeze(dim=2)
        features = self.dropout(features)
        prediction_vector = self.fc(features)

        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)
        
        return prediction_vector
