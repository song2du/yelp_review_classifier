import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import notebook

class MlpClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 모델 초기화 및 첫번째 층과 두번째 층 출력 크기 정의
        super(MlpClassifier, self).__init__() # nn.Module을 상속받아 초기화 진행
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_in, apply_softmax=False):
        # 정방향 계산 
        prediction_vector = self.mlp_layers(x_in) # 지정해 놓은 레이어에 인풋 통과 시키기
        
        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1) # 예측 결과를 사람이 보기 편하게 만듦
        
        return prediction_vector