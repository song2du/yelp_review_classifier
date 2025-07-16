import numpy as np
import torch
import os
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed, cuda): # 실험의 재현성 확보
    np.random.seed(seed) 
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath): # 결과 저장할 폴더를 만드는 함수
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def make_train_state(args): # 훈련 관련 설정 생성
    return {
        'stop_early': False, # 조기 종료 여부
        'early_stopping_step': 0, # 조기 종료까지 단계
        'early_stopping_best_val': 1e8, # 조기 종료시 최적 값
        'learning_rate': args.learning_rate, # 학습률
        'epoch_index': 0, # 현제 에폭의 인덱스
        'train_loss': [], # 훈련 시 손실 함수 값 
        'train_acc': [], # 훈련 시 정확도 
        'train_f1': [], # 훈련 시 f1
        'val_loss': [], # 검증 시 손실 함수 값
        'val_acc': [], # 검증 시 정확도
        'val_f1': [], # 검증 시 f1
        'test_loss': -1, # 테스트 시 손실 함수 값
        'test_acc': -1, # 테스트 시 정확도
        'test_f1': -1, # 테스트 시 f1
        "model_filename": args.model_state_file # 모델 저장할 파일 이름
    }

def compute_accuracy(y_pred, y_target):
    y_true = y_target.cpu().numpy() # sklearn이 이해할 수 있도록 텐서를 numpy 배열로 변환
    y_pred_labels = (torch.sigmoid(y_pred) > 0.5).long() # y_pred (원본 점수)를 레이블 (0 or 1)로 변환
    y_pred_labels = y_pred_labels.detach().cpu().numpy() # 변환된 예측 레이블 (텐서)를 numpy로 변환

    return accuracy_score(y_true, y_pred_labels)

def compute_f1(y_pred, y_target):
    y_true = y_target.cpu().numpy() 
    y_pred_labels = (torch.sigmoid(y_pred) > 0.5).long() 
    y_pred_labels = y_pred_labels.detach().cpu().numpy() 
    return f1_score(y_true, y_pred_labels, average='macro')