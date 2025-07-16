# visualize.py
import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(all_results):
    """
    훈련 결과를 받아 손실 및 정확도 그래프를 그리고 파일로 저장합니다.
    """
    # 결과를 보기 쉽게 DataFrame으로 변환
    df = pd.DataFrame(all_results).T

    # 각 모델의 훈련/검증 손실 및 정확도 리스트를 추출
    mlp_train_loss = df.loc['mlp', 'train_loss']
    mlp_val_loss = df.loc['mlp', 'val_loss']
    cnn_train_loss = df.loc['cnn', 'train_loss']
    cnn_val_loss = df.loc['cnn', 'val_loss']

    mlp_train_acc = df.loc['mlp', 'train_acc']
    mlp_val_acc = df.loc['mlp', 'val_acc']
    cnn_train_acc = df.loc['cnn', 'train_acc']
    cnn_val_acc = df.loc['cnn', 'val_acc']

    # 1행 2열의 서브플롯 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Loss 그래프
    ax1.plot(mlp_train_loss, 'r--', label='MLP Train Loss')
    ax1.plot(mlp_val_loss, 'r-', label='MLP Val Loss')
    ax1.plot(cnn_train_loss, 'b--', label='CNN Train Loss')
    ax1.plot(cnn_val_loss, 'b-', label='CNN Val Loss')
    ax1.set_title('Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy 그래프
    ax2.plot(mlp_train_acc, 'r--', label='MLP Train Accuracy')
    ax2.plot(mlp_val_acc, 'r-', label='MLP Val Accuracy')
    ax2.plot(cnn_train_acc, 'b--', label='CNN Train Accuracy')
    ax2.plot(cnn_val_acc, 'b-', label='CNN Val Accuracy')
    ax2.set_title('Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # 그래프 저장
    plt.savefig('training_comparison.png')
    print("그래프가 'training_comparison.png' 파일로 저장되었습니다.")

if __name__ == '__main__':
    with open('results.json', 'r') as fp:
        all_results = json.load(fp)
    plot_results(all_results)