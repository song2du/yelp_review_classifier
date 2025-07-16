import json
import os
from argparse import Namespace
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from preprocessing import load_and_preprocess_data
from data_loader import ReviewVectorizer, SequenceVectorizer, ReviewDataset
from model_MLP import MlpClassifier
from model_CNN import CnnClassifier
from trainer import Trainer
from utils import set_seed, handle_dirs

def main():
    # 1. 설정 및 초기화
    args = Namespace(
        # 경로 및 파일 정보
        raw_dataset_csv='data/yelp/raw_train.csv',
        save_dir='model_storage/yelp/',
        
        # Vectorizer 및 Vocabulary 하이퍼파라미터   
        frequency_cutoff=25,
        
        # 모델 하이퍼파라미터
        hidden_dim=30,
        output_dim=1,
        embedding_dim=100,
        num_channels=256,
        dropout_p=0.1,
        
        # 훈련 하이퍼파라미터
        batch_size=128,
        early_stopping_criteria=5,
        learning_rate=0.001,
        num_epochs=100,
        seed=1337,
        
        # 데이터 분할 비율
        proportion_subset_of_train=0.5,
        train_proportion=0.7,
        val_proportion=0.15,
        test_proportion=0.15,
        
        # 실행 옵션
        cuda=True,
        expand_filepaths_to_save_dir=True
    )

    # 2. 데이터 전처리
    # raw_train.csv를 읽어 train/val/test로 분할된 final_df 생성
    final_df = load_and_preprocess_data(raw_csv=args.raw_dataset_csv, args=args)
    
    # Vectorizer 생성을 위해 train 스플릿만 따로 추출
    train_df = final_df[final_df.split=='train']

    all_results = {}

    # 3. 실험 루프 (MLP, CNN 순차 실행)
    for model_type in ['mlp', 'cnn']:
        print(f"\n===== Training {model_type.upper()} Model =====")

        # 3-1. 루프별 설정 초기화
        args.model_state_file = os.path.join(args.save_dir, f'{model_type}_model.pth')
        set_seed(args.seed, args.cuda)
        handle_dirs(args.save_dir)
        args.device = torch.device("cuda" if args.cuda else "cpu")
        print(f"Using device: {args.device}")

        # 3-2. 모델 타입에 맞는 Vectorizer 및 Dataset 생성
        if model_type == 'mlp':
            vectorizer = ReviewVectorizer.from_dataframe(train_df, cutoff=args.frequency_cutoff)
        else: # cnn
            vectorizer = SequenceVectorizer.from_dataframe(train_df, cutoff=args.frequency_cutoff)
        
        dataset = ReviewDataset(review_df=final_df, vectorizer=vectorizer)
        print("Data-pipeline is ready!")

        # 3-3. 모델 및 훈련 컴포넌트 생성
        if model_type == 'mlp':
            model = MlpClassifier(input_dim=len(vectorizer.review_vocab),
                                  hidden_dim=args.hidden_dim,
                                  output_dim=args.output_dim)
        else: # cnn
            model = CnnClassifier(embedding_dim=args.embedding_dim,
                                  vocab_size=len(vectorizer.review_vocab),
                                  num_channels=args.num_channels,
                                  num_classes=args.output_dim,
                                  dropout_p=args.dropout_p)
        
        model = model.to(args.device)
        loss_func = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=1)
        
        # 3-4. Trainer 생성 및 훈련/평가 실행
        trainer = Trainer(model=model, dataset=dataset, loss_func=loss_func, 
                          optimizer=optimizer, scheduler=scheduler, device=args.device, args=args)
        
        final_state = trainer.train_evaluate()

        # 3-5. 최종 테스트 및 결과 저장
        print("Loading the best model and running on the test set...")
        trainer.model.load_state_dict(torch.load(args.model_state_file))
        trainer.test_epoch()

        all_results[model_type] = final_state

    # 4. 최종 결과 비교 및 파일 저장
    print("\n\n===== FINAL COMPARISON =====")
    results_df = pd.DataFrame.from_dict(all_results).T
    print(results_df[['test_loss', 'test_acc', 'test_f1']])

    with open('results.json', 'w') as fp:
        json.dump(all_results, fp, indent=4)
    print("\nFull experiment results saved to results.json")

# 스크립트 실행 시작점
if __name__ == '__main__':
    main()