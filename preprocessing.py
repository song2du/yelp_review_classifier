import collections
import numpy as np
import pandas as pd
import re

def _clean_text(text):
    # 텍스트 정규식으로 전처리
    text = text.lower() # 소문자로 변환
    text = re.sub(r"([.,!?])", r" \1 ", text) # 특수문자 혹은 구두점 발견 시 양옆에 공백 추가
    text = re.sub(r"[^a-zA-z.,!?]+", r" ", text) # 허용되지 않은 문자들을 공백으로 치환 

    return text

def _create_subset(reviews_df, args):
     # rating 순으로 정렬된 딕셔너리 생성
    by_rating = collections.defaultdict(list) # 딕셔너리 생성
    for _, row in reviews_df.iterrows():
        by_rating[row.rating].append(row.to_dict()) # 레이팅 별로 딕셔너리에 삽입 / 각 세트에 긍정/부정 리뷰가 동일한 비율로 들어가도록

    review_subset = []

    for _, item_list in sorted(by_rating.items()):  # by_rating.items()에는 부정과 긍정에 대한 리뷰 딕셔너리 리스트가 있음.
        n_total = len(item_list) # 리뷰의 전체 길이 구하기
        n_subset = int(args.proportion_subset_of_train * n_total) # 리뷰의 길이와 서브셋 비율을 곱해 서브셋 크기 결정
        review_subset.extend(item_list[:n_subset]) # n_subset 개수만큼의 리뷰를 잘라내 review_subset에 하나씩 추가

    review_subset = pd.DataFrame(review_subset) # 데이터 셋으로 변형
    
    return review_subset

def _add_split_labels(subset_df, args):
    by_rating = collections.defaultdict(list) # 딕셔너리 생성
    for _, row in subset_df.iterrows(): # 리뷰 딕셔너리의 각 행 반복
        by_rating[row.rating].append(row.to_dict()) # 딕셔너리에 레이팅 별로 삽입

    final_list = [] # 리스트 생성
    np.random.seed(args.seed) # 동일한 실험 결과를 위해 시드 설정

    for _, item_list in sorted(by_rating.items()): # by_rating.items() 안의 레이팅과 리뷰 반복
        np.random.shuffle(item_list) # 데이터 무작위로 섞음

        n_total = len(item_list)
        n_train = int(args.train_proportion * n_total)
        n_val = int(args.val_proportion * n_total)
        n_test = int(args.test_proportion * n_total)

        for item in item_list[:n_train]: # n_train까지 데이터 잘라서 'train' 레이블 추가
            item['split'] = 'train'
        
        for item in item_list[n_train:n_train+n_val]: # n_train부터 n_train+n_val까지 데이터 잘라서 'val' 레이블 추가
            item['split'] = 'val'
        
        for item in item_list[n_train+n_val:n_train+n_val+n_test]: # n_train+n_val 부터 n_train+n_val+n_test까지 잘라서 'test'레이블 추가
            item['split'] = 'test'
        
        final_list.extend(item_list)  # 최종 리스트에 하나씩 추가
    
    final_reviews = pd.DataFrame(final_list) # 데이터프레임으로 변환

    return final_reviews
    

def load_and_preprocess_data(raw_csv, args):
    # 원본 파일 로드, 전처리 및 분할 수행, 최종 데이터 프레임 반환
    data = pd.read_csv(raw_csv, header=None, names=['rating', 'review']) # 원본 데이터 읽어 오기
   
    # 서브셋 생성
    review_subset = _create_subset(reviews_df=data, args=args)

    # 데이터 분할
    final_df = _add_split_labels(review_subset, args=args)

    # 데이터 전처리
    final_df.review = final_df.review.apply(_clean_text) # final_df review 컬럼에 전처리 함수 적용
    final_df['rating'] = final_df.rating.apply({1: 'negative', 2: 'positive'}.get) # rating의 정수를 텍스트로 변환

    return final_df




