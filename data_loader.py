
from collections import Counter
from vocabulary import Vocabulary
import numpy as np
import pandas as pd
import string
import json
from torch.utils.data import Dataset, DataLoader

# 데이터 로딩, 전처리, 파이토치 공급을 하나의 객체로 묶은 데이터 파이프라인
class ReviewDataset(Dataset):
    def __init__(self, review_df, vectorizer):
        # 데이터를 train, test, val로 나눈 후 딕셔너리로 관리한다.
        # 관리하는 법 : set_split을 통해 사용할 데이터셋을 정한다.
        self.review_df = review_df 
        self._vectorizer = vectorizer

        self.train_df = self.review_df[self.review_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.review_df[self.review_df.split=='val']
        self.val_size = len(self.val_df)

        self.test_df = self.review_df[self.review_df.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            'train':  (self.train_df, self.train_size),
            'val' : (self.val_df, self.val_size),
            'test' : (self.test_df, self.test_size)    
        }

        self.set_split('train')
    
    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_csv):
        # 데이터셋을 로드하고 새로운 ReviewVectorizer 객체 생성
        review_df = pd.read_csv(review_csv)
        train_review_df = review_df[review_df.split=='train']
        return cls(review_df, ReviewVectorizer.from_dataframe(train_review_df)) # 전체 데이터와 훈련데이터로만 만든 어휘 사전 저장

    @classmethod
    def load_dataset_and_load_vectorizer(cls, review_csv, vectorizer_filepath):
        # 데이터셋을 로드하고 새로운 ReviewVectorizer 객체 생성
        # 캐쉬된 객체를 재사용할 때 사용
        review_df = pd.read_csv(review_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(review_df, vectorizer)
    
    @staticmethod # -> 클래스에 의존하지 않는 독립적인 헬퍼함수 만들 때 사용
    def load_vectorizer_only(vectorizer_filepath):
        # 직렬화 된 파일에서 ReviewVectorizer 객체를 로드
        with open(vectorizer_filepath) as fp: # 파일 경로 열기
            return ReviewVectorizer.from_serializable(json.load(fp)) # 파일 경로에 있는 json 파일 딕셔너리로 변경 후 ReviewVectorizer 객체로 만들어 반환

    def save_vectorizer(self, vectorizer_filepath):
        # 객체를 직렬화된 파일 (json) 형태로 저장
        with open(vectorizer_filepath, "w") as fp: # 파일 경로를 쓰기 모드로 열기
            json.dump(self._vectorizer.to_serializable(), fp) # 직렬 형태로 변환한 객체를 json으로 저장

    def get_vectorizer(self):
        # 벡터 변환 객체 반환
        return self._vectorizer
    
    def set_split(self, split="train"):
        # 어떤 데이터셋을 선택할지 결정
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        # 타겟 사이즈 반환
        return self._target_size

    def __getitem__(self, index):
        # 리뷰 텍스트 -> 숫자 벡터, 평점 -> 숫자 레이블로 변환
        row = self._target_df.iloc[index] # 선택한 데이터프레임에서 해당 인덱스 행 선택

        review_vector = self._vectorizer.vectorize(row.review) # 해당 행에서 리뷰를 벡터화

        rating_index = self._vectorizer.rating_vocab.lookup_token(row.rating) # 해당 레이팅에 대한 인덱스 저장

        return {'x_data': review_vector,
                'y_target': rating_index} # 데이터와 타겟으로 반환
        

    def get_num_batches(self, batch_size):
        # 배치 크기가 주어지면 데이터셋으로 만들 수 있는 배치 개수 반환
         return len(self) // batch_size

class ReviewVectorizer(object):
    def __init__(self, review_vocab, rating_vocab):
        # 정수로 매핑된 리뷰와 평점 어휘 사전 저장 
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab

    def vectorize(self, review):
        # 리뷰에 대한 BoW 생성
        bow = np.zeros(len(self.review_vocab), dtype=np.float32) # review_vocab의 길이 만큼 0.00 배열 생성

        for token in review.split(" "): # 리뷰를 공백으로 나눈 후 반복
            if token not in string.punctuation: # 토큰이 구두점이 아니라면
                bow[self.review_vocab.lookup_token(token)] = 1 # 토큰에 해당하는 인덱스에 1 삽입
        
        return bow


    @classmethod
    def from_dataframe(cls, review_df, cutoff=25): # cutoff는 노이즈 감소와 모델 경량화를 위해 사용
        # 전체 리뷰에서 단어 사용 빈도 계산 후 cutoff보다 빈도 높은 단어만 새로운 어휘 사전에 포함
        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)

        for rating in sorted(set(review_df.rating)): # 레이팅별로 오름차순 정렬한 review_df에서 레이팅 반복
            rating_vocab.add_token(rating) # rating_vocab에 추가
        
        word_counts = Counter() # 카운터 생성, 카운터는 각 항목이 몇번 등장했는지 빈도 카운트
        for review in review_df.review:  # 리뷰데이터 반복
            for word in review.split(" "): # 각 리뷰데이터 공백 기준으로 나눈 후 반복
                if word not in string.punctuation: # 만약 구두점이 아니라면 
                    word_counts[word] += 1 # 해당 토큰 위치에 1 추가
        
        for word, count in word_counts.items(): # 토큰을 카운트한 딕셔너리에서 아이템 반복
            if count > cutoff: # 카운트가 cutoff보다 크다면
                review_vocab.add_token(word) # review_vocab에 word 추가
        
        return cls(review_vocab, rating_vocab)

    @classmethod
    def from_serializable(cls, contents):
        # 직렬화 된 딕셔너리에서 객체 생성 -> 미리 만든 어휘 사전을 사용하여 시간 절약
        review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
        rating_vocab = Vocabulary.from_serializable(contents['rating_vocab'])

        return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)

    def to_serializable(self):
        # 직렬화 된 딕셔너리 생성 -> 객체를 파일로 저장하기 위해 딕셔너리로 변환
        return {'review_vocab': self.review_vocab.to_serializable(),
                'rating_vocab': self.rating_vocab.to_serializable()}

class SequenceVectorizer(object):
    def __init__(self, review_vocab, rating_vocab, max_length):
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab
        self.max_length = max_length
        
    
    def vectorize(self, review):
        indices = [self.review_vocab.lookup_token(word) for word in review.split(' ')]

        if len(indices) > self.max_length: # 최대 길이보다 길면 잘라내기
            indices = indices[:self.max_length]
        else:
            pad_width = self.max_length - len(indices) # 패딩 크기 결정
            padding = [self.review_vocab.pad_index] * pad_width # 패드 인덱스에 패딩 크기만큼 추가
            indices.extend(padding) # 최종 패딩을 문장 끝에 삽입
        return np.array(indices, dtype=np.int64)

    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
        review_vocab = Vocabulary(add_unk=True, add_pad=True)
        rating_vocab = Vocabulary(add_unk=False)
        max_len = 0

        for rating in sorted(set(review_df.rating)): # 레이팅별로 오름차순 정렬한 review_df에서 레이팅 반복
            rating_vocab.add_token(rating) # rating_vocab에 추가
        
        word_counts = Counter() # 카운터 생성, 카운터는 각 항목이 몇번 등장했는지 빈도 카운트
        for review in review_df.review:  # 리뷰데이터 반복
            words = review.split(" ") # 각 리뷰데이터 공백 기준으로 나눔
            if len(words) > max_len: # 최대 길이 계산
                max_len = len(words) 
            for word in words: # 단어의 빈도 수 1 증가
                word_counts[word] += 1
        
        for word, count in word_counts.items(): # 토큰을 카운트한 딕셔너리에서 아이템 반복
            if count > cutoff: # 카운트가 cutoff보다 크다면
                review_vocab.add_token(word) # review_vocab에 word 추가
        
        return cls(review_vocab, rating_vocab, max_len)

    

def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device='cpu'):
    # 각 텐서를 지정된 장치로 이동
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last) # 데이터셋에서 미니배치 만큼 데이터를 꺼냄
    
    for data_dict in dataloader: # 데이터 배치를 하나씩 꺼냄
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device) # 배치를 구성하는 각 텐서를 지정된 장치로 이동
        yield out_data_dict # 장치로 이동이 완료된 새로운 데이터배치 반환