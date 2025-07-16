from argparse import Namespace
from collections import Counter
import json
import os
import re
import string

class Vocabulary(object):
    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>",
                 add_pad=True, pad_token='<PAD>'):
        # 토큰을 정수 인코딩하고 어휘 사전에 없는 토큰 처리
        if token_to_idx is None: # 만약 token_to_idx가 None이라면 딕셔너리 생성
            token_to_idx = {}
        self._token_to_idx = token_to_idx  
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()} # _token_to_idx를 돌면서 인덱스를 키로, 토큰을 value로 하는 딕셔너리 생성

        self._add_pad = add_pad
        self._pad_token = pad_token
        self.pad_index = -1
        if self._add_pad:
            self.pad_index = self.add_token(self._pad_token) # 가장 먼저 추가되어 0번 인덱스 할당

       
        self._add_unk = add_unk # unk 토큰을 추가할지 지정하는 플래그
        self._unk_token = unk_token # vocabulary에 추가할 unk 토큰

        self.unk_index = -1 # unk 기능 비활성화
        if add_unk:
            self.unk_index = self.add_token(unk_token) # 만약 unk 토큰이 추가되면 unk_index에는 0 이상의 값이 덮어써진다.


    def to_serializable(self):
        # 직렬화할 수 있는 딕셔너리 반환
        return {'token_to_idx': self._token_to_idx,
                'add_unk': self._add_unk,
                'unk_token': self._unk_token}
    
    @classmethod
    def from_serializable(cls, contents):
        # 직렬화된 딕셔너리에서 Vocabulary 객체 생성
        return cls(**contents) # cls -> 클래스 호출 **-> 딕셔너리를 풀어서 함수의 인자로 전달

    def add_token(self, token):
        # 토큰을 기반으로 매핑 딕셔너리 업데이트
        if token in self._token_to_idx:
            index = self._token_to_idx[token] # 만약 매핑 딕셔너리에 토큰이 있다면 토큰의 인덱스 저장
        else:
            index = len(self._token_to_idx) # 새로운 인덱스를 어휘 사전의 현재 크기로 할당
            self._token_to_idx[token] = index # 토큰의 값으로 인덱스 저장
            self._idx_to_token[index] = token # 인덱스의 값으로 토큰 저장
        
        return index

    def add_many(self, tokens):
        # 토큰 리스트를 어휘 사전에 추가
        return [self.add_token(token) for token in tokens] # 각 토큰을 사전에 추가 후 그 토큰에 해당하는 인덱스의 리스트 반환

    def lookup_token(self, token):
        # 토큰에 대응하는 인덱스 출력
        if self.unk_index >= 0: # unk 활성화 되었다면
            return self._token_to_idx.get(token, self.unk_index) # 만약 해당 토큰이 있으면 해당 토큰의 인덱스 반환, 없으면 unk 인덱스 반환
        else:
            return self._token_to_idx[token] # 해당 토큰의 인덱스 반환
    
    def lookup_index(self, index):
        # 인덱스에 해당하는 토큰 출력
        if index not in self._idx_to_token:
            raise KeyError("Vocabulary에 인덱스(%d)가 없습니다." % index)
        return self._idx_to_token[index] # 해당 인덱스에 해당하는 토큰 반환

    def __str__(self):
        # print로 어휘 사전 출력할 때 양식 설정
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        # len 출력할 때 양식 설정
        return len(self._token_to_idx)
    