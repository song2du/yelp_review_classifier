from tqdm import tqdm
from utils import make_train_state, compute_accuracy, compute_f1
from data_loader import generate_batches
import torch

class Trainer:
    def __init__(self, model, dataset, loss_func, optimizer, scheduler, device, args):
        """
        매개변수:
            model (nn.Module): 훈련시킬 모델
            dataset (Dataset): 데이터셋 객체
            loss_func: 손실 함수
            optimizer: 옵티마이저
            scheduler: 학습률 스케줄러
            device (torch.device): 연산을 수행할 장치 (cpu or cuda)
            args (Namespace): 하이퍼파라미터 및 설정값을 담은 객체
        """
        self.model = model
        self.dataset = dataset
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.args = args
        self.train_state = make_train_state(self.args)
    
    def _update_and_save_checkpoint(self): # 객체 상태를 변경하는 함수
        
        current_val_loss = self.train_state['val_loss'][-1] # val_loss 리스트에서 가장 최신 값을 가져옴
        
        if current_val_loss < self.train_state['early_stopping_best_val']: # 새로운 최적값 발견시
            self.train_state['early_stopping_best_val'] = current_val_loss # 최적값 갱신
            torch.save(self.model.state_dict(), self.train_state['model_filename']) # 모델 저장
            self.train_state['early_stopping_step'] = 0 # 조기 종료 단계 초기화
        else: # 성능이 개선 되지 않았을 때
            self.train_state['early_stopping_step'] += 1 # 조기 종료 단계 + 1
        
        # 조기종료 여부 결정
        self.train_state['stop_early'] = (
            self.train_state['early_stopping_step'] >= self.args.early_stopping_criteria
        )
        


    def train_evaluate(self):
        # 각 에폭 당 훈련하고 평가
        epoch_bar = tqdm(desc="training routine",
                                  total=self.args.num_epochs, position=0) # 진행률 표시줄
        
        for epoch_index in range(self.args.num_epochs):
            self.train_state['epoch_index'] = epoch_index # 현제 에폭 인덱스 저장?
            self._train_epoch() # 훈련 데이터로 한 에폭 실행
            self._validate_epoch() # 검증 데이터로 한 에폭 실행
            self._update_and_save_checkpoint() # 조기 종료 여부 판단 및 모델 저장
            self.scheduler.step(self.train_state['val_loss'][-1]) # 스케줄러 업데이트 -> 학습률을 훈련 과정에 동적으로 조절

            if self.train_state['stop_early']: # 조기 종료 신호 확인 후 종료
                print("Early Stopping.")
                break
        
        return self.train_state
                



    def _train_epoch(self):
        # 1개 에폭의 훈련 로직
        self.model.train() # 모델을 훈련모드로 전환
        self.dataset.set_split('train') # 훈련 데이터셋 가져오기
        batch_generator = generate_batches(self.dataset, batch_size=self.args.batch_size,
                                           device=self.args.device) # 훈련 데이터셋으로 배치 생성
        
        running_loss = 0.0
        running_acc = 0.0
        running_f1 = 0.0

        train_bar = tqdm(batch_generator, 
                              total=self.dataset.get_num_batches(self.args.batch_size),
                              desc=f"[Epoch {self.train_state['epoch_index']} TRAIN]")

        for batch_index, batch_dict in enumerate(batch_generator): # 각 배치 당 학습 진행
            self.optimizer.zero_grad() # 기울기 0으로 초기화
            y_pred = self.model(x_in=batch_dict['x_data'].float()) # 출력 계산

            y_target = (batch_dict['y_target']-1).float() # 정답 텐서 가져오기
            y_target = y_target.unsqueeze(1) # 텐서 모양 (128, ) -> (128, 1)
            loss = self.loss_func(y_pred, y_target) # 손실 함수 계산
            loss_t = loss.item() # 텐서에서 숫자 값만 추출
            running_loss += (loss_t  - running_loss) / (batch_index + 1) # 손실의 추세를 보여주는 역할

            loss.backward() # 역전파
            self.optimizer.step() # 가중치 업데이트

            acc_t = compute_accuracy(y_pred, batch_dict['y_target']-1) # 정확도 계산
            running_acc += (acc_t - running_acc) / (batch_index + 1) # 정확도 추세 계산

            f1_t = compute_f1(y_pred, batch_dict['y_target']-1) # f1 계산
            running_f1 += (f1_t - running_f1) / (batch_index + 1) # f1 추세 계산

            train_bar.set_postfix(loss=running_loss, acc=running_acc, f1=running_f1) # 진행바 업데이트
        
        self.train_state['train_loss'].append(running_loss)
        self.train_state['train_acc'].append(running_acc)
        self.train_state['train_f1'].append(running_f1)




    def _validate_epoch(self):
        # 1개 에폭의 검증 로직
        self.model.eval() # 모델을 평가모드로 전환
        self.dataset.set_split('val') # 검증 데이터셋 가져오기
        batch_generator = generate_batches(self.dataset, batch_size=self.args.batch_size,
                                           device=self.args.device) # 검증 데이터셋으로 배치 생성
        
        running_loss = 0.0
        running_acc = 0.0
        running_f1 = 0.0

        val_bar = tqdm(batch_generator, 
                              total=self.dataset.get_num_batches(self.args.batch_size),
                              desc=f"[Epoch {self.train_state['epoch_index']} VAL]")

        for batch_index, batch_dict in enumerate(batch_generator): # 각 배치 당 검증 진행
            with torch.no_grad():
                y_pred = self.model(x_in=batch_dict['x_data'].float()) # 출력 계산
                y_target = (batch_dict['y_target']-1).float()
                y_target = y_target.unsqueeze(1)
                loss = self.loss_func(y_pred, y_target)
                loss_t = loss.item() # 텐서에서 숫자 값만 추출
                running_loss += (loss_t  - running_loss) / (batch_index + 1) # 손실의 추세를 보여주는 역할

                acc_t = compute_accuracy(y_pred, batch_dict['y_target']-1) # 정확도 계산
                running_acc += (acc_t - running_acc) / (batch_index + 1) # 정확도 추세 계산

                f1_t = compute_f1(y_pred, batch_dict['y_target']-1) # f1 계산
                running_f1 += (f1_t - running_f1) / (batch_index + 1) # f1 추세 계산

            val_bar.set_postfix(loss=running_loss, acc=running_acc, f1=running_f1) # 진행바 업데이트
        
        self.train_state['val_loss'].append(running_loss)
        self.train_state['val_acc'].append(running_acc)
        self.train_state['val_f1'].append(running_f1)
    
    def test_epoch(self):
        # 최종 모델 테스트
        self.model.eval() # 모델을 평가모드로 전환
        self.dataset.set_split('test') # 테스트 데이터셋 가져오기
        batch_generator = generate_batches(self.dataset, batch_size=self.args.batch_size,
                                           device=self.args.device) # 테스트 데이터셋으로 배치 생성
        
        running_loss = 0.0
        running_acc = 0.0
        running_f1 = 0.0

        test_bar = tqdm(batch_generator, 
                              total=self.dataset.get_num_batches(self.args.batch_size),
                              desc=f"[Epoch {self.train_state['epoch_index']} TEST]")

        for batch_index, batch_dict in enumerate(batch_generator): # 각 배치 당 검증 진행
            with torch.no_grad():
                y_pred = self.model(x_in=batch_dict['x_data'].float()) # 출력 계산
                y_target = (batch_dict['y_target']-1).float()
                y_target = y_target.unsqueeze(1)
                loss = self.loss_func(y_pred, y_target)
                loss_t = loss.item() # 텐서에서 숫자 값만 추출
                running_loss += (loss_t  - running_loss) / (batch_index + 1) # 손실의 추세를 보여주는 역할

                acc_t = compute_accuracy(y_pred, batch_dict['y_target']-1) # 정확도 계산
                running_acc += (acc_t - running_acc) / (batch_index + 1) # 정확도 추세 계산

                f1_t = compute_f1(y_pred, batch_dict['y_target']-1) # f1 계산
                running_f1 += (f1_t - running_f1) / (batch_index + 1) # f1 추세 계산

            test_bar.set_postfix(loss=running_loss, acc=running_acc, f1=running_f1) # 진행바 업데이트
        
        self.train_state['test_loss'] = running_loss
        self.train_state['test_acc'] = running_acc
        self.train_state['test_f1'] = running_f1




