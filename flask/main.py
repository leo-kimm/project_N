import pandas as pd
from sklearn import preprocessing as pr
import joblib
from flask import Flask, request
from datetime import timedelta
import jsonify
import time

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

# kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

# transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


# Kobert 학습모델 만들기

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=2,  # softmax 사용 <- binary일 경우는 2
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


# kobert 입력 데이터로 만들기

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

# GPU 사용 시
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Bert모델, Voca 불러오기
bertmodel, vocab = get_pytorch_kobert_model()

# path 설정
PATH = 'C:/Users/kjhkm/Downloads/'

model = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장


# Setting parameters 파라미터 세팅
max_len = 64  # 해당 길이를 초과하는 단어에 대해선 bert가 학습하지 않음
batch_size = 64
warmup_ratio = 0.1
num_epochs = 7  # 훈련 반복횟수
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

# 토큰화
# 기본 Bert tokenizer 사용
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


def hola_function():

    print("대구 사는 현규가 만든 모델 호출합니다. 좀 느릴꺼에요")
    start = time.time()  # 시작 시간 저장

    # path 설정
    PATH = 'C:/Users/kjhkm/Downloads/'

    model = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장


    print("bert model = on")
    print("time :", time.time() - start)


def getSentimentValue(comment, tok, max_len, batch_size, device):


    commnetslist = []  # 텍스트 데이터를 담을 리스트
    emo_list = []  # 감성 값을 담을 리스트
    for c in comment:  # 모든 댓글
        commnetslist.append([c, 5])  # [댓글, 임의의 양의 정수값] 설정

    pdData = pd.DataFrame(commnetslist, columns=[['뉴스', '감성']])
    pdData = pdData.values
    test_set = BERTDataset(pdData, 0, 1, tok, max_len, True, False)
    test_input = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=5)

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_input):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        # 이때, out이 예측 결과 리스트

        out = model(token_ids, valid_length, segment_ids)
        print(out)
        # e는 2가지 실수 값으로 구성된 리스트
        # 0번 인덱스가 더 크면 부정, 긍정은 반대
        for e in out:
            if e[0] > e[1]:  # 부정
                value = 0
                emo_list.append("부정")
                print('부정')
            else:  # 긍정
                value = 1
                emo_list.append("긍정")
                print('긍정')

    return emo_list  # 텍스트 데이터에 1대1 매칭되는 감성값 리스트 반환


# 뉴스기사 테스트 함수
def news():
    comment = []
    comment.append(input("원하는 기사를 입력하세요"))

    for c in comment:
        print(f'\n기사 : {c}\n')

    return getSentimentValue(comment, tok, max_len, batch_size, device)


def sayHello(input_name,input_age):
    # 그동안 몇달동안 했었던 주가예측 모델 호출하면 되겠지..............
    if input_name == "정민화":
        msg = input_name + "님은"  + "노땅이시구만유" + "___" + input_age
    else:
        msg = index()

    return msg + "___"


app = Flask(__name__)
# app.config['PERMANENT_SESSION_LIFETIME'] =  timedelta(minutes=5)
@app.route('/test')
def index():
    input_name = request.args.get("name")
    input_age = request.args.get("age")
    output = sayHello(input_name, input_age)
    return output

@app.route('/')
def info():
    input_company = request.args.get("company")

    msg = ""

    if input_company == "sec":
        msg = hola1()
    elif input_company == "lg":
        msg = hola2()

        # [-1 - 1 - 1 - 1 - 1 - 1 - 1 - 1 - 1 - 1 - 1 - 1 - 1 - 1 - 1  1  1  1  1 - 1]
    # {"D","D+1","D+2","D+3"}
    return str(msg)

@app.route('/hola1')
def hola1():
    output = news()
    return output

@app.route('/hola2')
def hola2():
    output = "[hola2() 호출되었는데........] 방장...... 왜 아무말이 없소?"
    return output

@app.route('/hola3')
def hola3():
    input_date = request.args.get("date")
    input_mento = request.args.get("mento")
    msg = ""
    if input_mento == "권소희":
        msg = "노땅이 아닌디유???????????????"
    else:
        msg = "전부 노땅인디유*********************************"

    output = input_date + "   ====   " + msg
    return output


def hello(name, address):
    msg = address + "에 사는 " + name + "님은 못생겼구먼유..........................."
    return msg

@app.route('/hola4')
def hola4():
    input_name = request.args.get("name")
    input_address = request.args.get("address")
    msg = hello(input_name, input_address)

    return msg


@app.route('/hola5')
def hola5():
    input_a = request.args.get("a")
    input_b = request.args.get("b")

    msg = gugudan(input_a, input_b)

    return msg




def gugudan(a,b):

    value = int(a) * int(b)

    msg = str(a) + "x" + str(b) + " = " + str(value)

    return msg



if __name__ == '__main__':
    # app.run()
    app.run(host='0.0.0.0', port=8888)