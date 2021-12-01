#! -*- coding:utf-8 -*-
import os, time, json, sys
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0

import numpy as np
from utils.bert4keras.backend import keras, K
from utils.bert4keras.models import build_transformer_model
from utils.bert4keras.tokenizers import Tokenizer
from utils.bert4keras.layers import ConditionalRandomField
from keras.models import Model
from keras.layers import Dense, Lambda

# 基本信息
crf_lr_multiplier = 100  # 必要时扩大CRF层的学习率

# bert配置
config_path = 'F:/git_repositories/ner-Demo/utils/pretrained_model/albert_tiny_google_zh_489k/albert_config.json'
checkpoint_path = 'F:/git_repositories/ner-Demo/utils/pretrained_model/albert_tiny_google_zh_489k/albert_model.ckpt'
dict_path = 'F:/git_repositories/ner-Demo/utils/pretrained_model/albert_tiny_google_zh_489k/vocab.txt'

#标签
id2label = {0: 'O', 1: 'B-PER', 2: 'M-PER', 3: 'B-LOC', 4: 'M-LOC', 5: 'B-ORG', 6: 'M-ORG'}
label2id = {'PER': 0, 'LOC': 1, 'ORG': 2}
num_labels = len(label2id) * 2 + 1

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


# 加载预训练模型
model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='albert'
)

def CRFGraph(model_output, num_labels, crf_lr_multiplier):
    # CRF(crf of bert4keras, 条件概率随机场)
    # Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data(https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)

    output = Dense(num_labels)(model_output)
    CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
    output = CRF(output)
    return output, CRF

model_output = Lambda(lambda x: x)(model.output)
output, CRF = CRFGraph(model_output, num_labels, crf_lr_multiplier)
model = Model(model.input, output)
# model.summary()

def viterbi_decode(nodes, trans):
    """Viterbi算法求最优路径
    其中nodes.shape=[seq_len, num_labels],
        trans.shape=[num_labels, num_labels].
    """
    labels = np.arange(num_labels).reshape((1, -1))
    scores = nodes[0].reshape((-1, 1))
    scores[1:] -= np.inf  # 第一个标签必然是0
    paths = labels
    for l in range(1, len(nodes)):
        M = scores + trans + nodes[l].reshape((1, -1))
        idxs = M.argmax(0)
        scores = M.max(0).reshape((-1, 1))
        paths = np.concatenate([paths[:, idxs], labels], 0)
    return paths[:, scores[:, 0].argmax()]


def extract_arguments(text):
    """arguments抽取函数
    """
    tokens = tokenizer.tokenize(text)
    while len(tokens) > 126:
        tokens.pop(-2)
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    nodes = model.predict([[token_ids], [segment_ids]])[0]
    trans = K.eval(CRF.trans)
    labels = viterbi_decode(nodes, trans)
    labels_id = []
    flag = 0
    labels = labels[1:len(labels)-1]#去掉cls和sep
    for i, label in enumerate(labels):
        if label != 0:
            if labels[i+1] == 0:
                if flag == 0:
                    labels_id.append(id2label[label])
                else:
                    labels_id.append(id2label[label].replace('M-', 'E-'))
            else:
                labels_id.append(id2label[label])
                flag += 1
        else:
            labels_id.append(id2label[label])

    return [labels_id]


if __name__ == '__main__':
    start = time.time()
    model.load_weights('./save/save_albert/best_model.weights')#

    #预测
    text = "我国驻美大使李道豫５年来多次欣赏黄河艺术团的演出"
    R = extract_arguments(text)
    print(R)

    delta_time = time.time() - start
    print("时长：", delta_time)