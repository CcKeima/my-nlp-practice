# -*- coding: utf-8 -*-
import math
import xlrd
import jieba
import gensim
import bp_torch
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import numpy as np

#求Word Vector平均值
def avg(vec):
    sum = []
    for i in range(len(vec[0])):
        sum.append(0)
        for j in vec:
            sum[i] += j[i]
        sum[i] /= len(vec)
    return sum

#求句子余弦相似度
def similarity(a, b):
    res = 0
    suma = 0
    sumb = 0
    for i in range(0, len(a)):
        res += a[i] * b[i]
        suma += a[i] * a[i]
        sumb += b[i] * b[i]
    suma = math.sqrt(suma)
    sumb = math.sqrt(sumb)
    res = res / (suma * sumb)
    return res

#求最相似的问题
def index(sentence, q):
    mx = 0
    idx = 0
    for i in range(0, len(sentence)):
        if similarity(sentence[i],q) > mx:
            mx = similarity(sentence[i],q)
            idx = i
    return idx

#输入数据
data = xlrd.open_workbook('data.xls')
sheet = data.sheet_by_index(0)
clist = []
qlist = [] 
alist = []
n = sheet.nrows
for i in range(1, n):
    clist.append(sheet.cell(i,0).value)
    qlist.append(jieba.lcut(sheet.cell(i,1).value))
    alist.append(sheet.cell(i,2).value)
# cdict = {}
# id = 0
# for i in clist:
#     if i not in cdict:
#         cdict[i] = id
#         id += 1
# n_class = len(cdict)

#建立TF-IDF模型
dictionary = gensim.corpora.Dictionary(qlist)
corpus = [dictionary.doc2bow(q) for q in qlist]
tf_idf_model = gensim.models.TfidfModel(corpus)

#建立Word2Vector模型
w2v_model = gensim.models.Word2Vec(qlist, size = 30, window = 5, min_count = 1, workers = 4)

#SIF加权的Word Vector平均得出Sentence Vector
sentence_vector = [avg(w2v_model.wv[x]) for x in qlist]

X = np.array(sentence_vector)
pca=PCA(n_components=1)
x = pca.fit_transform(X)
sif_sentence_vector = np.dot(np.dot(x, x.T), sentence_vector)

# #网络参数
# batch_size = 5
# learning_rate = 0.01
# n_epoches = 10

# #构造数据集
# train_set = []
# train_data = []
# for i in range(0, n - 1):
#     train_set.append((torch.tensor(sentence_vector[i]), cdict[clist[i]]))
#     train_data.append((torch.tensor(sentence_vector[i]), torch.zeros(n_class)))
#     train_data[i][1][cdict[clist[i]]] = 1  

# #训练神经网络
# bp = bp_torch.bpNet(30, 50, 50, n_class)
# optimizer = torch.optim.SGD(bp.parameters(), lr = learning_rate)
# #loss_function = torch.nn.L1Loss()
# loss_function = torch.nn.BCELoss()

# for i in range(n_epoches):
#     train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
#     for data in train_loader:
#         optimizer.zero_grad()
#         vec, label = data
#         out = bp(vec)
#         loss = loss_function(out, label)
#         loss.backward()
#         optimizer.step()
#     print("epoch ", i, " complete")

# #测试准确度
# test_result = [(torch.argmax(bp(x)), y) for x, y in train_set]
# [print(x, y) for x, y in test_result]
# n_right = sum([int(x == y) for x, y in test_result])
# print(n_right)

#查找最相似问题
question = input("please input the question: ")
query = jieba.lcut(question)
w2v_model = gensim.models.Word2Vec(size = 30, window = 5, min_count = 1, workers = 4)
w2v_model.build_vocab(qlist + [query])
w2v_model.train(qlist + [query], total_examples = w2v_model.corpus_count, epochs = w2v_model.iter)
sentence_vector = [avg(w2v_model.wv[x]) for x in qlist]
query = avg(w2v_model.wv[query])
idx = index(sentence_vector, query)
print('anwser:\n   ', alist[idx].split("答：")[1])