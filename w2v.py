# -*- coding: utf-8 -*-
import math
import xlrd
import jieba
import gensim
import bp

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
cdict = {}
id = 0
for i in clist:
    if i not in cdict:
        cdict[i] = id
        id += 1
n_class = len(cdict)

#训练Word2Vector模型
w2v_model = gensim.models.Word2Vec(qlist, size = 100, window = 5, min_count = 1, workers = 4)
word_vector = w2v_model.wv

#SIF加权的Word Vector平均得出Sentence Vector
sentence_vector = [avg(word_vector[x]) for x in qlist]

#构造数据集
train_data = []
for i in range(0, n - 1):
    train_data.append((sentence_vector[i], cdict[clist[i]]))

#训练bp神经网络分类器
classifier = bp.bp([100, 200, 200, n_class])
classifier.train_SGD(train_data, 10, 100, 1.0)

#问题分类
question = input("please input the question: ")
query = jieba.lcut(question)
query = avg(word_vector[query])
class_idx = classifier.calculate(query)
print(class_idx)

#查找最相似问题

idx = index(sentence_vector, query)
print('anwser:\n   ', alist[idx].split("答：")[1])