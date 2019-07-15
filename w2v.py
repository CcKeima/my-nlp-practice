# -*- coding: utf-8 -*-
import math
import xlrd
import jieba
from gensim.test.utils import common_texts, get_tmpfile
from gensim import corpora, models, similarities

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
qlist = [] 
alist = []
for i in range(1, sheet.nrows):
    qlist.append(jieba.lcut(sheet.cell(i,1).value))
    alist.append(sheet.cell(i,2).value)

#训练Word2Vector模型
w2v_model = models.Word2Vec(qlist, size = 100, window = 5, min_count = 1, workers = 4)
word_vector = w2v_model.wv

#SIF加权的Word Vector平均得出Sentence Vector
sentence_vector = [avg(word_vector[x]) for x in qlist]

#查找最相似问题
question = input("please input the question: ")
query = jieba.lcut(question)
query = avg(word_vector[query])
idx = index(sentence_vector, query)
print('anwser:\n   ', alist[idx].split("答：")[1])