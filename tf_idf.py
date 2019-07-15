# -*- coding: utf-8 -*-
import xlrd
import jieba
from gensim import corpora, models, similarities

#输入数据
data = xlrd.open_workbook('data.xls')
sheet = data.sheet_by_index(0)
qlist = [] 
alist = []
for i in range(1, sheet.nrows):
    qlist.append(jieba.lcut(sheet.cell(i,1).value))
    alist.append(sheet.cell(i,2).value)

#建立字典，语料库，TF-IDF模型
dictionary = corpora.Dictionary(qlist)
corpus = [dictionary.doc2bow(q) for q in qlist]
tf_idf_model = models.TfidfModel(corpus)

#建立相似度矩阵及查找最相似问题
index = similarities.MatrixSimilarity(tf_idf_model[corpus])
question = input("please input the question: ")
query = jieba.lcut(question)
query = tf_idf_model[dictionary.doc2bow(query)]
sims = index[query]
idx = max(list((x[1],x[0]) for x in enumerate(sims)))[1]
print('anwser:\n   ', alist[idx].split("答：")[1])