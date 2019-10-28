import math
import pandas as pd
import random
import numpy as np
import re
import log_reg
import nltk
import gensim

def avg(vec):
    sum = []
    for i in range(len(vec[0])):
        sum.append(0)
        for j in vec:
            sum[i] += j[i]
        sum[i] /= len(vec)
    return sum

def change(vec, size):
    res = np.zeros(size)
    for x, y in vec:
        res[x - 1] = y
    return res

#读入数据
reader = pd.read_csv('tra.tsv', sep = '\t', header = 0, index_col = 'PhraseId')
phrase = reader['Phrase'].values
label = reader['Sentiment'].values

corpus = [nltk.word_tokenize(re.sub(r'[^\w\s]', '', i)) for i in phrase]
dictionary = gensim.corpora.Dictionary(corpus)
corpus = [change(dictionary.doc2bow(i), dictionary.num_pos) for i in corpus]

#w2v_model = gensim.models.Word2Vec(corpus, size = 5, window = 5, min_count = 1, workers = 4)

data = []
for x, y in zip(corpus, label):
    a = np.zeros((5))
    a[y - 1] = 1
    data.append((x, a))

classifier = log_reg.log_reg(dictionary.num_pos, 5)
classifier.train(data, 100, 50, 0.1)

while 1:
    test = input()
    test = dictionary.doc2bow(nltk.word_tokenize(re.sub(r'[^\w\s]', '', test)))
    test = change(test, dictionary.num_pos)
    np.append(test, 1)
    classifier.predict([test])