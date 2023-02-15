# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:33:57 2020

@author: lixy
"""
import config
import sys,codecs
import pandas as pd
import numpy as np

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# tf-idf获取文本topK关键词
def getKeywords_tfidf(corpus,titleList,topK):
    file_num = len(titleList)
    # 1、构建词频矩阵，将文本中的词语转换成词频矩阵
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus) # 词频矩阵,a[i][j]:表示j词在第i个文本中的词频
    # 2、统计每个词的tf-idf权值
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    # 3、获取词袋模型中的关键词
    word = vectorizer.get_feature_names_out()
    # 4、获取tf-idf矩阵，a[i][j]表示j词在i篇文本中的tf-idf权重
    weight = tfidf.toarray()
    # 5、打印词语权重
    ids, titles, keys = [], [], []
    for i in range(file_num):
        #print(u"-------这里输出第",i+1 ,u"篇文本的词语tf-idf------")
        ids.append(i)
        titles.append(titleList[i])
        df_word,df_weight = [],[] # 当前文章的所有词汇列表、词汇对应权重列表
        for j in range(len(word)):
            # print(word[j],weight[i][j])
            df_word.append(word[j])
            df_weight.append(weight[i][j])
        df_word = pd.DataFrame(df_word,columns=['word'])
        df_weight = pd.DataFrame(df_weight,columns=['weight'])
        word_weight = pd.concat([df_word, df_weight], axis=1) # 拼接词汇列表和权重列表
        word_weight = word_weight.sort_values(by="weight",ascending = False) # 按照权重值降序排列
        keyword = np.array(word_weight['word']) # 选择词汇列并转成数组格式
        topK = min(len(keyword), topK)
        word_split = [keyword[x] for x in range(0,topK)] # 抽取前topK个词汇作为关键词
        word_split = " ".join(word_split)
        #print(word_split)
        keys.append(word_split)
    result = pd.DataFrame({"id": ids, "title": titles, "key": keys},columns=['id','title','key'])
    return result

# def loadData(file_path):
#     Data = []
#     file_len = 0
#     with open(file_path,"r",encoding = 'utf8') as f:
#         for line in f:
#             Data.append(str(line.strip()))
#             file_len += 1
#     return Data,file_len

# if __name__ == "__main__":
#     titleList,file_num = loadData(config.tagged_file_path)
#     corpus,corpus_num = loadData(config.processed_file_path)
#     topK = 500
#     result = getKeywords_tfidf(corpus,titleList,topK)
#     result.to_csv("result/keys_TFIDF.csv",index=False)

    
