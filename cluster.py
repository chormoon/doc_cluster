# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:07:20 2020

@author: lixy
"""
import data_process
import numpy as np
import pandas as pd
import sys
from sklearn.metrics.pairwise import cosine_similarity
import collections
from data_process import get_train_data,Doc2Vec,assess_model
import read_config 
import GLogger
import os
import csv
from singlepass import SingelPassCluster
from getkeyword import getKeywords_tfidf
from sklearn.cluster import KMeans
import joblib
import torch

conf = read_config.loadConf()
log=GLogger.Log(conf.log_file_path)
log.info('start read the config info!')

'''加载数据，返回Data: [xxx.txt ,...] 和 index2corpus: OrderedDict([(0，xxx.txt),(1,xxx.txt),....])'''
def loadData(file_path):
    Data = []
    index2corpus = collections.OrderedDict()
    index = 0
    with open(file_path,"r",encoding = 'utf8') as f:
        for line in f:
            Data.append(line.replace("\n",""))
            index2corpus[index] = line
            index += 1
        text2index = list(index2corpus.keys())
    # print('docs total size:{}'.format(len(text2index)))
    return Data,index2corpus
''' Data: [xxx.txt,...]或者[key key key, key key key,...]  file_len:文件数'''
def getData(file_path):
    Data = []
    file_len = 0
    with open(file_path,"r",encoding = 'utf8') as f:
        for line in f:
            Data.append(str(line.strip()))
            file_len += 1
    # print(f'file line:{file_len}')
    return Data,file_len


#训练&评估&生成文档向量
def get_train_vector(train_type,cluster_id,train_path,k_num):
    """
    训练&评估&生成文档向量
    :param content: 文本向量模型的路径
    :return: 文本向量矩阵
    """
    # conf = read_config.loadConf()
    # log=GLogger.Log(conf.log_file_path)

    "加载模型路径"
    model_name = str(cluster_id)+'_'+"d2c.model"
    model_path = os.path.join(conf.d2v_model_path, model_name)
    "加载处理后的文件和目标文件路径"
    processed_file_name = str(cluster_id)+'_'+"processed.txt"
    processed_file_path = os.path.join(conf.processed_file_path, processed_file_name)
    tagged_file_name = str(cluster_id)+'_'+"tagged.txt"
    tagged_file_path = os.path.join(conf.tagged_file_path, tagged_file_name)
    log.info('processed file:'+processed_file_path)
    log.info('tagged file:'+tagged_file_path)

    if(train_type == 1): #从头训练
        #生成训练数据
        log.info('start get the document vectors!')
        documents = get_train_data(train_path,processed_file_path,tagged_file_path)
        model = None
        #训练模型
        log.info('start traing model!')
        model= Doc2Vec(vector_size=100,epochs=20,alpha=0.06, min_alpha=0.01,min_count = 1,sample=1e-3,negative=5,workers=4,dm=0,window=10)
        model.build_vocab(documents)

        for epoch in range(2):
        	model.train(documents, total_examples=model.corpus_count, epochs=10)
        	model.alpha -= 0.002
        	model.min_alpha = model.alpha
        	model.train(documents, total_examples=model.corpus_count, epochs=10)

        model.save(model_path)
        log.info('save the docvec model in :'+model_path)
        #对模型进行评估
        assess_model(model_path, processed_file_path)
    else:
        model = Doc2Vec.load(model_path)
        documents = get_train_data(train_path, processed_file_path, tagged_file_path)
        titleList_,file_len = getData(tagged_file_path)
        tte = model.corpus_count + file_len
        model.train(documents, total_examples=tte, epochs=80)
        log.info('take an increment train')
        model.save(model_path)

    if os.path.exists(processed_file_path):
        log.info("processed_file_path exists:"+processed_file_path)
    else:
        log.info("processed_file_path not exists:"+processed_file_path)
        return
    if os.path.exists(tagged_file_path):
        log.info("tagged_file_path exists:"+tagged_file_path)
    else:
        log.info("tagged_file_path not exists:"+tagged_file_path)
        return
    if os.path.exists(model_path):
        log.info("mode_path exists:"+model_path)
    else:
        log.info("model path not exists:"+model_path)
        return
    #获取doc vector
    #得到所有的文本向量
    doc2vec = data_process.get_vector(model_path)

   
    #加载想要区分的文件的语料库以及对应指针  或者说每个文件的文件名和对应指针
    corpus, index2corpus= loadData(tagged_file_path)

    #single_cluster = SingelPassCluster()
    '''用单聚类策略分出队应的类和文本
    clusters,cluster_text = single_cluster.doc2vec_single_pass(doc2vec,corpus,similarity)
    '''
    '''Kmeans '''

    n_clusters = k_num
    # K-means方法   无监督学习
    # 建立模型。n_clusters参数用来设置分类个数，即K值，这里表示将样本分为k_num类。
    cluster = KMeans(n_clusters=n_clusters, random_state=0,algorithm='auto',max_iter = 200).fit(doc2vec)
    y_pred = cluster.labels_
    print(y_pred)
    quantity = pd.Series(y_pred).value_counts()
    print(f"cluster聚类数量：\n{quantity}")
    for i in range(n_clusters):
        print(f'第{i}类：')
        '''
            这里是创建分类文件夹
        '''
        for j in range(len(corpus)):
            if y_pred[j] == i:
                '''
                    这里是移动文件
                '''
                print ("     "+corpus[j])


    '''print("............................................................................................")
    print("得到的类数量有: {} 个 ".format(len(clusters)))
    print("............................................................................................\n")'''

    #保存cluster result
    '''result_file = str(cluster_id)+'_'+"cluster.csv"
    result_file = os.path.join(conf.result_dir, result_file)
    keyList = cluster_text.keys()
    
    valueList = []
    for k, v in cluster_text.items():
        file_list = ''
        print('第', k + 1, '类：')
        for i in v:
            print('     ',i)
            file_list += i + ";"
        valueList.append(file_list)
        
    rows = zip(keyList, valueList)

    with open(result_file, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    '''
    #生成doc 的 关键词
    key_file = str(cluster_id)+'_'+"keys.csv"
    key_file = os.path.join(conf.result_dir, key_file)
    titleList,file_num = getData(tagged_file_path)
    corpus, corpus_num = getData(processed_file_path)
    topK = 50
    result = getKeywords_tfidf(corpus,titleList,topK)
    result.to_csv(key_file,index=False)
    
        

if __name__ == "__main__":
    # 1为重头训练模式  0为增量训练模式
    # 分类任务的ID
    # 训练数据的地址
    # 最终想要的类的数量
    get_train_vector(0,101,"E:/Work/BWD/DAY1/doc_cluster/data",8)