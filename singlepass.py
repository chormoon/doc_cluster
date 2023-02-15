# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:21:35 2020

@author: lixy
"""
import data_process
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models, similarities, matutils
import collections
from data_process import get_train_data,Doc2Vec,assess_model
import read_config 
import GLogger
import os
import csv

conf = read_config.loadConf()
log=GLogger.Log(conf.log_file_path)

class SingelPassCluster(object):
     def get_max_similarity(self, cluster_cores, vector):
         max_value = 0
         max_index = -1
         print('vector:{}'.format(vector))
         for k, core in cluster_cores.items():
             print('core:{}'.format(core))
             similarity = matutils.cossim(vector, core)
             if similarity > max_value:
                 max_value = similarity
                 max_index = k
         return max_index, max_value

     def get_doc2vec_similarity(self, cluster_cores, vector):
         max_value = 0
         max_index = -1
         for k, core in cluster_cores.items():  # core -> np.ndarray
             similarity = cosine_similarity(vector.reshape(1, -1), core.reshape(1, -1))
             similarity = similarity[0, 0]
             if similarity > max_value:
                 max_value = similarity
                 max_index = k
         return max_index, max_value
 
     def doc2vec_single_pass(self, corpus_vec, corpus, theta):
         print("start****************************")
         clusters = {}
         cluster_cores = {}
         cluster_text = {}
         num_topic = 0
         cnt = 0
         for vector,text in zip(corpus_vec,corpus):
             if num_topic == 0:
                 clusters.setdefault(num_topic, []).append(vector)
                 cluster_cores[num_topic] = vector
                 cluster_text.setdefault(num_topic, []).append(text)
                 num_topic += 1
             else:
                 max_index, max_value = self.get_doc2vec_similarity(cluster_cores, vector)
                 if max_value > theta:
                     clusters[max_index].append(vector)
                     core = np.mean(clusters[max_index], axis=0)  # 更新簇中心
                     cluster_cores[max_index] = core
                     cluster_text[max_index].append(text)
                 else:  # 创建一个新簇
                     clusters.setdefault(num_topic, []).append(vector)
                     cluster_cores[num_topic] = vector
                     cluster_text.setdefault(num_topic, []).append(text)
                     num_topic += 1
             cnt += 1
             if cnt % 100 == 0:
                 print('processing {}...'.format(cnt))
         #'尝试打印core 发现是矩阵坐标没有语义信息'
         #print("clust core:", cluster_cores)
         #print("clust:",len(clusters))
         #print("clust:",cluster_text)
         return clusters, cluster_text

     def kmeans(doc2vec,corpus,k):
         # 随机取质心
         centroids = random.sample(doc2vec, k)
         # 更新质心 直到变化量全为0
         changed, newCentroids = classify(doc2vec, centroids, k)
         while np.any(changed != 0):
             changed, newCentroids = classify(doc2vec, newCentroids, k)
         centroids = sorted(newCentroids.tolist())  # tolist()将矩阵转换成列表 sorted()排序
         cluster = []
         for vector,text in zip(corpus_vec,corpus):
             clalist = calcDis(vector, centroids, k)  # 调用欧拉距离
             minDistIndices = np.argmin(clalist, axis=1)
             for i in range(k):
                 cluster.append([])
             for i, j in enumerate(minDistIndices):  # enymerate()可同时遍历索引和遍历元素
                 cluster[j].append(text)
         return centroids, cluster


    
