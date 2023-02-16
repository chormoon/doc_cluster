# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:45:52 2020

@author: lixy
"""
import os
import re
import collections
import codecs
import jieba.posseg as pseg
import numpy as np 
import gensim
from gensim.models import Doc2Vec
import read_config
import torch
import GLogger
conf = read_config.loadConf()
log=GLogger.Log(conf.log_file_path)

#判断是否中文
def ischinese(char):
    if '\u4e00'<= char <='\u9fff':
        return True
    return False

#判断是否数值
def isnum(char):
    if '\u0030-'<= char <='\u0039':
        return True
    return False

def get_stop_words(stop_word_path):
    """
    获取停用词列表
    ：return 停用词列表
    """
    stoplist = [i.strip() for i in codecs.open(stop_word_path, "r", encoding='UTF8').readlines()]
    return stoplist


def cut_sentence(sentence,stop_word_path):
    """
       fun:切词，过滤没有意义的词，停用词，数字,根据具体的文章做具体的处理
           保留名词，英文和自定义词库中的词，长度大于2的词
    """ 
    
    #停用词加载
    stopwords_list = get_stop_words(stop_word_path)
    #切词
    seg_list = pseg.lcut(sentence)
    seg_list = [i for i in seg_list if i.word not in stopwords_list]
    filtered_word_list = []
    
    #过滤eg['今天','t'],['的','d']
    for seg in seg_list:
        if len(seg.word) <= 1:
            continue
        elif seg.flag == "eng":
            if len(seg.word) <= 2:
                continue
            else:
                filtered_word_list.append(seg.word)
        elif seg.flag.startswith("n"):
            filtered_word_list.append(seg.word)
        elif seg.flag in ["x","vn","v","an"]:
            filtered_word_list.append(seg.word)
    return filtered_word_list
        

def split_text(text):
    split_index=[]

    # pattern1 = '。|，|,|;|；|\.|\?'
    pattern1 = '。|\.|\?'
    for m in re.finditer(pattern1,text):
        #获取断点的下标
        idx=m.span()[0]
        if idx == len(text)-1 or idx == len(text) - 2 or idx == len(text)-3:
            break
        if text[idx-1]=='\n':
            continue
        if text[idx-1].isdigit() and text[idx+1].isdigit():#前后是数字
            continue
        if text[idx-1].isdigit() and text[idx+1].isspace() and text[idx+2].isdigit():#前数字 后空格 后后数字
            continue
        if text[idx+1].isspace() and text[idx+2].isupper():#后空格，后大写字母
            continue
        if text[idx-1].islower() and text[idx+1].islower():#前小写字母后小写字母
            continue
        if text[idx-1].islower() and text[idx+1].isdigit():#前小写字母后数字
            continue
        if text[idx-1].isupper() and text[idx+1].isdigit():#前大写字母后数字
            continue
        if text[idx - 1].isdigit() and text[idx + 1].islower():#前数字后小写字母
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isupper():#前数字后大写字母
            continue
        if text[idx+1] in set('.。;；,，'):#前句号后句号
            continue

        if text[idx-1].isupper() and text[idx+1].isupper() :#前大些后大写
            continue
        if text[idx]=='.' and text[idx+1:idx+4]=='com':#域名
            continue
        split_index.append(idx+1)
    pattern2='\([一二三四五六七八九零十]\)|[一二三四五六七八九零十]、|'
    pattern2+='注:|附录 |表|表 \d|Tab \d+|\[摘要\]|\[提要\]|表\d[^。，,;]+?\n|图 \d|Fig \d|'
    pattern2+='\[Abstract\]|\[Summary\]|前  言|【摘要】|【关键词】|结    果|讨    论|'
    pattern2+='and |or |with |by |because of |as well as '
    for m in re.finditer(pattern2,text):
        idx=m.span()[0]
        if (text[idx:idx+2] in ['or','by'] or text[idx:idx+3]=='and' or text[idx:idx+4]=='with')\
            and (text[idx-1].islower() or text[idx-1].isupper()):
            continue
        split_index.append(idx)

    pattern3='\n\d\.'#匹配1.  2.  这些序号
    for m in re.finditer(pattern3, text):
        idx = m.span()[0]
        if ischinese(text[idx + 3]):
            split_index.append(idx+1)
    
    pattern4='\n\d\ \n'#匹配1  2  这些大标题序号 换行后 +中文标题
    for m in re.finditer(pattern4, text):
        idx = m.span()[0]
        if ischinese(text[idx + 4]):
            split_index.append(idx+1)

    pattern5='\n\d\、'#匹配1、2、  这些序号
    for m in re.finditer(pattern5, text):
        idx = m.span()[0]
        if ischinese(text[idx + 3]):
            split_index.append(idx+1)
    
    for m in re.finditer('\n\(\d\)',text):#匹配(1) (2)这样的序号
        idx = m.span()[0]
        split_index.append(idx+1)
    for m in re.finditer('\n\（\d\）',text):#匹配（1） （2）这样的序号
         idx = m.span()[0]
         split_index.append(idx+1)
    split_index = list(sorted(set([0, len(text)] + split_index)))#所有的split断电，包括开始和结尾

    other_index=[]
    for i in range(len(split_index)-1):
        begin=split_index[i]
        end=split_index[i+1]
        if text[begin] in '一二三四五六七八九零十' or \
                (text[begin]=='(' and text[begin+1] in '一二三四五六七八九零十'):#如果是一、和(一)这样的标号
            for j in range(begin,end):
                if text[j]=='\n':#一 *****之后截断
                    other_index.append(j+1)
    split_index+=other_index
    split_index = list(sorted(set([0, len(text)] + split_index)))


    for i in range(1,len(split_index)-1):# 去掉全部是空格的句子，把空格给下一个句子开头
        idx=split_index[i]
        while idx>split_index[i-1]-1 and text[idx-1].isspace():
            idx-=1
        split_index[i]=idx
    split_index = list(sorted(set([0, len(text)] + split_index)))

    #查看最长句和最短句的长度
    lens = [split_index[i+1]-split_index[i] for i in range(len(split_index)-1)]
    # print(max(lens),min(lens))
    result=[]
    for i in range(len(split_index)-1):
        result.append(text[split_index[i]:split_index[i+1]])


    #做一个检查
    s=''
    for r in result:
        s+=r
    # print(len(s))
    # print(len(result))
    assert  len(s)==len(text)
    # print(result)
    
    context = []
    for seg in result:
        context.append(seg.replace('\n',''))
        
    return context


def process_text(file_name, split_method=None):
    """
    读取文本  切割 分词 
    ：param file：文件命
    ：param split_method 切割文本的方法
    :return
    """
    # print(file_name)
    if split_method is None:
        with open(file_name,'r',encoding='utf-8') as f:
            texts=f.readlines()
    else:
        with open(file_name, 'r', encoding='utf-8') as f:
            texts=f.read()
            texts=split_method(texts)
    return texts
 
    
def list2str(text_list):              
    outstr = ''  # 返回值是字符串
    for line in text_list:
        for word in line:
            outstr += str(word)
            outstr += " "
    return outstr

#数据预处理
def get_train_data(train_path, processed_file_path,tagged_file_path):
    """
    对数据预处理，将数据处理成模型训练所需要的格式
    :return: 返回处理完成的数据列表
    """
    filter_pattern = "^[u4e00-\u9fa5-zA-Z0-9]" #去除数值
    #读取停用词
    documents =[]
    file_list = os.listdir(train_path)
    #遍历数据
    num = 0
    
    f_processed = open(processed_file_path,"w",encoding = 'utf8')
    f_tagged = open(tagged_file_path,"w",encoding = 'utf8')

    for file in file_list:
        text = process_text(os.path.join(train_path,file),split_text)
        doc_list = []
        for sentence in text:
            filter_sen = cut_sentence(sentence,conf.stop_word_path)     
            doc_list.append(filter_sen)
        text = list2str(doc_list)
        f_processed.write(text)
        f_processed.write('\n')
        f_tagged.write(file)
        f_tagged.write('\n')
    f_processed.close()
    f_tagged.close()
    texts = gensim.models.doc2vec.TaggedLineDocument(processed_file_path)
    return texts
            

#训练doc2vec 模型
def train():
    """
    训练文档向量并保存训练好的模型
    :return:
    """
    #生成训练数据
    documents = get_train_data(conf.processed_file_path,conf.tagged_file_path)
    # print("documents:",documents)
    model = None
    #训练模型
    model= Doc2Vec(vector_size=300,inter=2,alpha=0.025, min_alpha=0.025,min_count = 1,sample=1e-3,negative=5,workers=4,dm=0,epoches = 54,window=10)

    model.build_vocab(documents)
    # model.train(documents, total_examples=model.corpus_count,epochs = model.epochs)
    for epoch in range(5):
    	model.train(documents, total_examples=model.corpus_count, epochs=10)
    	model.alpha -= 0.002
    	model.min_alpha = model.alpha
    	model.train(documents, total_examples=model.corpus_count, epochs=10)
    model.save(conf.d2v_model_path)
    # model = Doc2Vec.load(conf.d2v_model_path)
    # for x,y in documents:
    #     infer_vec = model.infer_vector(x,epochs=64)
    #     sims = model.docvecs.most_similar([infer_vec],topn=len(model.docvecs))
    #     # rank = [docid for docid,sim in sims].index(i)
    #     # ranks.append(rank)

    #     for docid,sim in sims:
    #         print(y,docid,sim)
    # import sys 
    # sys.exit()
  
#文本向量模型效果评估
def assess_model(model_path,processed_file_path):
    rank = []
    ranks = []
    tag = []
    f_processed = open(processed_file_path,"r",encoding = 'utf8')
    i = 0
    # model = Doc2Vec.load(model_path)
    model = torch.load(model_path)
    for line in f_processed:
        seg = line.strip().split(" ") 
        infer_vec = model.infer_vector(seg,epochs=64)
    
        sims = model.docvecs.most_similar([infer_vec],topn=len(model.docvecs))
        rank = [docid for docid,sim in sims].index(i)
        ranks.append(rank)
        # print(rank)
        # if(i == 0):
        #     print(sims)
        i = i+1  
    counter = collections.Counter(ranks)
    print(counter)
    all_count = 0
    for item,value in counter.items():
        all_count += value
    acc = counter.get(0)/all_count
    log.info('model acc：'+str(acc))
    print("model acc: ",acc)

#加载文本向量
def load_model(model_path,processed_file_path):
    rank = []
    ranks = []
    tag = []
    f_processed = open(processed_file_path,"r",encoding = 'utf8')
    i = 0
    model = Doc2Vec.load(model_path)
    for line in f_processed:
        seg = line.strip().split(" ") 
        infer_vec = model.infer_vector(seg,epochs=64)
    
        sims = model.docvecs.most_similar([infer_vec],topn=len(model.docvecs))
        rank = [docid for docid,sim in sims].index(i)
        ranks.append(rank)
        # print(rank)
        # if(i == 0):
        #     print(sims)
        i = i+1  
    counter = collections.Counter(ranks)
    print(counter)
    all_count = 0
    for item,value in counter.items():
        all_count += value
    acc = counter.get(0)/all_count
    log.info('model acc：'+str(acc))
    # print("acc",acc)
    
#文本向量查询
def query_vector(content,model):
    """
    新文本向量生成
    :param content: 文本内容
    :param model:训练好的模型
    :return: 新文本的向量
    """
    #处理文本
    words = cut_sentence(content,conf.stop_word_path)
    #s生成文本向量
    vector =model.infer_vector(words)
    return vector

#得到所有的文本向量
def get_vector(model_path):
    """
    得到所有的文本向量
    :param content: 文本向量模型的路径
    :return: 文本向量矩阵
    """
    model = torch.load(model_path)
    #model = Doc2Vec.load(model_path)
    vecs = [np.array(model.docvecs[i].reshape(1, len(model.docvecs[i]))) for i in range(model.corpus_count)]
    return np.concatenate(vecs)


    
if __name__ == '__main__':
    print("test")
        
