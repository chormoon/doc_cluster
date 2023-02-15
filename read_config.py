# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 21:45:41 2020

@author: lixy
读配置文件的路径参数&模型参数
"""

import sys
import configparser
import os

def loadConf():  # 加载配置文件
    cf = configparser.ConfigParser()
    print(os.getcwd())
    # 配置文件路径修改为绝对路径--niusj
    filepath = os.path.abspath(os.path.join(os.getcwd(), "config/system.conf"))
    cf.read(filepath,encoding='UTF8')
    print(filepath)
    return SysConf(cf)


def getValue(cf, active, name):  # 读取配置文件中的对应值
    print(cf.get(active, name))
    try:
        return cf.get(active, name)
    except:
        try:
            tmp = name.split("_")
            return cf.get(tmp[0], tmp[1])
        except:
            return None


class SysConf:  # 配置文件对象
    def __init__(self,cf):
        self.stop_word_path = getValue(cf, 'sys', "stop_word_path")
        self.d2v_model_path = getValue(cf, 'sys', "d2v_model_path")
        self.train_dir = getValue(cf, 'sys', "train_dir")
        self.processed_file_path = getValue(cf, 'sys', "processed_file_path")
        self.tagged_file_path = getValue(cf, 'sys', "tagged_file_path")
        self.log_file_path = getValue(cf, 'sys', "log_file_path")
        self.result_dir = getValue(cf, 'sys', "result_dir")
        # print("~~~~~~~~~~~~~~" ,self.stop_word_path)
        # print("~~~~~~~~~~~~~~" ,self.d2v_model_path)
        # print("~~~~~~~~~~~~~~" ,self.train_dir)
        # print("~~~~~~~~~~~~~~" ,self.processed_file_path)
        # print("~~~~~~~~~~~~~~" ,self.tagged_file_path)


  
        
