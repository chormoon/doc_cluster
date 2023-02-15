# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:25:36 2020

@author: lixy
"""

from win32com import client as wc
import os
 
 

mypath = 'E:\\2020\\data_classfier\\学习数据_all'
all_FileNum = 0
 
def Translate(level, path):
    global all_FileNum
    '''
    将一个目录下所有doc文件转成txt
    '''
    #该目录下所有文件的名字
    files = os.listdir(path)
    docFiles = [f for f in files if f.endswith(".doc") or f.endswith(".docx")]

    for docFiles in files:
        if (docFiles[0] == '~' or docFiles[0] == '.'):
            continue
        new = path + '\\' + docFiles
        print(new)
        #除去后边的.doc后缀
        if docFiles.endswith(".doc"):
            tmp = new[:-4]
        else:
            tmp = new[:-5]
        #改成txt格式
        word = wc.Dispatch('Word.Application')
        doc = word.Documents.Open(tmp)
        doc.SaveAs(tmp + '.txt', 4)
        doc.Close()
        all_FileNum = all_FileNum + 1
if __name__ == '__main__':
    Translate(1, mypath)
    print('文件总数 = ', all_FileNum)
 

