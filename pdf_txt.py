# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:24:09 2020

@author: lixy
"""

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument, PDFNoOutlines
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal, LAParams
from pdfminer.pdfpage import PDFPage
import  os

def pdf_to_word(folder,password):
    # 获取指定目录下面的所有文件
    files = os.listdir(folder)
    # 获取pdf类型的文件放到一个列表里面
    pdfFiles = [f for f in files if f.endswith(".pdf")]
    for pdfFile in pdfFiles:
        # 将pdf文件放到指定的路径下面
        pdfPath = os.path.join(folder, pdfFile)
        # 设置将要转换后存放word文件的路径
        wdPath = pdfPath
        # 判断是否已经存在对应的word文件，如果不存在就加入到存放word的路径内
        if pdfPath[-3:] != 'txt':
            wdPath = wdPath[0:-4] + ".txt"
            fn = open(pdfPath, 'rb')
              # 创建一个PDF文档分析器：PDFParser
        

            print(pdfPath)
            device = PDFPageAggregator(PDFResourceManager(), laparams=LAParams())
            interpreter = PDFPageInterpreter(PDFResourceManager(), device)

            parser = PDFParser(open(pdfPath, 'rb'))
            document = PDFDocument(parser)
            # 检测文档是否提供txt转换，不提供就忽略
 
            if not document.is_extractable:
                print("PDFTextExtractionNotAllowed")
            else:
                with open(wdPath, 'w', encoding='utf-8') as f:
                    page_list = list(PDFPage.create_pages(document))
                    page_list_length = len(page_list)
                    print('The number of PDF is: ', page_list_length)
                
                    for page in PDFPage.create_pages(document):
                        # 接受LTPage对象
                        interpreter.process_page(page)
                
                        # 获取LTPage对象的text文本属性
                        layout = device.get_result()
                        for x in layout:
                            if isinstance(x, LTTextBoxHorizontal):
                                results = x.get_text()
                                f.write(results)

 
if __name__ == '__main__':
    pdf_to_word("E:/Work/BWD/DAY1/doc_cluster/data_pdf","")
