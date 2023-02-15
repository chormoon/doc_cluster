# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 20:02:54 2020

@author: lixy
"""

import os,time,logging
#日志存放路径

class Log():
    def __init__(self,log_path):       
    #再日志路径下添加日志文件名        
        self.logname=os.path.join(log_path,'%s.log'%time.strftime('%Y_%m_%d'))        
    #logger日志对象初始化        
        self.logger=logging.getLogger()        
    #设置日志等级        
        self.logger.setLevel(logging.DEBUG)        
    #日志输出格式        
        self.formatter=logging.Formatter('[%(asctime)s]-%(filename)s]-%(levelname)s:%(message)s')    
    def __console(self,level,message):        
    # 创建一个 FileHandler，用于写到本地        
        fh=logging.FileHandler(self.logname,'a',"utf-8")     
        print(self.logname)
        fh.setLevel(logging.DEBUG)        
        fh.setFormatter(self.formatter)        
        self.logger.addHandler(fh)        
    # 创建一个 StreamHandler,用于输出到控制台        
        ch = logging.StreamHandler()        
        ch.setLevel(logging.DEBUG)        
        ch.setFormatter(self.formatter)        
        self.logger.addHandler(ch)        
        if level=='info':          
            self.logger.info(message)        
        elif level=='debug':          
            self.logger.debug(message)        
        elif level=='warning':           
            self.logger.warning(message)        
        elif level=='error':           
            self.logger.error(message)        
      # 这两行代码是为了避免日志输出重复问题        
        self.logger.removeHandler(ch)        
        self.logger.removeHandler(fh)        
      # 关闭打开的文件        
        fh.close()
    def debug(self, message):       
        self.__console('debug', message)    
    def info(self, message,*args):       
        self.__console('info', message)   
    def warning(self, message,*args):       
        self.__console('warning', message)    
    def error(self, message,*args):      
        self.__console('error', message)
if __name__ == "__main__":    
    log=Log(".\log")    
    log.info("---测试开始---")    
    log.info("操作步骤1,2,3")    
    log.warning("---测试结束---")  