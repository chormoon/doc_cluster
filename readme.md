只需调用cluster.py即可

get_train_vector(0,101,"E:/Work/BWD/DAY1/doc_cluster/data","E:/Work/BWD/DAY1/doc_cluster/cluster_data",8)

参数解析：  


   first arg:  训练模式
               # 1为从零训练模式   清零模型  将data文件夹中所有语料文件用于训练并分类
               # 0为增量训练模式   加载原有模型  将cluster_data文件夹中所有文件进行向量计算与分类
               # 少量文件新增时选用增量训练  定期从零训练可增加准确率
         
         
   second arg: 分类任务的ID，影响处理过程中的文件命名，可以随意填写
   
   
   third arg:  全部训练数据的地址
   
   
   Forth arg:  新增想要分类的数据的地址
   
   
   Fifth arg:  设置最终想得到的类的数量
    

使用完成后cluster_data文件夹将自动清空  所有文件分类后需手动检验归档
如果有需要也可以保留cluster_data文件夹中的文件  shutil.move改成shutil.copy即可
