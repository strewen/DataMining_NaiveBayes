#!/usr/bin/env python3
"""
NaiveBayes模型
author:strewen
Create On:2018/06/02
"""
import pandas as pd
import random
import math

#将数据集分为训练集与测试集
#将前test_count条记录作为测试集，剩下的当训练集
def split_train_test_set(x,y,test_count):
    x_test=x[:test_count]
    y_test=y[:test_count]
    x_train=x[test_count:]
    y_train=y[test_count:]
    return x_train,y_train,x_test,y_test

#数据预处理
def pretreatment():
    data=pd.read_table('./fruit.txt')
    pos=len(data.columns)
    #将数据的第二列与第三列合并为新标签(如：apple_granny_smith,mandarin_mandarin,apple_braeburn等）
    new_class=["_".join([row[1],row[2]]) for row in data.values]

    #将新标签集插入数据的最后一列
    data.insert(pos,pos,new_class)

    #提取数据集中第四列以后的特征
    #新的数据列为('mass' 'width' 'height' 'color_score' 'new_class')
    data=data.iloc[:,3:].values.tolist()

    #将数据集打乱
    random.shuffle(data)
    
    #提取特征集('mass' 'width' 'height' 'color_score')
    x=[data[i][:-1] for i in range(len(data))]

    #提取标签集('new_class')
    y=[data[i][-1] for i in range(len(data))] 
    return split_train_test_set(x,y,int(0.2*len(data)))
    

#分类准确率
def accurracy(pred_y,y):
    correct=0
    count=len(y)
    for i in range(count):
        #因为pred_y，y都是新类别的集合，
        #需将“_“作为分割符分割，再取第一值(即apple,lenmon...)计较
        if pred_y[i].split('_')[0]==y[i].split('_')[0]:
            correct+=1
    return correct/count

class NaiveBayes:
    def __init__(self):
        self.train=None  #训练集
        self.labels=None #类标签
        self.mean=None   #各类各特征的均值
        self.var=None    #各类各特征的方差
        self.class_group=None  #按类分组结果集   

    def fit(self,x,y):
        self.train=pd.DataFrame(x)
        pos=len(self.train.columns)
        #将标签集插入特征集最后一列，作为训练集
        self.train.insert(pos,pos,y)

        #将训练集按标签分组
        self.class_group=self.train.groupby(self.train.iloc[:,-1])

        #获取分组后各特征的均值（结果类似如下）
        #       特征1  特征2  特征3
        #class1  n11    n12    n13
        #class2  n21    n22    n23
        #...... ...     ...    ...
        #ps:n11是类别1，特征1的均值
        self.mean=self.class_group.mean()

        #获取分组后各特征的方差（结果类似如下）
        #       特征1  特征2  特征3
        #class1  n11    n12    n13
        #class2  n21    n22    n23
        #...... ...     ...    ...
        #ps:n11是类别1，特征1的方差        
        self.var=self.class_group.var() 

        #获取标签
        self.labels=self.class_group.count().index.tolist()  

    #高斯函数
    def gauss(self,mean,var,value):
        if var==0:
            var=0.0001
        coff=1/(math.sqrt(2*math.pi*var))
        exponent=math.exp(-pow(value-mean,2)/(2*var))
        return coff*exponent

    #预测函数 
    def pred(self,x):
        #存放各样本的预测结果
        y_pred=[]

        for sample in x:
            class_probility=self.class_group.count().iloc[:,-1].tolist()   #各标签的实例总数
            
            #各标签的概率集[P(class1),P(class2)...]
            class_probility=[class_mem/sum(class_probility) for class_mem in class_probility]

            for i in range(len(sample)):
                #存放各标签某个特征的先验概率[P(class1|特征1),P(class2|特征1)....]
                probility=[]
                for j in range(len(self.labels)):
                    probility.append(self.gauss(self.mean.iloc[j,i],self.var.iloc[j,i],sample[i]))

                #计算后验概率
                class_probility=[class_probility[k]*probility[k] for k in range(len(probility))]
            #获取后验概率最大的标签索引
            max_index=class_probility.index(max(class_probility))
            y_pred.append(self.labels[max_index])
        return y_pred

def main():
    x_train,y_train,x_test,y_test=pretreatment()
    ctl=NaiveBayes()
    ctl.fit(x_train,y_train)
    res=ctl.pred(x_test)
    print(accurracy(res,y_test))

if __name__=='__main__':
    main()
