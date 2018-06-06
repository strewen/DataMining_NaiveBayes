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
    new_class=["_".join([row[1],row[2]]) for row in data.values]  #新的类型(fruit_name,subtype组成一列)
    data.insert(pos,pos,new_class)                
    data=data.iloc[:,3:].values.tolist()
    random.shuffle(data)   #将数据打乱
    x=[data[i][:-1] for i in range(len(data))]  #提取特征集
    y=[data[i][-1] for i in range(len(data))]   #提取标签集
    return split_train_test_set(x,y,int(0.2*len(data)))
    

#分类准确率
def accurracy(pred_y,y):
    correct=0
    count=len(y)
    for i in range(count):
        if pred_y[i].split('_')[0]==y[i].split('_')[0]:
            correct+=1
    return correct/count

class NaiveBayes:
    def __init__(self):
        self.train=None
        self.labels=None
        self.mean=None
        self.var=None
        self.class_group=None

    def fit(self,x,y):
        self.train=pd.DataFrame(x)
        pos=len(self.train.columns)
        self.train.insert(pos,pos,y)
        self.class_group=self.train.groupby(self.train.iloc[:,-1]) #将数据集按标签分组
        self.mean=self.class_group.mean()            #获取分组后各特征的均值
        self.var=self.class_group.var()             #获取分组后各特征的方差
        self.labels=self.class_group.count().index.tolist()  

    #高斯函数
    def gauss(self,mean,var,value):
        if var==0:
            var=0.001
        coff=1/(math.sqrt(2*math.pi*var))
        exponent=math.exp(-pow(value-mean,2)/(2*var))
        return coff*exponent

    #分类函数：分类一个实例 
    def classify(self,sample):
        class_probility=self.class_group.count().iloc[:,-1].tolist()   #各标签的实例总数
        class_probility=[class_mem/sum(class_probility) for class_mem in class_probility] #各标签的概率集
        for i in range(len(sample)):
            probility=[]
            for j in range(len(self.labels)):
                probility.append(self.gauss(self.mean.iloc[j,i],self.var.iloc[j,i],sample[i]))
            class_probility=[class_probility[k]*probility[k] for k in range(len(probility))]
        max_index=class_probility.index(max(class_probility))
        return self.labels[max_index]

    #预测函数
    def pred(self,x):
        y_pred=[]
        for sample in x:
            y=self.classify(sample)
            y_pred.append(y)
        return y_pred

def main():
    x_train,y_train,x_test,y_test=pretreatment()
    ctl=NaiveBayes()
    ctl.fit(x_train,y_train)
    res=ctl.pred(x_test)
    print(accurracy(res,y_test))

if __name__=='__main__':
    main()
