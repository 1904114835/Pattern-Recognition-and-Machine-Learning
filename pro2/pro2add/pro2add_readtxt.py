# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:15:01 2020

@author: 19041
"""

import numpy as np
import Pro2
if __name__=="__main__":
    #cate的顺序决定了从txt中读取的顺序，请务必不要更改
    cate=["boxing","handclapping","handwaving","jogging","running","walking"]
    hidden_num=10#需要调参
    ob_num=4#method1提取特征数量为8，method2是4
    d={}
    for i in cate:
        d[i]=[]
    f = open('seq.txt')
    for i in range(600):
        line =f.readline()
        line =np.array([int(ti) for ti in line.split()])
        d[cate[int(i/100)]].append(line)
    f.close()
    
    train={}
    test={}
    #构造训练测试序列
#    #奇数训练，偶数测试，即偶数下标训练，奇数下标测试
    for key in d.keys():
        train[key]=[]
        test[key]=[]
        for i in range(len(d[key])):
            if i%2==0:
                for t in (d[key][i]):
                    train[key].append(t)
            else:
                test[key].append(d[key][i])
        train[key]=np.array(train[key])
    hmm={}
    for key in d.keys():
        hmm[key]=Pro2.HMM(hidden_num,ob_num)
        hmm[key].bw_train(train[key])
    print("hidden num:",hidden_num)
    t=0
    score=np.zeros((6,6))
    for key in d.keys():
        for i in range(len(test[key])):
            testl=[]
            for keyhmm in hmm.keys():
                testl.append(hmm[keyhmm].prob(test[key][i]))
            score[t][np.argmax(testl)]+=1
        print(key,score[t][t]/(len(test[key])))
        t+=1
    print(score)