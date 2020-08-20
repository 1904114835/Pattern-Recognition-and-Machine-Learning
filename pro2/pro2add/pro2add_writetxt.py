# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 23:19:43 2020

@author: 19041
"""

import cv2  
import numpy as np
import matplotlib.pyplot as plt

def show(im):
    if len(np.shape(im))!=0:
        if len(np.shape(im))==2:
            plt.imshow(im,cmap ='gray')
        else:
            plt.imshow(im)
        plt.show()
        plt.close()
def get_seq_method1(im):
#    if len(np.shape(im))!=2:
#        return -1
    isum=np.sum(im,axis=1)
    jsum=np.sum(im,axis=0)
#    print(isum,jsum)
    m_isum=np.mean(isum)
    m_jsum=np.mean(jsum)
    max_isum=np.max(isum)
    max_jsum=np.max(jsum)
    
    where_i=np.where(isum<m_isum)[0]
    where_j=np.where(jsum<m_jsum)[0]
    if len(where_i)==0 or len(where_j)==0:
        return 0
    len_head=int((where_i[-1]-where_i[0])/6)
    upline=where_i[0]-len_head
    
    mid_of_j=np.mean(where_j)
    lline=int(mid_of_j-len_head*1.2)
    rline=int(mid_of_j+len_head*1.2)
    
    if lline<0:
        lline=0
    if rline>np.shape(im)[0]:
        rline=np.shape(im)[0]
    if upline<0:
        upline=0
#    print(np.shape(im),lline,rline,upline)
#    im[upline,:]=0
#    im[:,lline]=0
#    im[:,rline]=0
#    show(im)
    
    r_seq=[0,1,2,3,4,5,6,7]
    result=0
    if isum[upline]<m_isum+(max_isum-m_isum)/2:
        result+=4
    if jsum[lline]<m_jsum+(max_jsum-m_jsum)/2:
        result+=2
    if jsum[rline]<m_jsum+(max_jsum-m_jsum)/2:
        result+=1
#    print(r_seq[result])
    return r_seq[result]
#    print(isum,jsum)
def get_seq_method2(im):
    isum=np.sum(im,axis=1)
    jsum=np.sum(im,axis=0)
#    print(isum,jsum)
    m_isum=np.mean(isum)
    m_jsum=np.mean(jsum)
    iline=int(np.shape(im)[0]/5)
    jline=int(np.shape(im)[1]/2)
    im[iline,:]=0
    im[:,jline]=0
#    show(im)
    r_seq=[0,1,2,3]
    result=0
    if isum[iline]<m_isum:
        result+=2
    if jsum[jline]<m_jsum:
        result+=1
    return r_seq[result]
    
    
def get_one_pic_seq(person,cate,num,gap,method=1):
    videoCapture = cv2.VideoCapture('C:/Users/19041/Desktop/video/'+person+'_'+cate+'_'+num+'_uncomp.avi')
    success, frame = videoCapture.read()
    frame=[]
    flag=0
    while True:
        success, tframe = videoCapture.read() #获取下一帧
        if success==False:
            break
        if flag%gap!=0:
            flag+=1
            continue
        threth=np.mean(tframe)
        ret,tframe=cv2.threshold(tframe,threth*2/3,255,cv2.THRESH_BINARY)
        tframe=np.mean(tframe,axis=2).astype(np.int)
        #提取特征
        if method==1:
            seq=get_seq_method1(tframe)
        else:
            seq=get_seq_method2(tframe)
        frame.append(str(seq))
        flag+=1
#        show(tframe)
    print(len(frame))
    if len(frame)==0:
        print(person,cate,num)
    return frame
    
if __name__=="__main__":
    cate=["boxing","handclapping","handwaving","jogging","running","walking"]
    gap=5
#    get_one_pic_seq("person01","jogging","d1",gap,2)
    allcate={}
    for i in cate:
        allcate[i]=[]
    writetxt=False
    if writetxt==True:
        f = open('seq.txt','w')        
        for i in cate:
            for pnum in range(1,26):
                for dum in range(1,5):
                    if pnum<10:                    
                        person="person"+"0"+str(pnum)
                    else:
                        person="person"+str(pnum)
                    d_num="d"+str(dum)
    #                print(person,d_num,i)
                    allcate[i].append(get_one_pic_seq(person,i,d_num,gap,2))
                    f.write(" ".join(allcate[i][-1])+"\n")
    else:
        print("请使用pro2add_readtxt.py进行下一步操作")
        print("或将writetxt变量更改为Ture更新seq.txt")
    f.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    