import matplotlib.pyplot as plt
import numpy as np

'''
该文件只用于提供pca和lda的类，不做运行使用
'''

def printData(data1,data2=[],data3=[],title=""):
    if len(data2)!=0:
        plt.scatter(data2[:,0],data2[:,1], marker = 'o', color = 'b', label='2',s=20,alpha = 0.3)
        plt.legend()
    if len(data3)!=0:
        plt.scatter(data3[:,0],data3[:,1], marker = '+', color = 'black', label='3',s=20,alpha = 0.3)
        plt.legend() 
    plt.scatter(data1[:,0],data1[:,1], marker = 'x', color = 'r', label='1',s=20,alpha =0.3)
    plt.legend() 
    if title!="":
        plt.title(title)
    plt.show()
    plt.close()
    

def totalprint(pt,title=""):
    pt1=np.zeros((1000,2))
    pt1[:,0]=pt[0:1000,0]
    pt2=np.zeros((1000,2))
    pt2[:,0]=pt[1000:2000,0]
    pt3=np.zeros((1000,2))
    pt3[:,0]=pt[2000:3000,0]
    printData(pt1,pt2,pt3,title)
    
class myPca():
    def __init__(self,n):
        self.n_com=n
        self.M=np.array(())
        self.total_m=np.array(())
        
    def transform(self,test):
        result=[]
        for i in range(len(test)):
            result.append(np.dot(test[i]-self.total_m,self.M.T))
        return result
    
    def svd_flip(self,u,v):
        max_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
        return u, v
    
    def fit_transform(self,t):
        if type(t)!=np.ndarray:
            t=np.array(t)
        t=t.astype(np.float)
        t_shape=np.shape(t)
        if t_shape[1]<self.n_com:
            print("erro:The number of target features is larger than the number of original features")
            exit(0)
        self.total_m=np.mean(t,axis=0)
        t-=self.total_m
        U, S, V = np.linalg.svd(t,full_matrices=False)
        U, V = self.svd_flip(U, V)
        U = U[:, :self.n_com]
        self.M = V[:self.n_com]
        U *= S[:self.n_com]
        return U
    
class myLda():
    def __init__(self,n):
        self.n=n
        self.total_m=np.array(())
        self.M=np.array(())
    def transform(self,test):
        return np.dot(test-self.total_m,self.M)[:,0:self.n]
    
    def fit_transform(self,train,trainy):
        t_shape=np.shape(train)
        ty_shape=np.shape(trainy)
        if t_shape[0]!=ty_shape[0] or t_shape[0]==0:
            print("axis and category length error")
            exit(0)
        if type(train)!=np.ndarray:
            train=np.array(train)
        if type(trainy)!=np.ndarray:
            trainy=np.array(trainy)
        m={}
        #m是一个字典
        #以每个类别作为key，对应数据作为values
        for i in range(t_shape[0]):
            if trainy[i] not in m.keys():
                m[trainy[i]]=[]
            m[trainy[i]].append(np.array(train[i]))
        #类内散布矩阵
        mid_t=[[] for i in range(len(m.keys()))]
        for i,key in enumerate(m.keys()):
            x=np.array(m[key])
            xi=np.array([np.mean(x[:,ti]) for ti in range(np.shape(x)[1])])
            mid_t[i]=x-xi
            if i==0:
                SW=np.dot(np.transpose(mid_t[i]),mid_t[i])
            else:
                SW+=np.dot(np.transpose(mid_t[i]),mid_t[i])
#        print("SW",SW)
        total_m=np.array([[np.mean(train[:,ti]) for ti in range(np.shape(train)[1])]])
        
        for i,key in enumerate(m.keys()):
            tm=np.array(m[key])
            mi=np.array([[np.mean(tm[:,ti]) for ti in range(np.shape(tm)[1])]])
            Ni=len(m[key])
            if i==0:
                SB=Ni*np.dot(np.transpose(mi-total_m),mi-total_m)
            else:
                SB+=Ni*np.dot(np.transpose(mi-total_m),mi-total_m)
#        print("SB",SB)
        S = np.dot(np.linalg.inv(SW),SB).astype(np.float)
#        print("S",S)
        
        value,feature=np.linalg.eig(S)  
        value=np.abs(value)
        feature=np.transpose(feature).astype(np.float)
#        print(value,"\n",feature)
        top_idx=np.argsort(-value)[0:self.n]
#        print(top_idx)
        M=[]
        for i in top_idx:
            M.append(feature[i])
        M=-np.transpose(M)
        self.M=np.array(M)
        self.total_m=np.array(total_m).astype(np.float)
        train=np.dot(train-total_m,M)
#        print(np.shape(train[:,0]))
        return train[:,0:self.n].astype(np.float)