import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def readData():
    data1=[]
    data2=[]
    data3=[]
    with open("data1.txt", "r") as f:
        data = f.readlines()
        for i in data:
            data1.append([eval(i.strip().split()[0]),eval(i.strip().split()[1])])
    with open("data2.txt", "r") as f:
        data = f.readlines()
        for i in data:
            data2.append([eval(i.strip().split()[0]),eval(i.strip().split()[1])])
    with open("data3.txt", "r") as f:
        data = f.readlines()
        for i in data:
            data3.append([eval(i.strip().split()[0]),eval(i.strip().split()[1])])
    return np.array(data1),np.array(data2),np.array(data3)

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
        
    def fit_transform(self,t):
        #计算样本均值
        t_shape=np.shape(t)
        if t_shape[1]<self.n_com:
            print("erro:The number of target features is larger than the number of original features")
            exit(0)
        x0=[]
        for i in range(t_shape[1]):
            x0.append(np.mean(t[:,i]))
        x0=np.array(x0)
        #中心化
        mid_t=t-x0
        #计算散度矩阵,
        #这里本应用t-m作为z，但由于中心化无需再减
        S=np.dot(np.transpose(mid_t),mid_t)
        #求特征值和特征向量
        #这里非常坑，np的linalg.eig是按照行取的
        #但是我们的特征值要按照列
        #因为feature是方阵
        #做一个转置就相当于按列取了
        value,feature=np.linalg.eig(S)  
        value=np.abs(value)
        feature=np.transpose(feature)
#        print(value,"\n",feature)
        #因为要取大的，而argsort从小排，故传入负值
        top_idx=np.argsort(-value)[0:self.n_com]
#        print(top_idx)
        M=[]
        for i in top_idx:
            M.append(feature[i])
#        print(M)
        M=np.transpose(M)
        y=np.dot(mid_t,M)
        return -y
    
class myLda():
    def __init__(self,n):
        self.n=n
        self.total_m=np.array(())
        self.M=np.array(())
    def transform(self,test):
        return np.dot(test-self.total_m,self.M)
    def fit_transform(self,train,trainy):
        t_shape=np.shape(train)
        ty_shape=np.shape(trainy)
        if t_shape[0]!=ty_shape[0] or t_shape[0]==0:
            print("axis and category length error")
            exit(0)
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
        S = np.dot(np.linalg.inv(SW),SB)
#        print("S",S)
        
        value,feature=np.linalg.eig(S)  
        value=np.abs(value)
        feature=np.transpose(feature)
#        print(value,"\n",feature)
        top_idx=np.argsort(-value)[0:self.n]
#        print(top_idx)
        M=[]
        for i in top_idx:
            M.append(feature[i])
        M=-np.transpose(M)
        train=np.dot(train-total_m,M)
#        print(np.shape(train[:,0]))
        self.M=M
        self.total_m=total_m
        return train[:,0:self.n]

    
if __name__=='__main__':    
    d1,d2,d3=readData()
    train=np.vstack((d1[0:1000,],d2[0:1000,],d3[0:1000,]))
    trainy=np.zeros((len(train)))
    trainy[0:1000]=1
    trainy[1000:2000]=2
    trainy[2000:3000]=3
    test=np.vstack((d1[1000:2000,],d2[1000:2000,],d3[1000:2000,]))
    testy=np.copy(trainy)
    printData(d1[0:1000,],d2[0:1000,],d3[0:1000,])
#    for k in range(1,35,2):
    k=5
    print("k=",k)
    #先用sklearn的pca做一个示例
    pca=PCA(n_components=1)
    pt=pca.fit_transform(train)
    totalprint(pt,"sklearn_pca")
#    print()
    #手写的pca
    mypca=myPca(n=1)
    mpt=mypca.fit_transform(train)
    totalprint(pt,"mypca")
#    print("mypca",mpt[0:10,],"\nskpca",pt[0:10,])
    print("Sum of difference between sklearn pca and handwritten pca：",np.sum(abs(mpt-pt)))
    knn=sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(mpt,trainy)
    pcatest=mypca.fit_transform(test)
    scored=[0,0,0]
    testlist=knn.predict(pcatest)
    for i in range(len(testlist)):
        if testlist[i] == testy[i]:
            scored[int(testy[i])-1]+=1
    print("Correct number for each category(total 1000):",scored)
    print("error rate：",1-sum(scored)/len(pcatest))
    del pca
    del mypca
    #LDA
    lda = LinearDiscriminantAnalysis(n_components=1)
    lt=lda.fit_transform(train,trainy)
    totalprint(lt,"sklearn_lda")
    mylda=myLda(n=1)
    mlt=mylda.fit_transform(train,trainy)
    totalprint(mlt,"mylda")
#    print("sklda",lt[0:10,],"\nmylda",mlt[0:10,])
    print("Sum of difference between sklearn lda and handwritten lda：",np.sum(abs(mlt-lt)))
    knn2=sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    knn2.fit(mlt,trainy)
    ldatest=mylda.transform(test)
    scored=[0,0,0]
    testlist=knn2.predict(ldatest)
    for i in range(len(testlist)):
        if testlist[i] == testy[i]:
            scored[int(testy[i])-1]+=1
    print("Correct number for each category(total 1000):",scored)
    print("error rate：",1-sum(scored)/len(ldatest))
    knn2.fit(lt,trainy)
    ldatest=lda.fit_transform(test,testy)
    scored=[0,0,0]
    testlist=knn2.predict(ldatest)
    for i in range(len(testlist)):
        if testlist[i] == testy[i]:
            scored[int(testy[i])-1]+=1
    print("Correct number for each category(total 1000):",scored)
    print("error rate：",1-sum(scored)/len(ldatest))
    del lda
    del mylda
    
        
    
    