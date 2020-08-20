import HW1
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

def get_random_list(m, n):
    total=[i for i in range(0,m)]
    results=[]
    for i in range(n):
        t = random.randint(0,len(total)-1)
        results.append(total[t])
        del total[t]
    return results

def read_img(way_string):
    im = Image.open(way_string)    # 读取文件
#    im.show()    # 展示图片
    return im

def downim(im,smaller):
#    t=np.shape(im)
#    print(t)
#    if len(t)!=2:
#        print("not image")
#        exit(0)
    if type(im) is not np.ndarray:
        im=np.array(im)
    return im[0:-1:smaller,0:-1:smaller]

def listImShow(im):
    plt.figure()
    for i in range(1,len(im)+1):
        plt.subplot(1,len(im),i)
        if len(np.shape(im[0]))==2:   
            plt.imshow(im[i-1].astype(np.int),cmap=plt.cm.gray)
        else:
            plt.imshow(im[i-1].astype(np.int))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    plt.close()
    
def show(im):
    plt.imshow(im)  
    plt.show()
    plt.close()
    
def getData(randomlist,down=4):
    ims=[]
    N=len(randomlist)
    showims=[[] for j in range(len(randomlist))]
    ty=["centerlight","glasses","happy","leftlight","noglasses","normal","rightlight","sad","sleepy","surprised","wink"]
    for alli in range(N):
        ims.append([])
        i=randomlist[alli]
        for j in range(15):
            totalway="./project1-data-Recognition/"
            if j<=8:
                nums="0"+str(j+1)
            else:
                nums=str(j+1)
            ways=totalway+"subject"+nums+"."+ty[i]+".pgm"
            imone=np.array(read_img(ways))     # 调用read_img()
            imone=downim(imone,down)#4倍下采样
            showims[alli].append(imone)
            imone=np.reshape(imone,-1)
            ims[alli].append(imone)
    ims=np.array(ims)
#    print(ims.shape)
    return ims,showims

def getTestlist(randomlist):
    testRandomList=[]
    for i in range(11):
        if i not in randomlist:
            testRandomList.append(i)
    return testRandomList

class myKnn():
    def __init__(self,n_neighbors):
        self.K=n_neighbors
    def fit(self,trainx,trainy):
        if type(trainx)!=np.ndarray:
            self.trainx=np.array(trainx)
        else:            
            self.trainx=trainx
        if type(trainy)!=np.ndarray:
            self.trainy=np.array(trainy)
        else:            
            self.trainy=trainy
            
    def predict(self,test):
        if type(test)!=np.ndarray:
            test=np.array(test)
        if test.shape==self.trainx[0].shape:
            pass
        else:
            result=[]
            for i in range(len(test)):
                dif=np.tile( test[i],(self.trainx.shape[0],1))-self.trainx
                sqrdif=dif**2
                sqrdif_sum=sqrdif.sum(axis=1)
                dis=sqrdif_sum**0.5
                sortdis=dis.argsort()
                count = {}
                for ti in range(self.K):
                    voteLabel = self.trainy[ sortdis[ti] ]
                    count[ voteLabel ] = count.get(voteLabel, 0) + 1  
                maxCount = 0
                for key, value in count.items():
                    if value > maxCount:
                        maxCount = value
                        argmax = key
                result.append(argmax)
        return result
            
def onlypca():
#    for N in range(3,12,2):
#        for K in range(10,60,10):
#            for nn_K in range(1,9,2):
#    N=3
#    K=20
    nn_K=1
    episode=10
    misss=[]
    for N in [3,5,7]:
        for K in range(10,90,10):
            for tti in range(episode):
                randomlist=get_random_list(11,N)
                #获取数据
                ims,showims=getData(randomlist)
                test,showtest=getData(getTestlist(randomlist))
        #        #展示数据
        #        for i in range(np.shape(showims)[0]):
        #            listImShow(showims[i])
                #准备工作和knn
                trainx=[]
                trainy=[]
                testx=[]
                testy=[]
                for i in range(np.shape(ims)[0]):
                    for j in range(np.shape(ims)[1]):
                        trainx.append(ims[i][j])
                        trainy.append(j)
                pca=HW1.myPca(K)#只使用pca降维
                trainx=pca.fit_transform(trainx)
                
                for i in range(np.shape(test)[0]):
                    for j in range(np.shape(test)[1]):
                        testx.append(test[i][j])
                        testy.append(j)
                testx=pca.transform(testx)
                knn = myKnn(n_neighbors=nn_K)
                knn.fit(trainx,trainy)#只使用pca降维
                
                result=knn.predict(testx)
                
                miss=0
                for i in range(len(result)):
                    if result[i]!=testy[i]:
                        miss+=1
                miss/=len(result)
                print("N:",N,"K:",K,"nn_K:",nn_K,"m_rate:%.4f"%miss)
                misss.append(miss)
                if tti%10==9:
                    print("average miss of",10,"times:",round(sum(misss[-10:])/10,4))
        #    print("average miss of",str(episode),"times:",round(sum(misss)/episode,4))
        
    
def pca_lda():
#    for N in range(3,12,2):
#        for K in range(10,60,10):
#            for nn_K in range(1,9,2):
#    N=3
#    K=50
    nn_K=1
    episode=10
    misss=[]
    for N in [3,5,7]:
        for K in range(10,90,10):
            for tti in range(episode):
                randomlist=get_random_list(11,N)
                #获取数据
                ims,showims=getData(randomlist)
                test,showtest=getData(getTestlist(randomlist))
        #        #展示数据
        #        for i in range(np.shape(showims)[0]):
        #            listImShow(showims[i])
                #准备工作和knn
                trainx=[]
                trainy=[]
                testx=[]
                testy=[]
                for i in range(np.shape(ims)[0]):
                    for j in range(np.shape(ims)[1]):
                        trainx.append(ims[i][j])
                        trainy.append(j)
            #    pca预处理防止奇异矩阵
                pca=HW1.myPca(len(trainx)-15)
                trainx=pca.fit_transform(trainx)
                #lda降维
                lda = HW1.myLda(K)
                trainAfterLDA=lda.fit_transform(trainx,trainy)
                
                for i in range(np.shape(test)[0]):
                    for j in range(np.shape(test)[1]):
                        testx.append(test[i][j])
                        testy.append(j)
                testx=pca.transform(testx)
                knn = myKnn(n_neighbors=nn_K)
                testx=lda.transform(testx)
                knn.fit(trainAfterLDA,trainy)
                
                result=knn.predict(testx)
                miss=0
                for i in range(len(result)):
                    if result[i]!=testy[i]:
                        miss+=1
                miss/=len(result)
                print("N:",N,"K:",K,"nn_K:",nn_K,"m_rate:%.4f"%miss)
                misss.append(miss)
                if tti%10==9:
                    print("average miss of",10,"times:",round(sum(misss[-10:])/10,4))
        #    print("average miss of",str(episode),"times:",round(sum(misss)/episode,4))
    
if __name__ == "__main__":
    onlypca()
    pca_lda() 
    