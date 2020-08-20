import midpro1
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier

def plotP(d):
    colors=['Reds','Blues','binary']
    u=[0 for i in range(len(d1[0]))]
    sigma=np.identity( len(d1[0])  )
    for t in range(len(d)):
        fig = plt.figure()
        ax = Axes3D(fig)
        ledge=-10
        redge=10
        side=0.2
        x = np.arange(ledge,redge,side)
        y = np.arange(ledge,redge,side)
        z = np.zeros((int((redge-ledge)/side),int((redge-ledge)/side)))
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                for k in range(len(d[t])):
                    tx=[x[i],y[j]]-d[t][k]
                    z[i][j]+=midpro1.getG(tx,u,sigma)
        x,y=np.meshgrid(x, y)
        ax.plot_surface(x,y,z, rstride=1, cstride=1, cmap=colors[t],alpha=0.3)
        plt.show()
    plt.close()
    
def parzen(train,test):
    sptr=np.shape(train)
    spte=np.shape(test)
    print(sptr,spte)
    miss=[0 for i in range(spte[0])]
    u=[0 for i in range(sptr[-1])]
    sigma=np.identity( sptr[-1]  )
    for i in range(spte[0]):#对于测试集的每个类别
        for j in range(spte[1]):#对于测试集每个类别的每个点
            maxtemp=0
            maxarg=-1
            for t in range(sptr[0]):#对于训练集每个类别
                parzentemp=0
                for k in range(sptr[1]):#对于训练集每个类别每个点
                    tx=test[i][j]-train[t][k]
                    parzentemp+=midpro1.getG(tx,u,sigma)
                if parzentemp>maxtemp:
                    maxtemp=parzentemp
                    maxarg=t
            if maxarg!=i:
                miss[i]+=1
#                print(miss,j)
    print("miss:",miss)
    print("total miss classficarion rate: ", sum(miss)/(spte[0]*spte[1]))

def myKnn(train,test,K):
    sptr=np.shape(train)
    spte=np.shape(test)
    ttrain=[]
    for i in train:
        for j in i:
            ttrain.append(j)
#    print(np.shape(ttrain))
    miss=[0 for i in range(sptr[0])]
    labels=[int(i/sptr[1]) for i in range(sptr[0]*sptr[1])]
    for i in range(spte[0]):
        for j in range(spte[1]):
            dif=np.tile( test[i][j],(sptr[0]*sptr[1],1))-ttrain
            sqrdif=dif**2
            sqrdif_sum=sqrdif.sum(axis=1)
            dis=sqrdif_sum**0.5
            sortdis=dis.argsort()
            count = {}
            for ti in range(K):
                voteLabel = labels[ sortdis[ti] ]
                count[ voteLabel ] = count.get(voteLabel, 0) + 1  
            maxCount = 0
            for key, value in count.items():
                if value > maxCount:
                    maxCount = value
                    argmax = key
            if argmax!=i:
                miss[i]+=1
#                print(miss,j)
    print("my knn miss:",miss," k:",K)
    print("my knn total miss classficarion rate: ", sum(miss)/(spte[0]*spte[1]))

def knnSCIPY(train,test,K):
    sptr=np.shape(train)
    spte=np.shape(test)
    X_train=[]
    y_train=[]
    miss=[0 for i in range(sptr[0])]
    for i in range(sptr[0]):
        for j in range(sptr[1]):
            X_train.append(train[i][j])
            y_train.append(i)
    knn_clf = KNeighborsClassifier(n_neighbors=K)
#    print(np.shape(X_train),np.shape(y_train))
    knn_clf.fit(X_train,y_train)
    for i in range(spte[0]):            
        result=knn_clf.predict(test[i])
        for j in range(len(result)):
            if result[j]!=i:
                miss[i]+=1
    print("SCIPY knn miss:",miss," k:",K)
    print("scipy total miss classficarion rate: ", sum(miss)/(spte[0]*spte[1]))
    
if __name__=="__main__":
    d1,d2,d3=midpro1.readData()
    train=[d1[0:1000],d2[0:1000],d3[0:1000]]
    test=[d1[1000:2000],d2[1000:2000],d3[1000:2000]]
#    plotP(train)
#    parzen(train,test)
    for k in range(1,11):
        knnSCIPY(train,test,k)
        myKnn(train,test,k)