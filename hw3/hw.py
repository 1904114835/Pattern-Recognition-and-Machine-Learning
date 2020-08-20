import numpy as np
import cvxopt
import matplotlib.pyplot as plt
from cvxopt import matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def printDataAndHy(data,z,alpha=None,w=None,w0=None):
    if np.shape(data)[-1]!=2:
        print("not 2D plot")
        return
    fig = plt.figure()  
    ax1 = fig.add_subplot(111)
    ax1.set_title(' + : support vector')
    for i in range(len(data)):
        x=data[i][0]
        y=data[i][1]
        if z[i]==-1:
            c='r'
        else:
            c='b'
        if type(alpha)==type(None):
            ax1.scatter(x,y,color=c,marker = 'o') 
        elif alpha[i]<np.mean(alpha):
            ax1.scatter(x,y,color=c,marker = 'o') 
        else:
            ax1.scatter(x,y,s=400.,color=c,marker = '+')
    if type(alpha)!=type(None):
        wx=[]
        wy=[]
        l=np.min(data[:,0])
        r=np.max(data[:,0])
        for i in range(int(l),int(r)):
            wx.append(i)
            wy.append(-(w0+i*w[0][0])/w[0][1] )
        plt.plot(wx,wy)
    plt.show()
    plt.close()

def toy():
    x=np.array([ [1,6],
        [1,10],
        [4,11],
        [5,2],
        [7,6],
        [10,4]])
    z=np.array([[1],[1],[1],[-1],[-1],[-1]])   
    H=np.transpose(np.dot(x,x.T)*np.dot(z,z.T))
    H=(H+np.eye(6)*0.002+H.T)/2
    B=z
    A=-np.eye(6)
    H=matrix(H,(6,6),'d')
    f=cvxopt.matrix([-1 for i in range(6)],(6,1),'d')
    A=cvxopt.matrix(A,(6,6),'d')
    a=cvxopt.matrix([0 for i in range(6)],(6,1),'d')
    B=cvxopt.matrix(B,(1,6),'d')
    b=cvxopt.matrix([0],(1,1),'d')
    alpha= cvxopt.solvers.qp(H,f,A,a,B,b)
    alpha= np.round(np.array(alpha['x']),3)
    print("alpha\n",alpha)
    w=np.dot((alpha*z).T,x)
    print("w",w)
    w0=(1/z[0]-np.dot(w,x[0].T))
    print("w0",w0)
    printDataAndHy(x,z,alpha,w,w0)

def readData(path):
    data=[]
    z=[]
    with open(path, "r") as f:
        lines = f.readlines()
        f.close()
        for line in lines:
            t=list(map(float,line.strip('\n').split()))
            data.append([t[i] for i in range(len(t)-1)])
            z.append([t[-1]])
    return np.array(data),np.array(z)

def trainData1(path):
    x,z=readData(path)
    H=np.transpose(np.dot(x,x.T)*np.dot(z,z.T))
    nums=len(x)
    H=(H+np.eye(nums)*0.001+H.T)/2
    B=(np.eye(nums))*z
    B=z
    A=-np.eye(nums)
    H=matrix(H,(nums,nums),'d')
    f=matrix([-1 for i in range(nums)],(nums,1),'d')
    A=matrix(A,(nums,nums),'d')
    a=matrix([0 for i in range(nums)],(nums,1),'d')
    B=matrix(B,(1,nums),'d')
    b=matrix([0],(1,1),'d')
    alpha= cvxopt.solvers.qp(H,f,A,a,B,b)
    alpha= np.array(alpha['x'])
    w=np.dot((alpha*z).T,x)
    print("alpha\n",alpha)
    w=np.dot((alpha*z).T,x)
    print("w",w)
    for i in range(len(alpha)):
        if alpha[i]>np.mean(alpha):
            w0=1/z[i] - np.dot(w,x[i].T)
    print("w0",w0)
    printDataAndHy(x,z,alpha,w,w0)
    
def printDataAndSupportVector(data,z,alpha):
    if np.shape(data)[-1]!=2:
        print("not 2D plot")
        return
    fig = plt.figure()  
    ax1 = fig.add_subplot(111)
    ax1.set_title(' + : support vector')
    for i in range(len(data)):
        x=data[i][0]
        y=data[i][1]
        if z[i]==-1:
            c='r'
        else:
            c='b'
        if type(alpha)==type(None):
            ax1.scatter(x,y,color=c,marker = 'o') 
        elif alpha[i]<np.mean(alpha):
            ax1.scatter(x,y,color=c,marker = 'o') 
        else:
            ax1.scatter(x,y,s=400.,color=c,marker = '+')
            
class mySVM():
    def __init__(self):
        pass
    def fit(self,x,z):
        H=np.transpose(np.dot(x,x.T)*np.dot(z,z.T))
        nums=len(x)
        H=(H+np.eye(nums)*0.001+H.T)/2
        B=(np.eye(nums))*z
        B=z
        A=-np.eye(nums)
        H=matrix(H,(nums,nums),'d')
        f=matrix([-1 for i in range(nums)],(nums,1),'d')
        A=matrix(A,(nums,nums),'d')
        a=matrix([0 for i in range(nums)],(nums,1),'d')
        B=matrix(B,(1,nums),'d')
        b=matrix([0],(1,1),'d')
        alpha= cvxopt.solvers.qp(H,f,A,a,B,b)
        alpha= np.array(alpha['x'])
        w=np.dot((alpha*z).T,x)
        w=np.dot((alpha*z).T,x)
        for i in range(len(alpha)):
            if alpha[i]>np.mean(alpha):
                w0=1/z[i] - np.dot(w,x[i].T)
        self.w=w.reshape(-1)
        self.b=w0[0]
        self.alpha=alpha
#        print(self.w)
#        print(self.b)
#        print(alpha)
        
    def predict(self,x):
        if np.shape(x)!=np.ndarray:
            x=np.array(x)
        if np.shape(x)==(len(x[0]),):
            if np.dot(self.w,x)+self.b>0:
                return 1
            else:
                return -1
        else:
            result=[]
            for i in x:
                if np.dot(self.w,i)+self.b>0:
                    result.append(1)
                else:
                    result.append(-1)
            return np.array(result)
            
    def getKernel(self,x):
        if np.shape(x)!=np.ndarray:
            x=np.array(x)
        return np.array([x[:,0]**2,x[:,1]**2,x[:,0]*x[:,1],x[:,0],x[:,1]]).T

    def plotSVM(self,l,r,d,u,gap):
        data=[]
        l,r,d,u=int(l),int(r),int(d),int(u)
        for i in range(l,r,int(gap/2)):
            for j in range(d,u,gap):
                data.append([i,j])
        data=np.array(data)
        z=self.predict(self.getKernel(data))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for i in range(len(data)):
            x=data[i][0]
            y=data[i][1]
            if z[i]==-1:
                c='r'
            else:
                c='b'
            ax1.scatter(x,y,color=c,marker = 'o')
        plt.show()
        plt.close()
        
def trainData2SK(path):
    x,z=readData(path)
    mysvm=mySVM()
    newx=mysvm.getKernel(x)
    mysvm.fit(newx,z)
    printDataAndSupportVector(x,z,mysvm.alpha)
    mysvm.plotSVM(-50,60,-70,80,2)
    
if __name__=="__main__":
    toy()
    trainData1("TrainSet1.txt")
    trainData2SK("TrainSet2.txt")
    
    
    