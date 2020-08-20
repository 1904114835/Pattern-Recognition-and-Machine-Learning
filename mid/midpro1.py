import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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
        plt.scatter(data2[:,0],data2[:,1], marker = 'o', color = 'b', label='2',s=20,alpha = 0.5)
        plt.legend()
    if len(data3)!=0:
        plt.scatter(data3[:,0],data3[:,1], marker = '+', color = 'black', label='3',s=20,alpha = 0.5)
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
    
def mle(x):
    #最大似然
    u=np.array([np.mean(x[:,ti]) for ti in range(np.shape(x)[1])])
    return u, (np.dot((x - u).T,(x - u) / x.shape[0]))

def getG(x,u,sigma):
    if len(x)!=len(u) or len(x)!=np.shape(sigma)[0]:
        print("wrong")
        exit(0)
    t1=(1/(2*np.pi))*(1/np.sqrt(np.linalg.det(sigma)))
    t2=np.exp(-0.5* np.dot(np.dot(np.array(x-u).T,np.linalg.inv(sigma)),np.array(x-u)))
    return t1*t2

def plotG(uall,sigmaall):
    colors=['Reds','Blues','binary']
    if len(uall)!=len(sigmaall):
        print("erro:u is not as len as sigma")
        return
    else:
        for t in range(len(uall)):
            fig = plt.figure()
            ax = Axes3D(fig)
            u=uall[t]
            sigma=sigmaall[t] 
            x = np.arange(-10,10,0.2)
            y = np.arange(-10,10,0.2)
            z = np.zeros((100,100))
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    tx=[x[i],y[j]]
                    z[i][j]=getG(tx,u,sigma)
            x,y=np.meshgrid(x, y)
            ax.plot_surface(x,y,z, rstride=1, cstride=1, cmap=colors[t],alpha=0.3)
            plt.show()
    plt.close()
    
def getMAP(t,u,sigma):
    st=np.shape(t)
    maxargz=np.zeros((st[0],st[1]))
    for i in range(st[0]):
       for j in range(st[1]):
            x=t[i][j][0]
            y=t[i][j][1]
            maxt=-1
            tempmaxarg=-1
            for k in range(len(u)):    
                tx=[x,y]
                temp=getG(tx,u[k],sigma[k])
                if temp>maxt:
                    maxt=temp
                    tempmaxarg=k
            maxargz[i][j]=tempmaxarg
    miss=[0 for i in range(st[0])]
    for i in range(st[0]):
        for j in range(st[1]):
            if maxargz[i][j]!=i:
                miss[i]+=1
    print("miss simple num: ",miss)
    print("total miss classficarion rate: ", sum(miss)/(st[0]*st[1]))
if __name__=="__main__":
    d1,d2,d3=readData()
    printData(d1[0:1000],d2[0:1000],d3[0:1000])
    u1,sigma1=mle(d1[0:1000])
    u2,sigma2=mle(d2[0:1000])
    u3,sigma3=mle(d3[0:1000])
    u=[u1,u2,u3]
    sigma=[sigma1,sigma2,sigma3]
    for i in range(3):
        print("u",i,":",u[i])
        print("sigma",i,":\n",end="")
        for k in sigma[i]:
            print(k)
    plotG(u,sigma)
            
    getMAP([d1[1000:2000],d2[1000:2000],d3[1000:2000]],u,sigma)
    
    hu1,hs1=mle(d1[0:500])
    hu2,hs2=mle(d2[0:500])
    hu3,hs3=mle(d3[0:500])
    hu=[hu1,hu2,hu3]
    hs=[hs1,hs2,hs3]
    for i in range(3):
        print("u",i,":",hu[i])
        print("sigma",i,":\n",end="")
        for k in hs[i]:
            print(k)
    plotG(hu,hs)
    getMAP([d1[500:2000],d2[500:2000],d3[500:2000]],hu,hs)
    