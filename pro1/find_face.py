import pro1
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2

def getDetec():
    path="./project1-data-Detection/"
    im=[]
    RGB=[]
    for i in range(1,4):
        t=np.array(imageio.imread(path+str(i)+'.jpg'))
        t=imageio.imread(path+str(i)+'.jpg')
        RGB.append(t)
        t=cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
        im.append(t)
    return im,RGB

def printW(image,ipos,jpos,wshape,changeold=0):
    if changeold==0:
        im=np.copy(image)
    else:
        im=image
    wide=1
    c=255
    for i in range(ipos-wide,ipos+wide+1):
        for j in range(jpos-wide,jpos+wide+1):
            im[i:i+wshape[0],j]=c
            im[i:i+wshape[0],j+wshape[1]]=c
            im[i,j:j+wshape[1]]=c
            im[i+wshape[0],j:j+wshape[1]]=c
    if changeold==0:        
        imshow(im)
    
def imshow(im):
    plt.imshow(im,cmap='gray')
    plt.show()
    plt.close()
    
def pretest(test):
    di=np.shape(test)[0]
    dj=np.shape(test)[1]
    
    lefteye=np.sum(test[int(di/3*1):int(di/3*1)+int(di/6*1),int(dj/5*1):int(dj/5*1)+int(dj/5*1)])
#    printW(test,int(di/3*1),int(dj/5*1),[int(di/6*1),int(dj/5*1)],1)
    
    righteye=np.sum(test[int(di/3*1):int(di/3*1)+int(di/6*1),int(dj/5*3):int(dj/5*3)+int(dj/5*1)])
#    printW(test,int(di/3*1),int(dj/5*2),[int(di/6*1),int(dj/5*1)],1)
    
    mid=np.sum(test[int(di/3*1):int(di/3*1)+int(di/6*1),int(dj/5*2):int(dj/5*2)+int(dj/5*1)])
#    printW(test,int(di/3*1),int(dj/5*3),[int(di/6*1),int(dj/5*1)],0)
    t1=lefteye+righteye-mid
    
    eyes=np.sum(test[int(di/3*1):int(di/3*1)+int(di/6*1),int(dj/5*1):int(dj/5*4)])
    downeyes=np.sum(test[int(di/3*1)+int(di/6*1):int(di/3*2),int(dj/5*1):int(dj/5*4)])
    upeyes=np.sum(test[int(di/3*1)-int(di/6*1):int(di/3*2),int(dj/5*1):int(dj/5*4)])
    t2=2*eyes-downeyes-upeyes
    if t1<1000 and t2<0:
#        print(t1,t2)
#        imshow(test)
        return True
    return False
    
    
def to_one(d):
    data=np.copy(d)
    dmax=np.max(data)
    dmin=np.min(data)
    return (data-dmin)/(dmax-dmin)

def makesum(dim):
    im=np.copy(dim)
    for i in range(1,len(im)):
        im[i][0]+=im[i-1][0]
    for j in range(1,len(im[0])):
        im[0][j]+=im[0][j-1]
    for i in range(1,len(im)):
        for j in range(1,len(im[0])):
            im[i][j]+=im[i-1][j]+im[i][j-1]-im[i-1][j-1]
    return im

def detect(test,Wshape,meany,istep,jstep):
    i=0
    result=[]
    if test.shape[0]/3<Wshape[0]:
        edge=int(Wshape[0]*1.5)
    else:
        edge=int(test.shape[0]/3)
    while(i+Wshape[0]<edge):
        j=0
        while(j+Wshape[1]<test.shape[1]):
            window=np.array(test[i:i+Wshape[0],j:j+Wshape[1]])
            if pretest(to_one(window)):
                dis=np.sqrt(np.sum(np.square(window - meany)))
                result.append(([i,j,dis]))
            j=j+jstep
        i=i+istep
    result=sorted(result, key=lambda x:x[-1])
    return result

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    N=9
    K=10
    data,showims=pro1.getData([0 for i in range(11)],2)
    Wshape=showims[0][0].shape
    print("滑动窗口大小:",Wshape)
    train=[]
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            train.append(showims[i][j])
    meany=np.mean(train,axis=0)
    print("平均脸")
    print(np.sum(meany))
    imshow(meany)
    
    imD,RGBimD=getDetec()
    imD[0] = cv2.resize(imD[0],(int(np.shape(imD[0])[1]/3),int(np.shape(imD[0])[0]/3)))
    imD[1] = cv2.resize(imD[1],(int(np.shape(imD[1])[1]),int(np.shape(imD[1])[0])))
    imD[2] = cv2.resize(imD[2],(int(np.shape(imD[2])[1]*1.5),int(np.shape(imD[2])[0]*1.5)))

    istep=10
    jstep=10
    facenum=[1,3,7]
    for picnum in range(3):
        r=detect(imD[picnum],Wshape,meany,istep*(picnum+1),jstep*(picnum+1))
        for i in range(5*facenum[picnum]):
            printW(imD[picnum],r[i][0],r[i][1],Wshape,changeold=1)
        printW(imD[picnum],r[0][0],r[0][1],Wshape,changeold=0)
