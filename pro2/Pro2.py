import xml.dom.minidom
import numpy as np
import matplotlib.pyplot as plt
import copy
#打开xml文档
def getData(name):
    doc = xml.dom.minidom.parse(name)
    pos = []
    examples = doc.getElementsByTagName('trainingExample')
    for i in range(len(examples)): 
        pos.append([])
        for p in examples[i].getElementsByTagName('coord'):
            px = np.float(eval(p.getAttribute('x')))
            py = np.float(eval(p.getAttribute('y')))
#            t  = np.float(eval(p.getAttribute('t')))
            pos[i].append([px,py])
        pos[i]=np.array(pos[i])
    return pos

def makeOnes(d):
    newd=copy.deepcopy(d)
    for k in d.keys():
        for i in range(len(d[k])):
             for t in range(2):            
                dmax=d[k][i][:,t].max()
                dmin=d[k][i][:,t].min()
                newd[k][i][:,t]=(d[k][i][:,t]-dmin)/(dmax-dmin)
    return newd

def makeTotal(d):
    t=[]
    num=[[0,0] for i in range(5)]
    temp=0
    r=0
    for i in d.keys():
        num[r][0]=temp
        for j in d[i]:
            for k in j:
                t.append(k)
                temp+=1
        num[r][1]=temp
        r+=1
    return np.array(t),num
    
def printData(data,onelist=0):
    if onelist==1:
        plt.scatter(data[:,0],data[:,1], marker = 'o', color = 'b')
    else:
        for i in data:
            plt.scatter(i[:,0],i[:,1], marker = 'o', color = 'b')
    plt.show()
    plt.close()
    
class kmeans:
    def __init__(self,n_clusters):
        self.n=n_clusters    
        self.INF=9999999999
        
    def fit(self,data):
        self.fit_predict(data)
        
    def fit_predict(self,data):
        num_data=np.shape(data)[0]
        center =[data[np.random.randint(0,num_data)]]
        dist_all=np.zeros(num_data)+self.INF
        for j in range(self.n-1):
            for i in range(num_data):
                dist_all[i]=min(np.sum((center[j]-data[i])**2),dist_all[i])
            next_idx = dist_all.argmax()
            center = np.vstack([center, data[next_idx]])
        result = np.zeros((num_data))-1
        flag_change = True
        while flag_change:
            flag_change = False
            for i in range(num_data):
                c = np.argmin(np.sum((center-data[i])**2,axis=1))
                if result[i] != c:
                    result[i] = c
                    flag_change = True
            for j in range(self.n):
                center[j]= data[result[:] == j].mean(axis=0)
        self.cent=center
        return result.astype("int")
    
    def predict(self,data):
        result=np.zeros((len(data)))
        for i in range(len(data)):
            result[i] = np.argmin(np.sum((self.cent-data[i])**2,axis=1))
        return result.astype(np.int)
        
    def show(self,x,y,c):
        plt.scatter(x,y,c=c)
        plt.show()
        plt.close()
'''        
'''    
class HMM: 
    def __init__(self, state_num, ob_num, A=[], B=[], pi=[]):
        if len(A)==0:
            A=np.random.rand(state_num,state_num)
            B=np.random.rand(state_num,ob_num)
            pi=np.random.rand(state_num)
            for i in range(state_num):
                A[i]/=np.sum(A[i])
                B[i]/=np.sum(B[i])
            pi=pi/np.sum(pi)
        self.A = A
        self.B = B
        self.pi = pi
        self.states=state_num
        self.ob_num=ob_num
        np.set_printoptions(suppress=True)
        np.set_printoptions(linewidth=400)
    
    def bw_train(self, observations, outline=0.1,eps=20):
        n_samples = len(observations)
        for episode in range(eps):
            alpha,ct=self.scaling_forward(observations)
            beta=self.scaling_backward(observations,ct)
            sigma = np.zeros((self.states,self.states,n_samples-1))
            for t in range(n_samples-1):
                down = np.dot(np.dot(alpha[:,t].T, self.A) * self.B[:,observations[t+1]].T, beta[:,t+1])
                for i in range(self.states):
                    up = alpha[i,t] * self.A[i,:] * self.B[:,observations[t+1]].T * beta[:,t+1].T
                    sigma[i,:,t] = up / down
            prod =  (alpha[:,n_samples-1] * beta[:,n_samples-1]).reshape((-1,1))
            gamma = np.hstack((np.sum(sigma,axis=1),  prod / np.sum(prod)))
            sum_gamma = np.sum(gamma,axis=1)
            self.pi = gamma[:,0]
            self.A = np.sum(sigma,2) / np.sum(gamma[:,:-1],axis=1).reshape((-1,1))
            for lev in range(self.ob_num):
                flag = [ True if observations[i]==lev else False for i in range(len(observations))]
                self.B[:,lev] = np.sum(gamma[:,flag],axis=1) / sum_gamma
#        print("A",self.A)
#        print("B",self.B)
#        print("pi",self.pi)
            
    def scaling_forward(self, observations):
        n_samples = len(observations)
        alpha = np.zeros((self.states,n_samples))
        ct=np.zeros(n_samples)
        alpha[:,0] = self.pi * self.B[:, observations[0]]
        if alpha[:,0].sum() == 0:
            alpha[:,0]+=1e-10
        ct[0]=1/alpha[:,0].sum()
        alpha[:,0]=alpha[:,0]*ct[0]
        for t in range(1, n_samples):
            for n in range(self.states):
                alpha[n][t]=np.dot(alpha[:,t-1], (self.A[:,n])) * self.B[n, observations[t]]
                if alpha[n,t] == 0 or alpha[n,t] is np.nan:
                    alpha[n,t]=1e-10
            ct[t]=1/(alpha[:,t].sum())
            alpha[:,t]=alpha[:,t]*ct[t]
        return alpha,ct
    
    def scaling_backward(self, ob, ct):
        n_samples = len(ob)
        beta = np.ones((self.states,n_samples))
        for t in reversed(range(n_samples-1)):
            for n in range(self.states):
                beta[n,t] = np.sum(beta[:,t+1] * self.A[n,:] * self.B[:, ob[t+1]])
                if beta[n,t] is np.nan or beta[n,t]==0:
                    beta[n,t]=1e-10
                beta[n,t]=beta[n,t]*ct[t]
        return beta
    
    def prob(self,ob):
        alpha,ct=self.scaling_forward(ob)
        logp=0
        for t in range(len(ob)):
            logp+=np.log(ct[t])
        return np.exp(-logp)
    
def totaltrain(d,ob_num,state_num,gap):
    #ob和state越大，准确率越高
    #因为状态越多能够表现的信息就越多，各个类别的差异就越大
    trainkm={}
    for key in d.keys():
        trainkm[key]=[]
        for i in range(len(d[key])):
            if i%2==0:
                trainkm[key].append(d[key][i])
    trainkmlist,num=makeTotal(trainkm)
    mykm=kmeans(n_clusters=ob_num)
    c=mykm.fit_predict(trainkmlist)
    mykm.show(trainkmlist[:,0],trainkmlist[:,1],c)
    for i in range(len(num)):
        mykm.show(trainkmlist[num[i][0]:num[i][1],0],trainkmlist[num[i][0]:num[i][1],1],c[num[i][0]:num[i][1]])
    train={}
    test={}
    #构造训练测试序列
#    #奇数训练，偶数测试，即偶数下标训练，奇数下标测试
    for key in d.keys():
        train[key]=[]
        test[key]=[]
        for i in range(len(d[key])):
            if i%2==0:
                for t in mykm.predict(d[key][i]):
                    train[key].append(t)
            else:
                test[key].append(mykm.predict(d[key][i]).astype(np.int))
        train[key]=np.array(train[key])
                
    hmm={}
    for key in d.keys():
        hmm[key]=HMM(state_num,ob_num)
        hmm[key].bw_train(train[key][::gap])
    print("total train")
    t=0
    score=np.zeros((5,5))
    for key in d.keys():
        for i in range(len(test[key])):
            testl=[]
            for keyhmm in hmm.keys():
                testl.append(hmm[keyhmm].prob(test[key][i][::gap]))
            score[t][np.argmax(testl)]+=1
        print(key,score[t][t]/(len(test[key])))
        t+=1
    print(score)
    del hmm
    return score

def onesTrain(ob_num,state_num,d,gap=6):
    dones=makeOnes(d)
#    dones=d
    train={}
    test={}
    for key in dones.keys():        
        train[key]=[]
        test[key]=[]
        for i in range(len(dones[key])):
            if i%2==0:
                for j in dones[key][i]:
                    train[key].append(j)
            else:
                test[key].append(dones[key][i])
        train[key]=np.array(train[key])
    mykm={}
    for key in d.keys():
        mykm[key]=kmeans(ob_num)
        y=mykm[key].fit_predict(train[key])
        mykm[key].show(train[key][:,0],train[key][:,1],y)
    
    hmm={}
    for key in d.keys():
        hmm[key]=HMM(state_num, ob_num)
        hmm[key].bw_train(mykm[key].predict(train[key])[::gap],0.001)
    
    t=0
    print("ones train")
    score=np.zeros((5,5))
    for key in d.keys():
        for i in range(len(test[key])):
            testl=[]
            for keyhmm in hmm.keys():
                temp_test=mykm[keyhmm].predict(test[key][i])
                testl.append(hmm[keyhmm].prob(temp_test[::gap]))
            score[t][np.argmax(testl)]+=1
        print(key,score[t][t]/(len(test[key])))
        t+=1
    print(score)
    hmm.clear()
    mykm.clear()
    return score
    
if __name__=="__main__":
    d={}
    d["a"]=getData("a.xml")
    d["e"]=getData("e.xml")
    d["i"]=getData("i.xml")
    d["o"]=getData("o.xml")
    d["u"]=getData("u.xml")
    
    ob_num=6
    state_num=8
    gap=4#控制每隔多少个点取样，过多的取样会导致过拟合和噪点
    s=[]
#    for i in range(10):
    s.append(totaltrain(d,ob_num,state_num,gap))
#    for i in range(len(s)):
#        t=0
#        for j in range(len(s[i])):
#             
#            t+=s[i][j][j]/20
#        print("Correct classification of aeiou:", t/5)
    
    
    
#    为单个字母进行k-means聚类并产生序列，效果较差
#    ob_num=6
#    state_num=10
#    gap=6
#    s=[]
#    for i in range(10):
#        s.append(onesTrain(ob_num,state_num,d,gap))
#    for i in range(len(s)):
#        t=0
#        for j in range(len(s[i])):
#            t+=s[i][j][j]/20
#        print("Correct classification of a e i o u:", t/5)