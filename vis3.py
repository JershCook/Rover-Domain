import numpy as np
import matplotlib.pyplot as plt

from teaming import logger
X=[i*100 for i in range(101)]
n1="202"
n="16"
for q in ["jj"+n1,"v"+n,"r"+n]:
    T=[]
    R=[]
    for i in range(8):
        log = logger.logger()
        if q[0]=="j":
            log.load("tests/"+q+'-'+str(i)+".pkl")
        else:
            log.load("logs/"+str(i)+q+".pkl")
        r=log.pull("reward")
        L=log.pull("loss")
        t=log.pull("test")
        #print(t)
        r=np.array(r)

        t=np.array(t)

        t=np.average(t,axis=1)
        R.append(r)
        T.append(t)

    R=np.mean(R,axis=0)
    std=np.std(T,axis=0)/np.sqrt(8)
    T=np.mean(T,axis=0)

    #plt.subplot(2,1,1)
    #plt.plot(R)
    #plt.subplot(2,1,2)
    plt.plot(X,T)
    plt.fill_between(X,T-std,T+std,alpha=0.4)

    plt.ylim([0,0.9])
    plt.grid(True)
plt.plot(X,[0.5]*101,"--")
plt.plot(X,[0.8]*101,"--")
plt.legend(["Random Teaming + Types","Unique Learners","Types Only","Max single POI reward","Max reward"])
plt.xlabel("Episode")
plt.ylabel("Global Reward")
if n=="4":
    plt.title("5 agents, coupling req. of 2")
if n=="8":
    plt.title("8 agents, coupling req. of 3")
if n=="16":
    plt.title("16 agents, coupling req. of 6")
plt.show()