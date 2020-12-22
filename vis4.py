import numpy as np
import matplotlib.pyplot as plt

from teaming import logger
X=[i*100/(10000*16) for i in range(101)]
#frqs=[1,10,100,1000,10000]
#frqs=[1,2,3,4,6,8,12,16]
frqs=[1,2,3,4,5,6,7,8]
letter="qq"
data=[]
err=[]

for frq in frqs:
    T=[]
    R=[]
    for i in range(8):
        log = logger.logger()
        log.load("tests/"+letter+str(frq)+"-"+str(i)+".pkl")

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
    std=np.std(T,axis=0)
    T=np.mean(T,axis=0)
    
    data.append(T[-1])
    err.append(std[-1]/8**(0.5))

    #plt.subplot(2,1,1)
    #plt.plot(R)
    #plt.subplot(2,1,2)
    #plt.plot(X,T)
#frqs=[10000,1000,100,10,1]
plt.errorbar(frqs,data,yerr=err)
#plt.semilogx()#frqs,data)

plt.ylim([0,0.9])
plt.grid(True)
plt.plot(X,[0.5]*101,"--")
plt.plot(X,[0.8]*101,"--")
'''
plt.legend(["Max single POI reward","Max reward","This method"])
plt.xlabel("Number of Shuffles")
plt.ylabel("Global Reward")
plt.title("Effect of Type Shuffle Frequency")
'''
plt.xlabel("Number of Types")
plt.ylabel("Global Reward")
plt.title("Effect of Number of Types")
plt.show()