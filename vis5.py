import numpy as np
import matplotlib.pyplot as plt

from teaming import logger

letter="c"
data=[]
err=[]
keyz=[]
for frq in range(8):
    T=[]
    R=[]
    
    for i in range(8):
        log = logger.logger()
        log.load("tests/"+letter+str(frq)+"-"+str(i)+".pkl")

        r=log.pull("reward")
        L=log.pull("loss")
        t=log.pull("test")
        
        vals=log.pull("poi vals")
        
        vals=sorted(vals[0])
        mx=vals[-1]#+vals[-2]
        
        #print(t)
        r=np.array(r)

        t=np.array(t)

        t=np.average(t,axis=1)
        R.append(r)
        T.append(t/mx*0.5)
    
    keyz.append(vals[-2])
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
d=[[d,z] for d,z in zip(data,keyz)]
print(d)
d=sorted(d,key=lambda x:x[1])
data=[i[0] for i in d]
plt.bar(range(8),data,yerr=err)
#plt.semilogx()#frqs,data)

plt.ylim([0,0.99])
plt.grid(True)
plt.plot(range(8),[0.5]*8,"--")
#plt.plot(range(8),[0.8]*8,"--")

#plt.legend(["Max single POI reward","Max reward","This method"])
plt.legend(["Max single POI reward","This method"])
plt.xlabel("Set Number")
plt.ylabel("Global Reward")
plt.title("Effect of POI Values (B)")
'''
#plt.legend(["Max reward","Received Reward"])
#plt.legend(["Max single POI reward","Received Reward"])
plt.legend(["Max single POI reward","Max reward","Received Reward"])
plt.xlabel("Set Number")
plt.ylabel("Global Reward")
plt.title("Effect of POI Positions")
'''
plt.show()