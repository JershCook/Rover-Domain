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


log = logger.logger()
log.load("tests/evo121-0.pkl")

p=log.pull("position")
t=log.pull("types")
tst=log.pull("test")
print(tst)

tst=np.array(tst)
Tst=np.average(tst,axis=1)
tst=tst[-1]
print(tst)


poi=log.pull("poi")[0]
print(poi)

idx=3
nagents=5
pos=p

for idx1 in range(50):
    plt.ion()
    plt.clf()
    plt.subplot(1,2,1)
    vals=np.array([0.1,0.1,0.5,0.3,0.0,0.0])*200+10
    plt.scatter(poi[:,0],poi[:,1],s=vals)
    typ=t[0][idx]
    print(typ)
    for i in range(nagents):
        data=[]
        for j in range(len(pos)):
            p=pos[j][idx]
            data.append(p[i])
        data=np.array(data).T
        x,y=data
        tt=typ[i]
        color=['k','r','b','m','y'][tt]
        plt.plot(x,y,color)
    
    plt.title(str(idx)+':'+str(tst[idx]))

    plt.subplot(1,2,2)
    plt.plot(Tst)
    plt.pause(1.0)
plt.show()