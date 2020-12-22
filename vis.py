import numpy as np
import matplotlib.pyplot as plt

from teaming import logger
for i in range(8):
    log = logger.logger()
    log.load("logs/"+str(i)+"t3.pkl")

    r=log.pull("reward")
    L=log.pull("loss")
    t=log.pull("test")
    print(t)
    r=np.array(r)
    L=np.array(L)

    loss=[]
    for l in L:
        l=np.array(l)
        loss.append(np.mean(l,axis=0))
    loss=np.array(loss).T
    plt.subplot(4,1,1)
    #R=np.sum(r,axis=1)
    R=r
    plt.plot(R)


    plt.subplot(4,1,2)
    #R=R[:,-1]#np.max(r,axis=1)

    plt.plot(R)

    plt.subplot(4,1,3)
    t=np.array(t)
    #t=np.sum(t,axis=2)
    t=np.average(t,axis=1)
    print(t.shape)
    plt.plot(t)
    plt.subplot(4,1,4)
    N=5
    print(len(loss[0]))
    for l in loss:
        
        l=np.convolve(l,np.ones(N)/float(N),mode="valid")
        plt.semilogy(l)
plt.show()