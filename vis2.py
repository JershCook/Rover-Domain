import numpy as np
import matplotlib.pyplot as plt

from teaming import logger

log = logger.logger()
log.load("logs/0t.pkl")

r=log.pull("reward")
L=log.pull("loss")

R=[max(i) for i in r]
idx=np.argmax(R)
idx=200
print(max(R))
poi=log.pull("poi")
locs=log.pull("position")[idx]
plt.ion()
print(locs)
while True:
    for l,r_ in zip(locs,r[idx]):
        
        plt.clf()
        plt.title("reward: "+str(round(r_,3)))
        plt.scatter(l.T[0],l.T[1])
        plt.scatter(poi.T[0],poi.T[1])
        plt.xlim([0,50])
        plt.ylim([0,50])
        plt.pause(.01)
    plt.pause(2.0)