import numpy as np
import matplotlib.pyplot as plt
import mpltern
from mpltern.ternary.datasets import get_shanon_entropies
t, l, r, v = get_shanon_entropies()
print (t)
from teaming import logger

data=[]
err=[]
def rotate(v,n):
    idxs=[[] for i in range(n)]
    count=0
    for i in range(n):
        for j in range(i,n):
            idxs[j].append(count)
            count+=1
    idxs.reverse()
    idxs=[i for idx in idxs for i in idx]
    #idxs=np.array(idxs)
    return v[idxs]

def flip(v,n):
    idxs=[[] for i in range(n)]
    count=0
    for i in range(n):
        for j in range(n-i):
            idxs[i].append(count)
            count+=1
    for i in range(n):
        idxs[i].reverse()
    idxs=[i for idx in idxs for i in idx]
    #idxs=np.array(idxs)
    print(idxs)
    return v[idxs]
offset=0
for i in range(4):
    log = logger.logger()
    #log.load("tests/jj100-7.pkl")
    #log.load("tests/jj101-"+str(i)+".pkl")
    log.load("tests/evo121-"+str(i+offset)+".pkl")
    t=log.pull("types")
    tst=log.pull("test")
    #print(tst)

    tst=np.array(tst)


    nagents=len(t[0][0])

    x=[]
    y=[]
    for idx in range(len(t[0])):

        typ=t[0][idx]
        #print(typ)
        count=np.bincount(typ,minlength=3)
        count=count.astype(float)
        x.append(count/nagents)
        y.append(tst[-1][idx])

    y=np.array(y)

    if 0:
        vals=log.pull("poi vals")[0]
        print(vals)
        vals=sorted(vals,reverse=True)
        y/=(vals[0]+vals[1])
    else:
        #y/=0.8
        pass
    #rint(x)

    x=np.array(x).T
    t,l,r=x
    vt,vl,vr=np.sum(y*t),np.sum(y*l),np.sum(y*r)
    print(vt,vl,vr)
    if vl==max(vt,vl,vr):
        y=rotate(y,nagents+1)
        y=rotate(y,nagents+1)
    if vr==max(vt,vl,vr):
        y=rotate(y,nagents+1)

    
    vt,vl,vr=np.sum(y*t),np.sum(y*l),np.sum(y*r)
    if vl<vr:
        y=flip(y,nagents+1)
    vt,vl,vr=np.sum(y*t),np.sum(y*l),np.sum(y*r)
    print(vt,vl,vr)
    data.append(y)
    
y=np.mean(data,axis=0)
#y=np.linspace(0,1,num=len(y))
#y=rotate(y,nagents+1)
v=y
fig=plt.figure()
ax = plt.subplot(projection='ternary')

position="tick1"

ax.set_tlabel('Percent Agent A')
ax.set_llabel('Percent Agent B')
ax.set_rlabel('Percent Agent C')

ax.taxis.set_label_position(position)
ax.laxis.set_label_position(position)
ax.raxis.set_label_position(position)


cs=ax.scatter(t, l, r, c=v,s=4000)#,vmin=0,vmax=0.8)
#print(t)
#print(v)
cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
colorbar = fig.colorbar(cs, cax=cax)
colorbar.ax.set_ylabel('Global Reward')
plt.title("Random POI Positions and Values\n")
plt.title("Fixed POI Positions and Values\n")
plt.grid()
plt.show()

