import numpy as np 
import matplotlib.pyplot as plt 

plt.ion()
while True:
    plt.clf()
    data=np.genfromtxt("save/0.txt",delimiter=',')
    print(data.shape)
    r=data[:,1]
    r_=data[:,4]
    e=data[1:,2]

    r1=r[::2]
    r2=r[1::2]

    plt.subplot(411)
    plt.plot(r1)

    plt.subplot(412)
    plt.plot(r2)


    plt.subplot(413)
    plt.plot(r_)
    plt.subplot(414)
    plt.plot(np.log(e))
    plt.pause(10.0)