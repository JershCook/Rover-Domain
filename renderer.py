

import numpy as np
import matplotlib
import pickle
from rover_domain_core_gym import RoverDomainGym

import matplotlib.pyplot as plt

imageIndex=0

nagents=15
sim = RoverDomainGym(nagents,100)

def render(data):
    global imageIndex
    imageIndex+=1
    scale=.5
   
    plt.ion()
        
    plt.clf()
    
    plt.title("Episode Index "+str(data["Episode Index"]))
    
    plt.xlim(-data["World Width"]*scale,data["World Width"]*(1.0+scale))
    plt.ylim(-data["World Length"]*scale,data["World Length"]*(1.0+scale))
    
    plt.scatter(data["Agent Positions"][:,0],data["Agent Positions"][:,1])
    
    
    if ("Number of POI Types" in data):
        
        ntypes=data["Number of POI Types"]
        xpoints=[[] for i in range(ntypes)]
        ypoints=[[] for i in range(ntypes)]
        for i in range(len(data["Poi Positions"])):
            xpoints[i%ntypes].append(data["Poi Positions"][i,0])
            ypoints[i%ntypes].append(data["Poi Positions"][i,1])
        for i in range(ntypes):
            plt.scatter(xpoints[i],ypoints[i],label=str(i))
        plt.legend()
    
    else:
        print("Single")
             
    
    #plt.savefig("ims/test"+str(imageIndex)+".png")
    plt.draw()
    plt.pause(1.0/30.0)


with open("save/0.pkl" , 'rb') as f:
    data=pickle.load(f)

for i in range(100):
    plt.ion()
    plt.clf()
    plt.scatter(data[i,:,0],data[i,:,1])
    pois=sim.data["Poi Positions"]
    npois=sim.data["Number of POIs"]
    plt.scatter(pois[:,0],pois[:,1])
    plt.show()
    plt.xlim((0,50))
    plt.ylim((0,50))
    plt.pause(1.0/30.0)
