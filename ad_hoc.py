"""
An example using the rover domain gym-style interface and the standard, included CCEA learning algorithms.
This is a minimal example, showing the minimal Gym interface.
"""
import numpy as np
import sys
import multiprocessing as mp



from rover_domain_core_gym import RoverDomainGym
import code.ccea_2 as ccea
import code.agent_domain_2 as domain
import mods
from teaming.learner import learner
from sys import argv
import pickle
import tensorflow as tf


def make_env(progress):

    nagents=1

    sim = RoverDomainGym(nagents,100)
    #mods.recipePoi(sim)
    obs=sim.reset()
    #print(len(obs[0]))
    #for i in range(sim.data["Recipe Size" ]):
    #    sim.data["Item Held"][0][i]=progress[i]
    
    #obs=reduce_state(obs)

    sim.data["Coupling"]=1
    sim.data['Number of Agents']=nagents
    return sim

def test1(trial):
    env=make_env([1,1,1,1])
    team=[0,0,1,1,2,2,3,3]
    team=[0]
    with tf.Session() as sess:
        
        controller = learner(team,sess)
        init=tf.global_variables_initializer()
        sess.run(init)


        for i in range(10001):
            r=controller.run(env,i,0)# i%100 == -10)
            print(i,max(r))
            if i%1000==0 and 0:
                controller.test(env)
            if i%1000==0:
                controller.save("logs/"+str(trial)+"t.pkl")
            #print(r)
'''
test1(0)
for i in range(0):
    p=mp.Process(target=test1,args=(i,))
    p.start()
    #p.join()
'''
env=make_env(None)

from time import sleep

s=env.reset()
s=s[:,4:][0]
for i in range(100):
    
    idx=1
    loc=env.data["Poi Positions"][idx]
    ang=env.data["Agent Orientations"][0]
    pos=env.data["Agent Positions"][0]

    heading=[loc[0]-pos[0],loc[1]-pos[1]]

    trn=np.arccos( (heading[0]*ang[0]+heading[1]*ang[1])/( np.sqrt(heading[0]**2+heading[1]**2))* np.sqrt(ang[0]**2+ang[1]**2)  )    
    trn/=4
    spd=1.0

    a=[spd,trn]
    
    s,r,_,_=env.step([a])
    s=s[:,4:][0]
    print(i,r,trn,spd)
    env.render()
    sleep(0.033)

        
            
        
        
        
        
        
