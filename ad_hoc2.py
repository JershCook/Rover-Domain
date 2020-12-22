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
from teaming.learner3 import learner
from sys import argv
import pickle
import tensorflow as tf

def rand_loc(n):
    x,y=np.random.random(2)
    pos=[[x,y]]
    while len(pos)<6:
        X,Y=np.random.random(2)
        for x,y in pos:
            dist=((X-x)**2.0+(Y-y)**2.0 )**0.5
            if dist<0.2:
                X=None 
                break
        if not X is None: 
            pos.append([X,Y])
    
    return np.array(pos)


#print(vals)
def make_env(team):
    vals =np.array([0.1, 0.1, 0.5,0.3, 0.0, 0.0])
    
    pos=np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.5],
            [0.0, 0.5],
            [1.0, 0.0]
        ])

    #pos=rand_loc(6)#np.random.random((6,2))
    #vals=np.random.random(6)/2.0
    print(vals)
    nagents=len(team)

    sim = RoverDomainGym(nagents,100,pos,vals)
    #mods.recipePoi(sim)
    obs=sim.reset()
    #print(len(obs[0]))
    #for i in range(sim.data["Recipe Size" ]):
    #    sim.data["Item Held"][0][i]=progress[i]
    
    #obs=reduce_state(obs)

    sim.data["Coupling"]=2
    sim.data['Number of Agents']=nagents
    return sim


import time

def test1(trial,frq):
    frq=1
    #print(np.random.get_state())[1]    
    np.random.seed(int(time.time()*100000)%100000)
    team=[0,0,1,1,2,2,3,3]
    team=[0,1,2,2,1]
    
    #team=[0,1,2,2,1,2,0,0]
    #team=team+team
    env=make_env(team)
    with tf.compat.v1.Session() as sess:
        
        controller = learner(team,sess,env)
        init=tf.compat.v1.global_variables_initializer()
        sess.run(init)


        for i in range(10001):

            if i%int(frq)==0:
                controller.randomize()

            r=controller.run(env,i,0)# i%100 == -10)
            if i%10==0:
                print(i,r[-1],controller.team)
            if i%50==0 and 1:
                controller.test(env)
                
            if i%50==0:
                #controller.save("tests/q"+str(frq)+"-"+str(trial)+".pkl")
                #controller.save("logs/"+str(trial)+"r"+str(16)+".pkl")
                #controller.save("tests/jj"+str(121)+"-"+str(trial)+".pkl")
                controller.save("tests/evo"+str(121)+"-"+str(trial)+".pkl")
            #print(r)
            #print(r)


def test2(trial,f):
    #print(np.random.get_state())[1]    
    np.random.seed(int(time.time()*100000)%100000)
    team=[i for i in range(16)]
    #team=[0,1,2,3,4]
    env=make_env(team)
    with tf.compat.v1Session() as sess:
        
        controller = learner(team,sess)
        init=tf.compat.v1.global_variables_initializer()
        sess.run(init)


        for i in range(10001):
            r=controller.run(env,i,0)# i%100 == -10)
            if i%10==0:
                print(i,r[-1],controller.team)
            if i%100==0 and 1:
                controller.test(env)
            if i%1000==0:
                controller.save("logs/"+str(trial)+"v16.pkl")
            #print(r)


def test3(trial,f):
    #print(np.random.get_state())[1]    
    np.random.seed(int(time.time()*100000)%100000)
    team=[0,1,2,0,0,0,0,0]
    team=team+team
    #team=[0,1,2,0,0]
    env=make_env(team)
    with tf.compat.v1.Session() as sess:
        
        controller = learner(team,sess)
        init=tf.compat.v1.global_variables_initializer()
        sess.run(init)

        controller.randomize()
        for i in range(10001):
            r=controller.run(env,i,0)# i%100 == -10)
            if i%10==0:
                print(i,r[-1],controller.team)
            if i%100==0 and 1:
                controller.test(env)
            if i%1000==0:
                controller.save("logs/"+str(trial)+"r8.pkl")
            #print(r)

def test4(trial,frq):
    #print(np.random.get_state())[1]    
    np.random.seed(int(time.time()*100000)%100000)
    team=[0,0,1,1,2,2,3,3]
    team=[0,1,2,2,1,1,0,0]
    team=team+team
    team=np.array([i%int(frq) for i in range(8)])
    env=make_env(team)
    with tf.compat.v1.Session() as sess:
        
        controller = learner(team,sess)
        init=tf.compat.v1.global_variables_initializer()
        sess.run(init)


        for i in range(10001):

            if i%1==0:
                controller.randomize()

            r=controller.run(env,i,0)# i%100 == -10)
            if i%10==0:
                print(i,r[-1],controller.team)
            if i%100==0 and 1:
                controller.test(env)
            if i%1000==0:
                controller.save("tests/qq"+str(frq)+"-"+str(trial)+".pkl")

            #print(r)
def test5(trial,frq):
    #print(np.random.get_state())[1]    
    np.random.seed(int(time.time()*100000)%100000)
    team=[0,0,1,1,2,2,3,3]
    team=[0,1,2,2,1,1,0,0]
    #team=team+team
    #team=np.array([i%int(frq) for i in range(16)])
    env=make_env(team)
    with tf.compat.v1.Session() as sess:
        
        controller = learner(team,sess)
        init=tf.compat.v1.global_variables_initializer()
        sess.run(init)
        controller.put("poi vals",vals)

        for i in range(10001):

            if i%1==0:
                controller.randomize()

            r=controller.run(env,i,0)# i%100 == -10)
            if i%10==0:
                print(i,r[-1],controller.team)
            if i%100==0 and 1:
                controller.test(env)
            if i%1000==0:
                controller.save("tests/c"+str(frq)+"-"+str(trial)+".pkl")

if 0:

    test1(20)
else:
    f=sys.argv[1]
    print(f)
    f=int(f)
    for i in range(4):
        p=mp.Process(target=test1,args=(i+(8*f),f))
        p.start()
        time.sleep(0.01)
        #p.join()

#env=make_env(None)


        
            
        
        
        
        
        
