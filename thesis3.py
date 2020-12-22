"""
An example using the rover domain gym-style interface and the standard, included CCEA learning algorithms.
This is a minimal example, showing the minimal Gym interface.
"""
import numpy as np

from rover_domain_core_gym import RoverDomainGym
import code.ccea_2 as ccea
import code.agent_domain_2 as domain
import mods
from multiq.learner import learner
from sys import argv
import pickle

#heap
#paralell q

def discount(r,gamma):
    length=len(r)
    for i in range(2,length+1):
        r[-i][0]+=r[-i+1][0]*gamma
    return r

def reduce_state(state):
    state=np.asarray(state)
    return state[:,-4:]

episodeCount = 15000  # Number of learning episodes
populationSize = 50
nagents=8
RENDER=0
TEST=0
minr=1.0
#TRAIN_POLICY_IDX=4

sim = RoverDomainGym(nagents,100)
mods.recipePoi(sim)
obs=sim.reset()
#obs=reduce_state(obs)

sim.data["Coupling"]=3
sim.data['Number of Agents']=nagents
obs_size=len(obs[0])
act_size=4
print(obs_size)
nPolicies=4


controller = learner(nagents,obs_size,act_size, nPolicies,populationSize,minr)

fname="saves/qnet.pickle"
DATA = str(nagents)+"agent/bad"+argv[1]+".txt"
DATA2= str(nagents)+"agent/bad"+argv[1]+".pkl"
DATA3= str(nagents)+"agent/bad"+argv[1]+".ckpt"



for idx in range(nPolicies):
    controller.load('saves/genetic.pickle',idx)

open(DATA, 'w').close()
max_score=0.0

for episodeIndex in range(episodeCount):
    rewards=[]
    actions=[[] for i in range(nagents)]
    states =[[] for i in range(nagents)]
#   for worldIndex in range(populationSize):
    obs = sim.reset()

    done = False
    stepCount = 0        
    while not done:
    #    obs=reduce_state(obs)
        obs=np.asarray(obs)
        if TEST:
            if stepCount >0:
                controller.idxs=[0 for i in range(12)]
            if stepCount >50:
                controller.idxs=[1 for i in range(12)]
            if stepCount >100:
                controller.idxs=[2 for i in range(12)]
            if stepCount >150:
                controller.idxs=[3 for i in range(12)]

        obs_ = tr=np.sum(obs[:,-4:],axis=0)/nagents
        jointAction,idxs=controller.action2(obs,obs_)
        
        obs2, reward, done, info = sim.step(jointAction)
        #print(np.max(obs2[:,:4]),np.max(obs2[:,4:20]))
        
        
        rewards.append([reward[0]])
        for a in range(nagents):
            states[a].append( (obs[a],obs_) )
            actions[a].append(idxs[a])

        old_obs=obs            
        obs=obs2

        stepCount += 1
        if (RENDER == True) and episodeIndex%100==0:
            sim.render()
    
    tr=np.sum(old_obs[:,-4:],axis=0)
    rewards=discount(rewards,1.0)
    rewards=np.array(rewards)/(float(nagents*6)) +.01
    score,prob=rewards[-1][0],controller.rand_prob
    rewards=[rewards for i in range(nagents)]        
           
    #err=controller.train2(states,actions,rewards,5)#,np.random.randint(0,nagents))
    err=0.0
    print(tr,episodeIndex, score,prob,err)
    #controller.prent()
    if score>max_score:
        max_score=score
        with open(DATA2, 'wb') as handle:
            pickle.dump(np.asarray(sim.data["Agent Position History"]), handle, protocol=pickle.HIGHEST_PROTOCOL)
        controller.saveq(DATA3)

    with open(DATA, "a") as myfile:
        myfile.write( ",".join([str(f) for f in [episodeIndex, score,prob,err]]))
        myfile.write('\n')
    
    
    
    
