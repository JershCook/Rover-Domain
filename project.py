"""
An example using the rover domain gym-style interface and the standard, included CCEA learning algorithms.
This is a minimal example, showing the minimal Gym interface.
"""
import numpy as np

from rover_domain_core_gym import RoverDomainGym
import code.ccea_2 as ccea
import code.agent_domain_2 as domain
from transfer.agent import multi
from mod_funcs import IndivReward
from sys import argv
import pickle

#heap
#paralell q

def discount(r,gamma):
    length=len(r)
    r_max=max(r)
    for i in range(2,length+1):
        r[-i][0]+=r[-i+1][0]*gamma
        if r[-i][0]>r_max:
            r[-i][0]=r_max
    return np.clip(r,-0.95,0.95)


episodeCount = 15000  # Number of learning episodes
STEPS=100

nagents=2
RENDER=0
TEST=0
minr=1.0
#TRAIN_POLICY_IDX=4

sim = RoverDomainGym(nagents,STEPS)
#mods.recipePoi(sim)
obs=sim.reset()
#obs=reduce_state(obs)

sim.data["Coupling"]=1
sim.data['Number of Agents']=nagents
obs_size=len(obs[0])
act_size=2
print(obs_size)


controller = multi(nagents,2,8)

DATA = "save/0.txt"
DATA2= "save/0.pkl"
#DATA3= str(nagents)+"agent/"+argv[1]+".ckpt"


open(DATA, 'w').close()
max_score=-0.1

for episodeIndex in range(episodeCount):
    if episodeIndex % 2 == 0:
        controller.NOISE_RATIO=1.0
    else:
        controller.NOISE_RATIO=0.0
    rewards=[]
    actions=[[] for i in range(nagents)]
    states =[[] for i in range(nagents)]
    states_p=[[] for i in range(nagents)]
#   for worldIndex in range(populationSize):
    obs = sim.reset()
    err=0.0
    R_=[]
    done = False
    stepCount = 0
    while not done:
    #   obs=reduce_state(obs)
        obs=np.asarray(obs)
        obs[:,:4]=0.0
        action=controller.act(obs)
        
        obs2, reward, done, info = sim.step(action)
        #print(np.max(obs2[:,:4]),np.max(obs2[:,4:20]))
        IndivReward(sim.data)
        reward=sim.data["Agent Rewards"]
        gr=sim.data["Global Reward"]
        #if gr>0: print(gr,stepCount)
        rewards.append([reward[0]])
        for a in range(nagents):
            states[a].append( obs[a] )
            actions[a].append(action[a])
            states_p[a].append(obs2[a])

        old_obs=obs            
        obs=obs2

        stepCount += 1
        if (RENDER == True) and episodeIndex%100==0:
            sim.render()

        if stepCount%5==0:
            err_,r_=controller.learn()
            err+=err_
            R_.append(r_)


    score=max(rewards)[0]
    rewards=np.array(rewards)/(float(nagents))
    rewards=discount(rewards,0.99)
    m_r=max(rewards)[0]
    
    rewards=[rewards for i in range(nagents)]        
           
    controller.store(states,actions,rewards,states_p)#,np.random.randint(0,nagents))
    
    
    #print(tr,episodeIndex, score,prob,err)
    #controller.prent()
    if score>max_score:
        max_score=score
        with open(DATA2, 'wb') as handle:
            pickle.dump(np.asarray(sim.data["Agent Position History"]), handle, protocol=pickle.HIGHEST_PROTOCOL)
            #pickle.dump(sim.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    with open(DATA, "a") as myfile:
        log=[episodeIndex, score,err,m_r,max(R_)] + controller.idxs
        myfile.write( ",".join([str(f) for f in log]))
        print(log)
        myfile.write('\n')
    
    
    
    
