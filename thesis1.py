"""
An example using the rover domain gym-style interface and the standard, included CCEA learning algorithms.
This is a minimal example, showing the minimal Gym interface.
"""
from rover_domain_core_gym import RoverDomainGym
import code.ccea_2 as ccea
import code.agent_domain_2 as domain
import mods
from multiq.learner import learner

episodeCount = 10000  # Number of learning episodes
populationSize = 50
nagents=20
RENDER=0
LOAD=1
TRAIN_POLICY_IDX=3

sim = RoverDomainGym(nagents,50)
mods.recipePoi(sim)
obs=sim.reset()

sim.data["Coupling"]=0
#sim.data['Number of Agents']=nagents
obs_size=len(obs[0])
act_size=4
print(obs_size)

controller = learner(nagents,obs_size, act_size, 5, populationSize)
if LOAD:
    controller.load('saves/genetic.pickle',TRAIN_POLICY_IDX)

mods.posInit(sim.data,0.0,1.0)

r_avg=0.0
for episodeIndex in range(episodeCount):
    rewards=[]
    
    for worldIndex in range(populationSize):
        obs = sim.reset()

        done = False
        stepCount = 0        
        while not done:
        
            jointAction=controller.policy_action(obs,TRAIN_POLICY_IDX,worldIndex)
            
            obs, reward, done, info = sim.step(jointAction)
            
            stepCount += 1
            if ( episodeIndex%1==0 and worldIndex==0 and RENDER == True):
                sim.render()
                
        multiR=mods.multiReward(sim)
        #print(multiR)        
        rewards.append( multiR[TRAIN_POLICY_IDX] )
    score=max(rewards)
    r_avg=0.9*r_avg+0.1*score
    print(episodeIndex, round(score,3),round(r_avg,3))
    
    if score>0.9 or (TRAIN_POLICY_IDX == 4 and score>-2.5):
        
        controller.save('saves/genetic.pickle',TRAIN_POLICY_IDX)
    controller.policy_train(rewards,TRAIN_POLICY_IDX)
    
    
    
