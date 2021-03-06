import numpy as np
#import tensorflow as tf
import numpy as np


from .logger import logger
import pyximport
from .ccea import *

class learner:
    def __init__(self,team,sess,sim):
        self.log=logger()
        self.nagents=len(team)
       
        self.itr=0
        self.update_freq=20
        self.pol_freq=50
        self.types=max(team)+1
        self.team=team
        self.team_trials=1
        self.every_team=self.all_teams(self.nagents)
        initCcea(input_shape=8, num_outputs=6, num_units=20)(sim.data)
        #self.agents=[net(sess,[8,20,6], 0.0005) for i in range(self.types)]

    def act(self,S,data,trial):
        policyCol=data["Agent Policies"]
        A=[]
        for s,t in zip(S,self.team[trial]):
  
            a = policyCol[t].get_action(s)

            A.append(a)
        return A
    
    def store(self,S,A,R):
        for s,a,t,r in zip(S,A,self.team,R):
            
            self.agents[t].store(s,a,r)

    def learn(self,share=False):
        loss=[]
        for i in range(self.types):
            
            L=self.agents[i].batch_train()
            loss.append(L)
            if share:
                L=self.agents[i-1].batch_train(self.agents[i].buffer)
                loss.append(L)
        return loss

    def randomize(self):
        length=len(self.every_team)
        teams=[]
        for i in range(self.team_trials):
            idx=np.random.choice(length)
            teams.append(self.every_team[idx].copy())
        self.team=teams
        #self.team=np.random.randint(0,self.types,self.nagents)

    def save(self,fname="log.pkl"):
        print("saved")
        self.log.save(fname)

    def run(self,env,episode,render=False):
        populationSize=len(env.data['Agent Populations'][0])
        for worldIndex in range(populationSize):
            env.data["World Index"]=worldIndex
            
            for agent_idx in range(self.types):
                G=[]
                for trial in range(self.team_trials):
                    env.data["Current Agent"]=agent_idx #another loop needed
                    
                    s = env.reset()
                    
                    assignCceaPoliciesHOF(env.data)
                    #mods.assignHomogeneousPolicy(sim)

                    done=False
                
                    #self.log.clear("position")
                    i=0
                    while not done:
                        #self.log.store("position",np.array(env.data["Agent Positions"]),episode)
                        self.itr+=1
                        


                        if i%self.pol_freq==0:
                            #print(i)
                            a=self.act(s,env.data,trial)
                            #print(a)

                        action=self.idx2a(env,a)
                        
                        sp, r, done, info = env.step(action)
                        if r[0]==0:
                            r-=0.0
                        #r=r[0]
                        g=env.data["Global Reward"]
                        
                        s=sp
                        
                        if render:
                            env.render()
                        
                    G.append(g)
                    
                rewardCceaPoliciesHOF(env.data,sum(G))
        evolveCceaPolicies(env.data)

        self.log.store("reward",g)      
        return 0,g
    def put(self,key,data):
        self.log.store(key,data)

    def idx2a(self,env,idx):
        A_=[]
        for j in range(self.nagents):
            i=idx[j]

            loc=env.data["Poi Positions"][i]
            ang=env.data["Agent Orientations"][j]
            pos=env.data["Agent Positions"][j]

            heading=[loc[0]-pos[0],loc[1]-pos[1]]
            dst=(heading[0]**2.0+heading[1]**2.0)**0.5
            #trn=np.arccos( (heading[0]*ang[0]+heading[1]*ang[1])/( np.sqrt(heading[0]**2+heading[1]**2))* np.sqrt(ang[0]**2+ang[1]**2)  )    
            
            trn= np.arctan2( heading[1], heading[0] ) - np.arctan2(ang[1],ang[0])
            
            if trn>np.pi:
                trn-=2*np.pi
            if trn<-np.pi:
                trn+=2*np.pi

            spd=1.7#min(2.0,dst)

            a=[spd,trn]
            A_.append(a)
        
        return A_


    def test(self,env,itrs=50):
        print("test")
        old_team=self.team
        '''
        if self.dueling:
            idxs=[episode%2]*self.types
            #idxs=np.random.randint(2,size=self.types) #which of the pair of policies to use
        else:
        '''    
        #
        assignBestCceaPolicies(env.data)

        self.log.clear("position")
        self.log.clear("types")
        
        self.log.clear("poi")
        self.log.store("poi",np.array(env.data["Poi Positions"]))
        self.log.clear("poi vals")
        self.log.store("poi vals",np.array(env.data['Poi Static Values']))
        Rs=[]
        teams=self.all_teams(self.nagents)
        
        for i in range(len(teams)):
            
            team=np.array(teams[i])
         
            self.team=[team]
            #for i in range(itrs):

            #self.randomize()
            s=env.reset()
            done=False
            R=[]
            i=0
            self.log.store("types",self.team[0].copy(),i)
            
            while not done:
                
                self.log.store("position",np.array(env.data["Agent Positions"]),i)
                if i%self.pol_freq==0:
                    a=self.act(s,env.data,0)

                action=self.idx2a(env,a)
                sp, r, done, info = env.step(action)
                R.append(r[0])
                g=env.data["Global Reward"]
                s=sp
                i+=1
            Rs.append(g)
        self.log.store("test",Rs)
        
        self.team=old_team

    def quick(self,env,episode,render=False):
        s=env.reset()
        
        for i in range(100):
            a=[[0,0] for i in range(self.nagents)]
            sp, r, done, info = env.step(a)
        return [0.0]
            
    def all_teams(self,k):
        teams=[]
        for i in range(k+1):
            for j in range(k-i+1):
                team=[0]*i+[1]*j+[2]*(k-j-i)
                teams.append(team)
                #print(teams)
        return teams