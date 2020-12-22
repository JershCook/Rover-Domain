import numpy as np
import tensorflow as tf
from .ddpg import  noise,agent
#from .ddpg2 import agent

class multi:

    def __init__(self,n,a,s,rand=True):
        BATCH=32
        LR=1e-3
        self.epsilon=0.15
        self.NOISE_RATIO=1.0

        self.sess=tf.InteractiveSession()

        self.rand=rand
        self.n_agents=n
        self.noise=[noise(a) for i in range(n)]
        self.agents=[agent(self.sess,s,a,LR,BATCH) for i in range(n)]

        self.sess.run(tf.global_variables_initializer())
        self.set_idxs()

    def set_idxs(self):
        for n in self.noise:
            n.reset()
        self.idxs=[]
        for i in range(self.n_agents):
            if np.random.random()<self.epsilon and self.rand==True:
                self.idxs.append(np.random.randint(0,self.n_agents))

            else:
                self.idxs.append(i)

    def act(self,state):
        actions=[]
        
        for i in range(self.n_agents):
            s=state[i]
            s=np.array([s])
            idx=self.idxs[i]
            agents=self.agents[idx]   
            
            action=agents.act(s)[0]+self.NOISE_RATIO*self.noise[i].sample()
            actions.append(action)
            
        actions=np.array(actions)
        actions=np.clip(actions,-1,1)
        return actions

    def store(self,S,A,R,SP):
        self.set_idxs()
        for i in range(self.n_agents):
            self.agents[i].store(S[i],A[i],R[i],SP[i])
    

    def learn(self):
        e=0.0
        r=[]
        for i in range(self.n_agents):
            e_,r_=self.agents[i].train_all()
            e+=e_
            r.append(r_)
        return e,max(r)

