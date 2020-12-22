import tensorflow as tf
import numpy as np
from collections import deque
from random import seed, sample, random, randint
import pickle
import heapq

from .qnet import net
from .genagent import agent




class learner:

    def __init__(self,nagents,nstate,naction,npolicies,npop,minr=0.05):
        self.nagents=nagents
        self.nstate =nstate
        self.naction=naction
        self.npolicies=npolicies
        self.histories=[deque(maxlen=100000) for i in range(nagents)]
        self.zero_hist=[deque(maxlen=100000) for i in range(nagents)]
        self.heap=[[]for i in range(nagents)]
        self.sess=tf.InteractiveSession()
        

        self.rand_prob=1.0
        self.step=0
        self.idxs=None
        self.LR=0.001
        self.SPLIT_TRAIN=0
        self.minr=minr
        self.priority=False
        self.counter=0

        #qsize=[naction+nstate, int( (naction+nstate) * 2),4,1]
        #qsize=[nstate, 2*int( npolicies ),npolicies]

        qsize=[9,int( npolicies )*3,npolicies]

        psize=[nstate, int( (naction+nstate)/2 ),naction]
        
        self.qnets = [net(self.sess,qsize,self.LR) for i in range(nagents)]
        
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        self.policies=[ agent(psize,npop)  for i in range(self.npolicies) ]
        self.bestpolicies=[best.policy() for best in self.policies]
    def qstate(self,s,st):
        r=[]
        for i in range(5):
            r.append(max(s[i*4:(i+1)*4]))
        return np.hstack([r,s[-4:]])    
    
    def save(self,fname,idx):
        fname=fname.split('.')
        fname.insert(1,str(idx)+'.')
        fname=''.join(fname)
        
        with open(fname, 'wb') as handle:
            pickle.dump(self.policies[idx], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    def load(self,fname,idx):
        fname=fname.split('.')
        fname.insert(1,str(idx)+'.')
        fname=''.join(fname)
        
        with open(fname, 'rb') as handle:
            self.policies[idx] = pickle.load(handle)
        self.bestpolicies=[best.policy() for best in self.policies] 


    def saveq(self,fname):
        self.saver.save(self.sess, fname)
        
    
    def loadq(self,fname):
        self.saver.restore(self.sess, fname)
        
    def statemod(self,s,pindex):
        s=np.asarray(s).copy()
        mask=np.ones(s.shape,dtype=bool)
        mask[pindex*4+4:pindex*4+8]=False
        s[mask]=0.0
        return s

    def policy_action(self,states,pindex,popindex):
        policy = self.policies[pindex].policy(popindex)
        actions=[]
        for s in states:
            s=self.statemod(s,pindex)
            actions.append(policy.feed(s)[0])
        actions=np.array(actions)
        return actions
        
        
    def policy_train(self,rewards,pindex):
        policy=self.policies[pindex]
        best=policy.train(rewards)
        self.bestpolicies[pindex]=best    
        
        
    def action(self,state):
        Action=[]

        self.rand_prob*=.99999995
        if self.rand_prob<.15: self.rand_prob=.15
        if self.step == 0: self.idxs=[]
        for agentindex in range(self.nagents):
            s=state[agentindex]
            actions=[best.feed(s)[0] for best in self.bestpolicies]
            qnet=self.qnets[agentindex]
            
            if self.step == 0:
                exprewards=[ qnet.feed(  np.array([ np.hstack([s,a]) ]) ) for a in actions]
                #print(exprewards)
                index=np.argmax(exprewards)
            #print(index)

                        
                if self.rand_prob > random():
                    index=randint( 0, len(exprewards)-1 )
                #index=0
        
            if self.step!=0:    
                index=self.idxs[agentindex]
            else:
                self.idxs.append(index)

            Action.append(actions[index])
            

        self.step+=1
        if self.step>=10:
            self.step=0

        #print(index)    
        return np.array(Action),self.idxs
        
        
    def train(self,states,actions,rewards,train_steps=1):
        err=0.0
        for agentindex in range(self.nagents):
        
            for r,s,a in zip(rewards[agentindex],states[agentindex],actions[agentindex]):
                
                if r[0]==0.0 and self.SPLIT_TRAIN:
                    self.zero_hist[agentindex].append([r,np.hstack([s,a])])
                else:
                    self.histories[agentindex].append([r,np.hstack([s,a])])
            
            qnet=self.qnets[agentindex]
                
            for i in range(train_steps):
                if not self.SPLIT_TRAIN or (random()>.5 and len(self.histories[agentindex]) > 100):
                    hist=sample(self.histories[agentindex],64)
                     
                elif (len(self.zero_hist[agentindex])>100):
                    hist=sample(self.zero_hist[agentindex],64)

                else:
                    continue    

                SA,R=[],[]
                for h in hist:
                    r,sa=h
                    SA.append(sa)
                    R.append(r)
                    
                SA=np.array(SA)
                R=np.array(R)
                
                err+=qnet.train(SA,R)
        
        return err/float(train_steps)/float(self.nagents)


    def action2(self,state,st):
        Action=[]

        self.rand_prob*=.999995
        if self.rand_prob<self.minr: self.rand_prob=self.minr
        if self.step == 0: self.idxs=[]
        exprewards=None
        for agentindex in range(self.nagents):
            s=state[agentindex]
            
            s_=self.qstate(s,st)
            
            qnet=self.qnets[agentindex]
            
            if self.step == 0:
                exprewards= qnet.feed(  np.array([s_]) )[0]

                #print(exprewards)
                index=np.argmax(exprewards)
            #print(index)

                        
                if self.rand_prob > random():
                    index=randint( 0, len(exprewards)-1 )
                    #print(index)
                #index=0
        
            if self.step!=0:    
                index=self.idxs[agentindex]
            else:
                self.idxs.append(index)
            
            best = self.bestpolicies[index]
            s2=self.statemod(s,index)
            actions=best.feed(s2)[0]

            Action.append(actions)
            

        self.step+=1
        if self.step>=10:
            self.step=0
        #if exprewards is not None: print(exprewards)
        #print(index)    
        return np.array(Action),self.idxs
        
    def store(self,data,idx):
        r=data[0]

        if self.priority == True:
            data=(data[0][0],self.counter,data[1],data[2])
            self.counter+=1
            #print(data)
            
            heapq.heappush(self.heap[idx],data)
            if len(self.heap[idx])>100000:
                heapq.heappop(self.heap[idx])
            return
        
        if r[0]==0.0 and self.SPLIT_TRAIN:
            self.zero_hist[idx].append(data)
        else:
            self.histories[idx].append(data)

    def sample(self,idx):
        
        if self.priority == True:
            return sample(self.heap[idx],32)

        if not self.SPLIT_TRAIN or (random()>.5 and len(self.histories[idx]) > 100):
            return sample(self.histories[idx],32)
                
        elif (len(self.zero_hist[idx])>100):
            return sample(self.zero_hist[idx],32)

        else:
            return None    

    def prent(self):
        print(self.heap[0][0][0])

    def train2(self,states,actions,rewards,train_steps=1,train_idx=-1):
        err=0.0
        trainpop=range(self.nagents)
        if train_idx>=0:
            trainpop=[train_idx]
        for agentindex in trainpop:
        
            for r,s,a in zip(rewards[agentindex],states[agentindex],actions[agentindex]):
                idx=a
                data=[r,s,idx]
                self.store(data,agentindex)
               
            
            qnet=self.qnets[agentindex]
                
            for i in range(train_steps):
                hist=self.sample(agentindex)
                if hist == None:
                    continue

                S,R,IDX=[],[],[]
                for h in hist:
                    if self.priority == True:
                        r,c,s,idx=h
                        R.append([r])
                    else:
                        r,s,idx=h
                        R.append(r)
                    s,st=s
                    s=self.qstate(s,st)

                    S.append(s)
                    
                    IDX.append(idx)
                    
                S=np.array(S)

                R_=qnet.feed(S)
                #R_*=1.2

                for j in range(len(R_)):
                    
                    idx=IDX[j]
                    r=R[j][0]
                    #print(r-R_[j][idx])
                    R_[j][idx]=r

                err+=qnet.train(S,R_)
        
        return err/float(train_steps)/float(self.nagents)