import tensorflow as tf 
import numpy as np
from ddpg import agent,noise
import gym
import matplotlib.pyplot as plt

env = gym.make('BipedalWalker-v2')
s_size=len(env.observation_space.low)
a_size=len(env.action_space.low)
print(env.action_space.low,env.action_space.high)

print(s_size,a_size)
plt.ion()

BEST=-80.0

def disc(R,gamma,divider=1.0):
    for i in range(len(R)-1):
        R[-i-2][0]+=R[-i-1][0]*gamma
    for i in range(len(R)):
        R[i][0]= R[i][0]/divider
    
    return R
with tf.Session() as sess:
    
    player=agent(sess,s_size,a_size,0.001,64)
    
    sess.run(tf.global_variables_initializer())

    R_hist=[]
    L_hist=[]
    for i in range(10000):
        state=env.reset()
        variance=noise(a_size)
        S,A,R,L=[],[],[],[]
        
        for j in range(1000):
            if i%50==-1:
                env.render()
            act=player.act(np.array([state]))[0]
            act+=variance.sample()

            A.append(act)
            S.append(state)
            
            action=act
            #action=np.argmax(act)

            state, reward, done, info = env.step(action)
            R.append([reward])
            if done or j==999:
                summ=sum([i[0] for i in R])
                if summ>BEST:
                    BEST=summ
                    player.save("save/test0")
                R=disc(R,.99,100.0)
                
                print(i,round(R[0][0],3),round(R[-1][0],3),round(summ,3),j)
                R_hist.append(summ)
                if len(L)==0: L=[1.0]
                L_hist.append(np.mean(L))
                break
            if j%5:    
                l=player.train_all()
                if l!=None: 
                    L.append(l)
        player.store(S,A,R)
        print(max(R),min(R))
        if i%50==0:
            plt.clf()
            plt.subplot(3,1,1)
            plt.plot(R_hist)
            conv = np.convolve(np.ones(25)/25.0,R_hist,"valid")
            plt.subplot(3,1,2)
            plt.plot(conv)
            plt.subplot(3,1,3)
            plt.plot(np.log10(L_hist))
            plt.pause(0.01)
