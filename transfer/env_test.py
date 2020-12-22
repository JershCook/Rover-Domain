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


BEST=-80.0


with tf.Session() as sess:
    
    player=agent(sess,s_size,a_size,0.001,64)
    
    sess.run(tf.global_variables_initializer())
    player.load("save/test0")
    for i in range(10000):
        state=env.reset()       
        R=0.0
        for j in range(1000):
            
            env.render()
            act=player.act(np.array([state]))[0]
            
            action=act


            state, reward, done, info = env.step(action)
            R+=reward
            if done or j==999:
                print(i,round(R,4))
                break
            
       
