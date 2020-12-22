import tensorflow as tf 
import numpy as np 
from collections import deque
from random import sample

class noise:
    def __init__(self,size,th=.2,sig=.15,mu=0.0,dt=1e-1):
        self.size=size
        self.th=th
        self.dt=dt 
        self.sig=sig 
        self.mu=mu  
        self.reset()
    
    def sample(self):
        self.state=self.state \
        + self.th*(self.mu-self.state)*self.dt \
        + self.sig*(self.dt**0.5)*np.random.normal(size=self.size)
        return self.state

    def reset(self):
        self.state=np.zeros(self.size)


class agent:
    def __init__(self, sess, s_dim, a_dim,  lr, batch):
        self.sess=sess
        self.s_dim=s_dim
        self.a_dim=a_dim
        self.batch=batch

        self.base_line=False

        self.var=0.05
        self.a_hidden=10
        self.c_hidden=40

        self.s=tf.placeholder(tf.float32,[None,s_dim])
        self.a=tf.placeholder(tf.float32,[None,a_dim])
        self.r=tf.placeholder(tf.float32,[None,1])

        self.hist=deque(maxlen=1000000)

        self.actor, self.a_params  = self.gen_actor()
        self.critic,self.c_params = self.gen_critic()

        self.action_grads = tf.gradients(self.critic, self.a)

        self.a_grad= tf.placeholder(tf.float32, [None, self.a_dim])
        self.actor_gradients = tf.gradients(self.actor, self.a_params, -self.a_grad)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch), self.actor_gradients))

        self.a_opt = tf.train.AdamOptimizer(lr*0.1).apply_gradients(zip(self.actor_gradients, self.a_params))
        
        
        self.loss = tf.losses.mean_squared_error(self.critic,self.r)
        self.c_opt = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def activate(self,x):
        return tf.nn.tanh(x)

    def gen_critic(self):
        with tf.name_scope("critic") as scope:
            w1=tf.Variable(tf.random_normal([self.s_dim, self.c_hidden],stddev=self.var))
            b1=tf.Variable(tf.random_normal([self.c_hidden],stddev=self.var))
            netc=tf.matmul(self.s,w1)+b1

            w2=tf.Variable(tf.random_normal([self.a_dim, self.c_hidden],stddev=self.var))
            b2=tf.Variable(tf.random_normal([self.c_hidden],stddev=self.var))
            neta=tf.matmul(self.a,w2)
            
            net=self.activate(neta+netc)

            w3=tf.Variable(tf.random_normal([self.c_hidden,1],stddev=self.var))
            b3=tf.Variable(tf.random_normal([1],stddev=self.var))
            net=self.activate(tf.matmul(net,w3)+b3)

            return net, [w1,w2,w3,b1,b2,b3]


    def gen_actor(self):
        with tf.name_scope("actor") as scope:
            w1=tf.Variable(tf.random_normal([self.s_dim, self.a_hidden],stddev=self.var))
            b1=tf.Variable(tf.random_normal([self.a_hidden],stddev=self.var))
            #net=self.activate(tf.matmul(self.s,w1)+b1)

            net=tf.nn.relu(tf.matmul(self.s,w1)+b1)
            
            w2=tf.Variable(tf.random_normal([self.a_hidden,self.a_dim],stddev=self.var))
            b2=tf.Variable(tf.random_normal([self.a_dim],stddev=self.var))
            net=self.activate(tf.matmul(net,w2)+b2)

            return net, [w1,w2,b1,b2]

    def store(self,s,a,r,sp):
        for h in zip(s,a,r):
            self.hist.append(h)

    def actor_train(self,s,a):
        grads=self.sess.run(self.action_grads,feed_dict={self.s:s,self.a:a})
        self.sess.run(self.a_opt,feed_dict={self.s:s,self.a_grad:grads[0]})

    def critic_train(self,s,a,r):
        _,loss,r_ = self.sess.run([self.c_opt,self.loss,self.critic],feed_dict={self.s:s,self.a:a,self.r:r})
        return loss,max(r_)[0]

    def train_all(self):
        if len(self.hist)<self.batch:
            return 0.0,0.0
        hist=sample(self.hist,self.batch)
        S,A,R=[],[],[]
        for s,a,r in hist:
            S.append(s)
            A.append(a)
            R.append(r)

        L,r_=self.critic_train(S,A,R)
        self.actor_train(S,A)
        return L,r_

    def act(self,s):
        return self.sess.run(self.actor,feed_dict={self.s:s})

    def save(self,fname):
        saver=tf.train.Saver()
        saver.save(self.sess,fname+".ckpt")

    def load(self,fname):
        saver=tf.train.Saver()
        saver.restore(self.sess,fname+".ckpt")