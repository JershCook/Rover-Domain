import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

class net:

    def __init__(self,sess, size,lr=0.001):
    
        self.nlayers=len(size)
        self.size=size
        self.sess=sess
        
        self.x = tf.placeholder(tf.float32, [None, size[0]])
   
        y=self.x
        for i in range(self.nlayers-1):
            W = tf.Variable(tf.random_normal([ size[i], size[i+1] ],stddev=0.1))
            b = tf.Variable(tf.random_normal([ size[i+1] ],stddev=0.1)) 
            y=tf.matmul(y, W) + b
            #if i<self.nlayers-2:
            #y = tf.tanh(y)
            y = tf.sigmoid(y)

        self.out=y
        
        # Define loss and optimizer
        self.y_ = tf.placeholder(tf.float32, [None, size[-1]])
        self.loss = tf.losses.mean_squared_error(self.y_,y)
        self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)
        #self.train_step = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)
        
        
    def feed(self,x):
        return self.sess.run( self.out,feed_dict={self.x:x})
    
    def train(self,x,y): 
        err,_=self.sess.run( [self.loss,self.train_step],feed_dict={self.x:x,self.y_:y})
        return err

if __name__ == "__main__":    
    sess=tf.InteractiveSession()
    s=[1,3,2,1]
    n=net(sess,s)

    
    sess.run(tf.global_variables_initializer())
    x=np.array([[i] for i in np.linspace(-5,5,40) ])
    y=x**2
    print(x.T)
    for i in range(5000):
         n.train(x,y)
         
    #y_=n.feed(x)
    
    #plt.plot(x.T[0],y.T[0])
    #plt.plot(x.T[0],y_.T[0])
    #plt.show()
   

