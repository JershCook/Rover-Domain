import tensorflow as tf 
import numpy as np 

w=tf.Variable(tf.random_normal([4, 1]))

b=tf.Variable(tf.random_normal([4, 1]))

diff=tf.losses.KLD(w,b)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a=sess.run(diff)
    print(a)