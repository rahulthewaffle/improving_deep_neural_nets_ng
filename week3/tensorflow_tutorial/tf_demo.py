# TensorFlow Demo

import numpy as np
import tensorflow as tf

# w is the parameter we're adjusting so that we can complete our optimization objective, so we declare it as a TF Variable
w = tf.Variable(0, dtype = tf.float32)

# What if optimization objective is a function of input training data?
# placeholder function tells TF that this object is something you will provide values for later
x = tf.placeholder(tf.float32, [3,1])

coefficient = np.array([1.], [-10.], [25.]) # values for x

# TensorFlow knows how to calculate the derivative/backprop without issue since it uses a computational graph approach for all its tf.[mathematical function] calls
# Hence you only need to implement forward prop calculation for the cost
# Alternatively, once you declare something as a TensorFlow Variable object, TF overloads the standard python mathematical operators to do similar backprop/gradient computational graph magic
# cost = w**2 - 10*w + 25 is the same as
# cost = tf.add(tf.add(w**2, tf.multiply(-10.,w)), 25)

# Optimization objective that uses dynamic input x:
# x is data that controls the TF Variable optimized in this function
cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)
print(session.run(w))

# Sometimes the above three lines are written as:
# with tf.Session() as session:
#   session.run(init)
#   print(session.run(w))
# 'with' command is being used to ensure garbage collection of Session object in the event of an error or exception in the inner loop

session.run(train, feed_dict = {x:coefficients})
print(session.run(w))

for i in range(1000):
    session.run(train, feed_dict = {x:coefficients})
print(session.run(w))
