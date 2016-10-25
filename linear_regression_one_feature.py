import numpy as np
import tensorflow as tf

# Model linear regression y = Wx + b
#placeholder : value being supplied, arg: none(can't remember why) 1(1 feature:house size)
x = tf.placeholder(tf.float32, [None, 1])

#Variable: W and b to be trained, args: first 1: single feature(house size) - 2nd 1: single output(house price)
W = tf.Variable(tf.zeros([1,1]))
b = tf.Variable(tf.zeros([1]))
product = tf.matmul(x,W)
y = product + b
y_ = tf.placeholder(tf.float32, [None, 1])

# Cost function sum((y_-y)**2)
cost = tf.reduce_mean(tf.square(y_-y))

# Training using Gradient Descent to minimize cost
train_step = tf.train.GradientDescentOptimizer(0.0000001).minimize(cost)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
steps = 1000
for i in range(steps):
  # Create fake data for y = W.x + b where W = 2, b = 0
  xs = np.array([[i]])
  ys = np.array([[2*i]])
  # Train
  feed = { x: xs, y_: ys }
  sess.run(train_step, feed_dict=feed)
  print("After %d iteration:" % i)
  print("W: %f" % sess.run(W))
  print("b: %f" % sess.run(b))
  print("cost: %f" % sess.run(cost, feed_dict=feed))

# NOTE: W should be close to 2, and b should be close to 0
