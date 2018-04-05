import numpy as np
import tensorflow as tf
import time

np.random.seed(1234)
tf.set_random_seed(1234)

# ------------------
# 1. Define the network
# ------------------

# Define inputs
# x1, x2 with 4 samples [0,0] [0,1] [1,0] [1,1]
x_ = tf.placeholder(tf.float32, shape=[4,2], name = 'x-input')
# y with 4 results [0] [1] [1] [0]
y_ = tf.placeholder(tf.float32, shape=[4,1], name = 'y-input')

Weight1 = tf.Variable(tf.random_uniform([2, 2], -1, 1), name ="weight_hidden_layer")
Weight2 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name ="weight_output_layer")

tf.summary.scalar('weight_1', Weight1[0,0])
tf.summary.scalar('weight_2', Weight1[1,0])
tf.summary.scalar('weight_3', Weight1[0,1])
tf.summary.scalar('weight_4', Weight1[1,1])

Bias1 = tf.Variable(tf.zeros([2]), name = "bias_hidden_layer")
Bias2 = tf.Variable(tf.zeros([1]), name = "bias_output_layer")

with tf.name_scope("hidden_layer") as scope:
	op_layer2_output = tf.sigmoid(tf.matmul(x_, Weight1) + Bias1)
	tf.summary.histogram("weight_hidden_layer", Weight1)

with tf.name_scope("output_layer") as scope:
	op_output = tf.sigmoid(tf.matmul(op_layer2_output, Weight2) + Bias2)
	tf.summary.histogram("weight_output_layer", Weight2)

# Compute cost as Cross Entropy (1 - target) * log (1 - output)
with tf.name_scope("cost") as scope:
	# op_cost = tf.reduce_mean( ((y_ * tf.log(op_output)) +
	# 						  ((1 - y_) * tf.log(1.0 - op_output))) * -1)

	op_cost = tf.reduce_mean(tf.squared_difference(y_, op_output))

with tf.name_scope("train_gradient_descent") as scope:
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(op_cost)

# Prepare summary
tf.summary.scalar('cost', op_cost)
summaries = tf.summary.merge_all()

# Data to train the network
XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

# ------------------
# 2. Init Tensorflow
# ------------------
init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph)
sess.run(init)

# ------------------
# 3. Run
# ------------------
MAX_EPOCH = 400000
t_start = time.clock()
for i in range(MAX_EPOCH):
	sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
	if i % 1000 == 0:
		# print('Epoch ', i)
		print("\nOutput %d" % i)
		print(np.ravel(sess.run(op_output, feed_dict={x_: XOR_X, y_: XOR_Y})))
		# print('Theta1 ', sess.run(Theta1))
		# print('Bias1 ', sess.run(Bias1))
		# print('Theta2 ', sess.run(Theta2))
		# print('Bias2 ', sess.run(Bias2))
		print("cost ", sess.run(op_cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
		_, summ = sess.run([op_output, summaries], feed_dict={x_: XOR_X, y_: XOR_Y})
		writer.add_summary(summ, global_step=i)

final_weight2 = sess.run([Weight2], feed_dict={})

print ("\nSummary")
print ("------------")
print('Weight hidden layer w1,w2, w3, w4', np.ravel(sess.run([Weight1], feed_dict={})))
print('Bias hidden layer b1, b2', np.ravel(sess.run([Bias1], feed_dict={})))

print('Weight output layer w5, w6', np.ravel(sess.run([Weight2], feed_dict={})))
print('Bias output layer b3', np.ravel(sess.run([Bias2], feed_dict={})))

t_end = time.clock()
print('Elapsed time ', t_end - t_start)