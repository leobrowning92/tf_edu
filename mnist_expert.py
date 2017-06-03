import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

def weight_variable(shape):
    """initialize weights with a small ammount of noise to break
    symmetry and prevent 0 gradients"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """neurons of form f(x)=max(0,x)
    called ReLU (rectified linear unit ) neurons initialized with
    slight positive bias to prevent dead neurons"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess =tf.InteractiveSession()

# note none dimension will be set by batch size
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])



### 1st convolutional layer ###

# encodes a feature window of 5x5 from one
# input channel (image) to 32 output channels (32 features)
W_conv1 = weight_variable([5, 5, 1, 32])
# one shared bias for each features
b_conv1 = bias_variable([32])
#reshape the image to a 4d tensor with 2nd and 3rd dimension
#of height and witdth and 4th number of color channels
# -1 denotes an inferred dimension based on the input data
x_image=tf.reshape(x,[-1,28,28,1])
# apply the convolution to the x_image (convolve x_image and weights)
# apply the ReLU function and mac pool
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
# reduce to 14x14
h_pool1=max_pool_2x2(h_conv1)

### 2nd convolution layer####

#32 input layers (conv1) 64 output layers. 5x5 feature size
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#reduce to 7x7
h_pool2 = max_pool_2x2(h_conv2)


### fully connected layer ###
# allows processing on the entire image
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
#reshape tensor from pooling layers into batch of vectors.
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#vectors times weights +biases applying relu function
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


### Dropout layer ###

#reduces oversampling, but has the most effect on large networks
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

### Readout layer ###
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( labels=y_,logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy =  tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch=mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step {},training accuracy {}".format(i,train_accuracy))
    train_step.run(feed_dict = {x:batch[0],y_:batch[1],keep_prob:0.5})

print("test accuracy {}".format(accuracy.eval(feed_dict = {x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})))
