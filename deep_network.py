import tensorflow as tf
import load_image
import matplotlib.pyplot as plt
import numpy as np

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
def random_check(dataset,axes):
    num=int(np.random.uniform(0,len(dataset)))
    for i in range(3):
        if i==2:
            axes[i].imshow(np.reshape(y_conv.eval(feed_dict = {x:dataset[:,:,0]})[num],(16,16)))
        else:
            axes[i].imshow(np.reshape(dataset[num,:,i],(16,16)))


def visual_check(dataset):
    fig=plt.figure(facecolor="white")
    axes=[plt.subplot(3,3,i+1) for i in range(9)]
    for i in range(3):
        random_check(dataset,axes[i::3])
    plt.show()

def visualize_conv(W_conv1,num):
    fig=plt.figure(facecolor="white")
    size=int(np.sqrt(num))+1
    axes=[plt.subplot(size,size,i+1) for i in range(num)]
    print(W_conv1[:,:,0,0].shape)
    for i in range(num):
        axes[i].imshow(W_conv1[:,:,0,i])
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    plt.show()




size=16
tf.logging.set_verbosity(tf.logging.INFO)
sess=tf.InteractiveSession()

training=load_image.Dataset("line_data/training/",size)
evaluation=load_image.Dataset("line_data/evaluation/",size)


x= tf.placeholder(tf.float32, [None,size**2]) #note None allows any length
y_=tf.placeholder(tf.float32,[None,size**2])



# encodes a feature window of 5x5 from one
# input channel (image) to 32 output channels (32 features)
nfeatures=16
W_conv1 = weight_variable([5, 5, 1, nfeatures])
# one shared bias for each features
b_conv1 = bias_variable([nfeatures])

x_image=tf.reshape(x,[-1,size,size,1])

# apply the ReLU function and mac pool
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
# reduce to 8x8
h_pool1=max_pool_2x2(h_conv1)

W_fc1 = weight_variable([int(size/2 * size/2 * nfeatures), size**2])
b_fc1 = bias_variable([size**2])

#reshape tensor from pooling layers into batch of vectors.
h_pool1_flat = tf.reshape(h_pool1, [-1, int(size/2*size/2*nfeatures)])
#vectors times weights +biases applying relu function
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

### Dropout layer ###

#reduces oversampling, but has the most effect on large networks
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

### Readout layer ###
W_fc2 = weight_variable([size**2, size**2])
b_fc2 = bias_variable([size**2])
y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

#cross entropy function
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step=tf.train.GradientDescentOptimizer(1e1).minimize(cross_entropy)


# returns a list of booleans
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
#returns the average of the booleans cast to floats. ie 1=True,0=False
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
rms=tf.sqrt(tf.reduce_mean(tf.squared_difference(y_conv,y_)))
tf.global_variables_initializer().run()
for i in range(20):
    batch_xs,batch_ys = training.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    if i%10==0:
        # print(i,"evaluation accuracy {}".format(accuracy.eval(feed_dict = {x:evaluation.images[:,:,0],y_:evaluation.images[:,:,1]})))
        print(i,"training accuracy {}".format(rms.eval(feed_dict = {x:batch_xs,y_:batch_ys})))

# visual_check(evaluation.images)
visualize_conv(W_conv1.eval(),16)
# plt.imshow(np.reshape(W.eval(),(256,256)))
# plt.show()
