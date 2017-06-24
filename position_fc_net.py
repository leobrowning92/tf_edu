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
def random_check(dataset,axes):
    num=int(np.random.uniform(0,len(dataset)))
    for i in range(3):
        if i==2:
            axes[i].imshow(np.reshape(y.eval(feed_dict = {x:dataset[:,:,0]})[num],(16,16)))
        else:
            axes[i].imshow(np.reshape(dataset[num,:,i],(16,16)))


def visual_check(dataset):
    fig=plt.figure(facecolor="white")
    axes=[plt.subplot(3,3,i+1) for i in range(9)]
    for i in range(3):
        random_check(dataset,axes[i::3])
    plt.show()


size=16
sess=tf.InteractiveSession()
training=load_image.Dataset("line_data/training/",size)
evaluation=load_image.Dataset("line_data/evaluation/",size)
x= tf.placeholder(tf.float32, [None,size**2]) #note None allows any length
W_fc1= weight_variable([size**2,size**2*10])
b_fc1=bias_variable([size**2*10])
y1=tf.matmul(x,W_fc1)+b_fc1

W_fc2= weight_variable([size**2*10,size**2])
b_fc2=bias_variable([size**2])

#implement model
y=tf.matmul(y1,W_fc2)+b_fc2#this implements our Model
#placeholder for correct values
y_=tf.placeholder(tf.float32,[None,size**2])

#cross entropy function
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
reg=tf.nn.l2_loss(W_fc2)+tf.nn.l2_loss(W_fc1)
loss=tf.reduce_mean(cross_entropy+0.1*reg)
train_step=tf.train.GradientDescentOptimizer(1e-2).minimize(loss)


# returns a list of booleans
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#returns the average of the booleans cast to floats. ie 1=True,0=False
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
rms=tf.sqrt(tf.reduce_mean(tf.squared_difference(y,y_)))
tf.global_variables_initializer().run()

plt.ion()
step=[]
acc=[]
for i in range(2000):
    batch_xs,batch_ys = training.next_batch(10)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    if i%100==0:
        # print(i,"evaluation accuracy {}".format(accuracy.eval(feed_dict = {x:evaluation.images[:,:,0],y_:evaluation.images[:,:,1]})))
        test=accuracy.eval(feed_dict = {x:evaluation.images[:,:,0],y_:evaluation.images[:,:,1]})
        print(i,"training accuracy {}".format(test))
        step.append(i)
        acc.append(test)
        plt.plot(step,acc,'b')
        plt.pause(0.001)
plt.ioff()

visual_check(evaluation.images)
plt.imshow(W_fc1.eval())
plt.show()
plt.imshow(W_fc2.eval())
plt.show()
