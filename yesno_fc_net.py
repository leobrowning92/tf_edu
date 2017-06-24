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
training=load_image.Dataset("yn_data/training/",size)
evaluation=load_image.Dataset("yn_data/evaluation/",size)

x= tf.placeholder(tf.float32, [None,size**2]) #note None allows any length
W_fc1= weight_variable([size**2,size**2])
b_fc1=bias_variable([size**2])
y1=tf.matmul(x,W_fc1)+b_fc1

W_fc2= weight_variable([size**2,2])
b_fc2=bias_variable([2])

#implement model
y=tf.matmul(y1,W_fc2)+b_fc2#this implements our Model

#placeholder for correct values
y_=tf.placeholder(tf.float32,[None,2])

#cross entropy function
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
reg=tf.nn.l2_loss(W_fc2)+tf.nn.l2_loss(W_fc1)
loss=tf.reduce_mean(cross_entropy+0.001*reg)
train_step=tf.train.GradientDescentOptimizer(1e-3).minimize(loss)


# returns a list of booleans
correct_prediction=tf.equal(tf.argmax(y,axis=1),tf.argmax(y_,axis=1))
#returns the average of the booleans cast to floats. ie 1=True,0=False
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.global_variables_initializer().run()
print(evaluation.yn[:10])
plt.ion()
step=[]
eval_acc=[]
train_acc=[]
for i in range(10000):
    batch_xs,batch_ys = training.next_yesno_batch(10)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    if i%100==0:
        # print(i,"evaluation accuracy {}".format(accuracy.eval(feed_dict = {x:evaluation.images[:,:,0],y_:evaluation.images[:,:,1]})))
        eval_test=accuracy.eval(feed_dict = {x:evaluation.images[:,:,0],y_:evaluation.yn})
        train_test=accuracy.eval(feed_dict = {x:batch_xs,y_:batch_ys})
        print(i,"training accuracy {}".format(test))
        step.append(i)
        eval_acc.append(test)
        plt.plot(step,eval_acc,'b')
        plt.plot(step,train_acc,'r')
        plt.pause(0.001)
plt.ioff()
plt.show()
# visual_check(evaluation.images)
plt.imshow(W_fc1.eval())
plt.show()

# plt.imshow(W_fc2.eval())
# plt.show()
