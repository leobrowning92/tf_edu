import tensorflow as tf
import load_image

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

size=16
tf.logging.set_verbosity(tf.logging.INFO)
sess=tf.InteractiveSession()
training=load_image.Dataset("line_data/training/",size)
evaluation=load_image.Dataset("line_data/evaluation/",size)
x= tf.placeholder(tf.float32, [None,size**2]) #note None allows any length
W= weight_variable([size**2,size**2])
b=bias_variable([size**2])

#implement model
y=tf.nn.softmax(tf.matmul(x,W)+b)#this implements our Model
#placeholder for correct values
y_=tf.placeholder(tf.float32,[None,size**2])

#cross entropy function
cross_entropy=tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step=tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)


# returns a list of booleans
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#returns the average of the booleans cast to floats. ie 1=True,0=False
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
rms=tf.sqrt(tf.reduce_mean(tf.squared_difference(y,y_)))
tf.global_variables_initializer().run()

for i in range(20):
    batch_xs,batch_ys = training.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    print(i,"test accuracy {}".format(rms.eval(feed_dict = {x:evaluation.images[:,:,0],y_:evaluation.images[:,:,1]})))
