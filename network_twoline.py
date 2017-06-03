import tensorflow as tf
import load_image
training=load_image.Dataset("line_data/training/")
evaluation=load_image.Dataset("line_data/evaluation/")
x= tf.placeholder(tf.float32, [None,1024]) #note None allows any length
W= tf.Variable(tf.zeros([1024,1024]))
b=tf.Variable(tf.zeros([1024]))

#implement model
y=tf.nn.softmax(tf.matmul(x,W)+b)#this implements our Model
#placeholder for correct values
y_=tf.placeholder(tf.float32,[None,1024])

#cross entropy function
cross_entropy=tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs,batch_ys = training.next_batch(10)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

# tf.argmax returns the index of the highest entry in a tensor along a given axis
# returns a list of booleans
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#returns the average of the booleans cast to floats. ie 1=True,0=False
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(sess.run(accuracy,feed_dict={x:evaluation.images[:,:,0],y_:evaluation.images[:,:,1]}))