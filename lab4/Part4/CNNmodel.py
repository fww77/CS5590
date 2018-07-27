#%%
import time
from cifar import CifarLoader, CifarDataManager
import os
import tensorflow as tf


timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "Part4/runs", timestamp))
train_summary_dir = os.path.join(out_dir, "summaries", "train")

lr = .01
steps = 10000
batch_size = 100
w_1 = 16
w_2 = 32
w_3 = 64

start = int(round(time.time() * 1000))
#from tensorflow.examples.tutorials.mnist import input_data



#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

mnist = CifarDataManager()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
x = tf.placeholder(tf.float32, shape=[None, 32, 32,3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])



x_image = tf.reshape(x, [-1, 32, 32, 3])

W_conv1 = weight_variable([8, 8, 3, w_1 ])
b_conv1 = bias_variable([w_1])



h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([8, 8, w_1, w_2])
b_conv2 = bias_variable([w_2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([8 * 8 * w_2, w_3])
b_fc1 = bias_variable([w_3])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * w_2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([w_3, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
opt = 'Adagrad'
train_step = tf.train.AdagradOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.histogram('cross_hist', cross_entropy)
merged = tf.summary.merge_all()

trainwriter = tf.summary.FileWriter(train_summary_dir, sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(steps):
    batch = mnist.train.next_batch(batch_size)
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    trainwriter.add_summary(summary, i)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
results = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
print("test accuracy " + str(results))
end = int(round(time.time() * 1000))
print("Time for building convnet: ")
print(end - start)
sess.close()


with open('Part4/Notes.txt', 'a') as f:
    s = opt+": " + str(timestamp) + ": learning rate " + str(lr)
    s += ":batch size "+ str(batch_size)+ ": steps " + str(steps)
    s += ": filters "+ str([w_1, w_2, w_3])+ ": test accuracy "+ str(results) + "\n"
    f.write(s)
    f.close