from __future__ import print_function, division
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
# The code for this lab is based off the great climate testdata sample!

# Read and split data
DATA_FILE = pd.read_csv('testdata.txt')
X_train, X_test, Y_train, Y_test = train_test_split(
    DATA_FILE[["Temp", "Humid", "Bright", "CarbonDioxide", "HumidityRate"]].values,
    DATA_FILE["Occupancy"].values.reshape(-1, 1), random_state=42)

# Variables,feed actual training examples.
x = tf.placeholder(tf.float32, [None, 5], name='X')
y = tf.placeholder(tf.float32, [None, 1], name='Y')

# create variables: weight and  bias
w = tf.Variable(tf.zeros([5, 1]), name='weights')
b = tf.Variable(tf.zeros([1]), name='bias')

# build sigmoid model to predict Y
prediction = tf.nn.sigmoid(tf.matmul(x, w) + b)

# use the mean square error as the loss function
loss = tf.reduce_mean(tf.square(y - prediction, name='loss'))

# use gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.005).minimize(loss)

# redefine the accuracy
correct_prediction = tf.equal(tf.round(prediction), y)

# get the index with the largest value across axis of a tensor
LardestInd = tf.argmax(prediction, )

# takes the average of predictions across dimensions of a tensor.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)

# # Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Session: Usually needs to be quite large to get good results,
with tf.Session() as sess:
    # initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())
    # save the figure in the fold
    writer = tf.summary.FileWriter('./graphs/logis_reg', sess.graph)
    # train the model
    for i in range(50): # train the model 50 epochs
        total_loss = 0
        # We perform one update step by evaluating the training op (including it
        # in the list of returned values for session.run()
        _, l = sess.run([optimizer, loss], feed_dict={x: X_train, y: Y_train})
        total_loss += l
        plt.plot(i, total_loss, 'bo')
        print("Epoch:", "%04d" % i, "loss=", total_loss)

    # close the writer when you're done using it
    writer.close()

    # output the values of w and b
    w, b = sess.run([w,b])
    print(w,b)
    prediction = sess.run(prediction, feed_dict={x: X_test})
    print("accuracy =", sess.run(accuracy, feed_dict={x: X_test, y: Y_test}))

# plot the results,
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.legend()
plt.show()