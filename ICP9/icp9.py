import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = 'USA_Housing.xls'

# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows-1)])
n_samples = sheet.nrows - 1
data=np.delete(data,[0,3,4,6],axis=1) #delete the string column
data=data.astype(np.float64)

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
# 'Avg. Area House Age' vs 'Price'
X1 = tf.placeholder(tf.float32, name='X1')
Y1 = tf.placeholder(tf.float32, name='Y1')
#'Avg. Area Number of Rooms' vs 'Price'
X2 = tf.placeholder(tf.float32, name='X2')
Y2 = tf.placeholder(tf.float32, name='Y2')

# Step 3: create variavbles: weight and bias, initialized to 0
# Weight and Bias for 'Avg. Area House Age' vs 'Price'
w1 = tf.Variable(0.0, name='weights1')
b1 = tf.Variable(0.0, name='bias1')
# Weight and Bias for 'Avg. Area Number of Rooms' vs 'Price'
w2 = tf.Variable(0.0, name='weights2')
b2 = tf.Variable(0.0, name='bias2')

# Step 4: build model to predict Y
# model for 'Avg. Area House Age' vs 'Price'
Y_predicted1 = X1 * w1 + b1

# model for 'Avg. Area Number of Rooms'' vs 'Price'
Y_predicted2 = X2 * w2 + b2

# Step 5: use the square error as the loss function
# loss for 'Avg. Area House Age' vs 'Price'
loss1 = tf.square(Y1 - Y_predicted1, name='loss1')
#loss for 'Avg. Area Number of Rooms'' vs 'Price'
loss2 = tf.square(Y2 - Y_predicted2, name='loss2')

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss1)
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss2)
with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./graphs', sess.graph)

    # Step 8: train the model
    for i in range(20):  # train the model 50 epochs
        total_loss1 = 0
        total_loss2 = 0
        for x1, x2, y in data:
            # Session runs train_op and fetch values of loss
            _, l1 = sess.run([optimizer1, loss1], feed_dict={X1: x1, Y1: y})
            _, l2 = sess.run([optimizer2, loss2], feed_dict={X2: x2, Y2: y})
            total_loss1 += l1
            total_loss2 += l2
        print('Epoch {0}: {1}'.format(i, total_loss1 / n_samples))
        print('Epoch {0}: {1}'.format(i, total_loss2 / n_samples))

    # close the writer when you're done using it
    writer.close()

    # Step 9: output the values of w and b
    w1, b1,w2,b2 = sess.run([w1, b1, w2, b2])
    print (w1,b1,w2,b2)

# plot the results
X1,X2,Y = data.T[0], data.T[1],data.T[2]
plt.figure()
plt.plot(X1, Y, 'bo', label='Real data1')
plt.plot(X1, X1 * w1 + b1, 'r', label='Predicted data1')

plt.figure()
plt.plot(X2, Y, 'bo', label='Real data2')
plt.plot(X2, X2 * w2 + b2, 'r', label='Predicted data2')
plt.legend()
plt.show()

# Question2
from mpl_toolkits.mplot3d import axes3d

DATA_FILE = 'Smoking.xls'

# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X1 (Smoking Status),X2 (Age classification)and label Y (Death status 0=)
X1 = tf.placeholder(tf.float32, name='X1')
X2 = tf.placeholder(tf.float32, name='X2')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create variavbles: 2 weight and one bias, initialized to 0
w1 = tf.Variable(0.0, name='weights1')
w2 = tf.Variable(0.0, name='weights2')
b = tf.Variable(0.0, name='bias')

# Step 4: build model to predict Y
Y_predicted = w1*X1 + w2*X2 + b

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./graphs', sess.graph)

    # Step 8: train the model
    for i in range(50):  # train the model 50 epochs
        total_loss = 0
        for x1,x2,x3, y in data:
            # Session runs train_op and fetch values of loss
            Y_predicted, l = sess.run([optimizer, loss], feed_dict={X1: x1, X2: x2, Y: y})
            total_loss += l
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    # close the writer when you're done using it
    writer.close()

    # Step 9: output the values of w and b
    w1, w2, b = sess.run([w1,w2, b])
    print(w1,w2,b)

# plot the results, 3D plot
X1, X2, Y = data.T[0], data.T[1], data.T[2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X1, X2, X1 * w1 + X2 * w2 + b, 'bo', label='predicted Y')
ax.plot(X1, X2, Y, 'ko', label='Real Y')
plt.legend()
plt.show()

