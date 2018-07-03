import tensorflow as tf
#creat three axb matrix
matrix_a = tf.constant([2,3])
#Constant 1-D Tensor populated with value list [2,3].

matrix_b= tf.constant([3,4])
#Constant 1-D Tensor populated with value list [3,4].


matrix_c = tf.constant([4,5])
#Constant 1-D Tensor populated with value list [4,5].

d = tf.add(matrix_a**2,matrix_b)
#Constant 1-D Tensor populated with value list [7,13].

result = tf.multiply(d,matrix_c)
#Constant 1-D Tensor populated with value list [28,65].
with tf.Session() as sess:
    print (sess.run(result))


#Constant 2-D tensor populated with scalar value 2.
tensor1 = tf.constant(2, shape=[2, 2])
#Constant 2-D Tensor populated with value[[2,2][2,2]].
tensor2 = tf.constant(3, shape=[2, 2])
#Constant 2-D Tensor populated with value[[3,3][3,3]].
tensor3 = tf.constant(5, shape=[2, 4])
#Constant 2-D Tensor populated with value[[5,5,5,5,5] [5,5,5,5,5]].
T1 = tf.add(tensor1**2,tensor2)
#Constant 2-D Tensor populated with value[[7,7][7,7]].
T2 = tf.matmul(T1,tensor3)
#Here the matrix shapes are different, use tf.matmul
#Constant 2-D Tensor populated with value[[70,70,70,70,70][70,70,70,70,70].
with tf.Session() as sess:
    print (sess.run(T2))
