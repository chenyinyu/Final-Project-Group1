from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None,):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
prediction = add_layer(xs, 784, 10,  activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
aa_train = []
cc_train = []
ii_train = []
aa_test = []
cc_test = []
ii_test = []
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i % 100 == 0:
        c = sess.run([cross_entropy],
                        feed_dict={xs: batch_xs, ys: batch_ys})
        #print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))
        cc_train.append((c))
        ii_train.append((i))
    if i % 100 == 0:

        c = sess.run(cross_entropy, feed_dict={xs: mnist.test.images, ys: mnist.test.labels})
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
        cc_test.append((c))
        ii_test.append((i))
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
plt.plot(ii_train, cc_train, color='b', label='train loss')
plt.plot(ii_test, cc_test, color='r', label='test loss')
plt.legend(loc='upper right')
plt.ylim((0, 35))
plt.title('Single Layer NN Cross Entropy Loss')
plt.ylabel('Loss')
plt.show()