import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
X = tf.placeholder(tf.float32, [None, 784])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# variable learning rate
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.8 at training time
pkeep = tf.placeholder(tf.float32)
# step for variable learning rate
step = tf.placeholder(tf.int32)
L = 200
M = 100
N = 50
O = 10
# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values 0.1
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.ones([L]) / 10)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M]) / 10)
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.ones([N]) / 10)
W4 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B4 = tf.Variable(tf.ones([O]) / 10)
Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Y = tf.nn.softmax(Y4)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y4, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1])], 0)
allbiases = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1])], 0)

train_step = tf.train.GradientDescentOptimizer(0.003).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
fig = plt.figure()
aa_train = []
cc_train = []
ii_train = []
aa_test = []
cc_test = []
ii_test = []
for i in range(10000):
    batch_X, batch_Y = mnist.train.next_batch(100)

    if i % 100 == 0:
        a, c = sess.run([accuracy, cross_entropy],
                        feed_dict={X: batch_X, Y_: batch_Y})
        #print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))
        aa_train.append((a))
        cc_train.append((c))
        ii_train.append((i))
    if i % 100 == 0:
        a, c = sess.run([accuracy, cross_entropy],
                        feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        print(str(i) + ": ********* epoch " + str(
            i * 100 // mnist.train.images.shape[0] + 1) + " ********* test accuracy:" + str(a) + " test loss: " + str(
            c))
        aa_test.append((a))
        cc_test.append((c))
        ii_test.append((i))
    sess.run(train_step, {X: batch_X, Y_: batch_Y})
plt.plot(ii_train, cc_train, color='b', label='train loss')
plt.plot(ii_test, cc_test, color='r', label='test loss')
plt.legend(loc='upper right')
plt.title('Cross Entropy Loss')
plt.ylabel('Loss')
plt.ylim((0, 35))
plt.show()
plt.plot(ii_train, aa_train)
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.show()
