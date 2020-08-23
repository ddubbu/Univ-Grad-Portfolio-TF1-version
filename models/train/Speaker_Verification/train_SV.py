
from wav_tools_SV import *
import tensorflow as tf
import numpy as np
import pandas as pd

# PATH = './word'
# 
# # (24, 173)
# stop = feature_mfcc(PATH + "/up.wav")
# print(stop.shape)
# plot_mfcc(stop)
# 
# up = feature_mfcc(PATH + "/stop_not_mine2.wav")
# print(up.shape)
# plot_mfcc(up)

# 단어 30개 Classification NN 버전


tf.reset_default_graph()
tf.set_random_seed(777)
learning_rate = 0.001
training_epochs = 1000
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
sd = 1 / np.sqrt(13)  # standard deviation 표준편차(표본표준편차라 1/root(n))

num_mfcc = 24
num_class = 30
Tx = 173


X = tf.placeholder(tf.float32, [None, Tx, num_mfcc], name="X")
Y = tf.placeholder(tf.float32, [None, num_class], name="Y")

_X = tf.reshape(X, shape=[-1, Tx, num_mfcc, 1])  # (?, 173, 24, 1)


## convolution
CW1 = tf.Variable(tf.random_normal([5, 5, 1, 2], stddev=0.01))
C1 = tf.nn.conv2d(_X, CW1, strides=[1, 3, 3, 1], padding="VALID")
P1 = tf.nn.max_pool(C1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
print(P1.shape)  # (28*3*2)
# exit()

DNN_X = tf.reshape(P1, [-1, 28*3*2])

# ## Flatten -> DNN
# 1차 히든레이어
W1 = tf.get_variable("w1",
                     # tf.random_normal([216, 180], mean=0, stddev=sd),
                     shape=[28*3*2, 128],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([128], mean=0, stddev=sd), name="b1")
L1 = tf.nn.tanh(tf.matmul(DNN_X, W1) + b1)  # 1차 히든레이어는 'Relu' 함수를 쓴다.
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# # 4차 히든 레이어
# W4 = tf.get_variable("w4",
#                      # tf.random_normal([100, 50], mean=0, stddev=sd),
#                      shape=[256, 128],
#                      initializer=tf.contrib.layers.xavier_initializer())
# b4 = tf.Variable(tf.random_normal([128], mean=0, stddev=sd), name="b4")
# L4 = tf.nn.tanh(tf.matmul(L1, W4) + b4)  # 4차 히든레이어는 'Relu' 함수를 쓴다.
# L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# 최종 레이어
W8 = tf.get_variable("w8",
                     # tf.random_normal([50, num_class], mean=0, stddev=sd),
                     shape=[128, num_class],
                     initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([num_class], mean=0, stddev=sd), name="b8")
hypothesis = tf.add(tf.matmul(L1, W8), b8, name='hypothesis')
hypothesis = tf.nn.softmax(hypothesis)

cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y, name='myCost')
cost = tf.reduce_mean(cost)

tf.summary.scalar("cost", cost)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


merged = tf.summary.merge_all()

############################## Model 구축 끝.

X_train, X_test, Y_train, Y_test = np.load("./data.npy", allow_pickle=True)
X_train = X_train.astype("float")
X_test = X_test.astype("float")

print(num_class,"개의 클래스!!")
print("X_train :", np.shape(X_train))
print("Y_train :", np.shape(Y_train))
print("X_test :", np.shape(X_test))
print("Y_test :", np.shape(Y_test))


# num_example = len(X_train[0])
#
# # 짝수
# if (num_example % 5 == 0):
#     batch_size = 5
# elif (num_example % 4 == 0):
#     batch_size = 4
# elif (num_example % 3 == 0):
#     batch_size = 3
# elif (num_example % 2 == 0):
#     batch_size = 2
# else:
# batch_size = 1


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter("./log/", sess.graph)
    min_cost = 1000
    for batch in range(training_epochs):
        # avg_cost = 0
        batch_xs = X_train[batch:batch+1]
        batch_ys = Y_train[batch:batch+1]
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        cost_, _, summary = sess.run([cost, optimizer, merged], feed_dict=feed_dict)

            # if(epoch%10==0):
        print('Epoch:', '%04d' % (batch), 'cost =', '{:.9f}'.format(cost_))
        train_writer.add_summary(summary, global_step= batch)
        if (min_cost > cost_):
            min_cost = cost_
            saver.save(sess, './models/NN_cost_%.6f_epoch_%d'%(cost_, batch))

    is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 30개 Test data 중에 30개만
    # print("Accuracy: ", sess.run(accuracy, feed_dict={X: X_test[0:31], Y: Y_test[0:31], keep_prob: 1}))  # X_Test 개수 많을 거 같은데..

    print('Learning Finished!')




# path = "./word/stop.wav"
#
# X_test = feature_mfcc(path)
# '''
# 0 유인나
# 1 배철수
# 2 이재은
# 3 최일구
# 4 문재인 대통령
# '''
#
# class_stop =   # 번호 적기
#
# label = [0 for i in range(num_class)]
# label[class_stop] = 1  # class가 30개, one-hot vector 만드는 과정
# Y_test = []
# for i in range(len(X_test)):
#     Y_test.append(label)
#
# print(np.shape(X_test))
# print(np.shape(Y_test))
#
#
# print("predict")
# print(pd.value_counts(pd.Series(sess.run(tf.argmax(hypothesis, 1),
#                                     feed_dict={X: X_test, keep_prob:1}))))
#
# ## 여러개일 때 유용
# correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print("Accuracy: ", sess.run(accuracy, feed_dict={X: X_test, Y:Y_test, keep_prob:1}))