
from wav_tools_SV import *
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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


# tf.reset_default_graph()
# tf.set_random_seed(777)

learning_rate = 0.001
training_epochs = 20
total_batch_size = 225
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
sd = 1 / np.sqrt(13)  # standard deviation 표준편차(표본표준편차라 1/root(n))

num_mfcc = 24
num_class = 30
Tx = 173


X = tf.placeholder(tf.float32, [None, Tx, num_mfcc], name="X")
Y = tf.placeholder(tf.float32, [None, num_class], name="Y")


_X = tf.reshape(X, shape=[-1, Tx, num_mfcc, 1])  # for 2D conv

CW1 = tf.Variable(tf.random_normal([3, 3, 1, 1], stddev=0.01))
C1 = tf.nn.conv2d(_X, CW1, strides=[1, 2, 2, 1], padding="SAME")
P1 = tf.nn.max_pool(C1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

# print(P1.shape.as_list())
# exit()
P1 = tf.reshape(P1, shape=[-1, 43*6])

# 1차 히든레이어
W1 = tf.get_variable("w1",
                     # tf.random_normal([216, 180], mean=0, stddev=sd),
                     shape=[43*6, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.ones([256])
L1 = tf.nn.relu(tf.matmul(_X, W1) + b1)  # 1차 히든레이어는 'Relu' 함수를 쓴다.
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# 2차 히든 레이어
W2 = tf.get_variable("w2",
                     # tf.random_normal([180, 150], mean=0, stddev=sd),
                     shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.ones([256])
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)  # 2차 히든레이어는 'Relu' 함수를 쓴다.
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)


# 4차 히든 레이어
W4 = tf.get_variable("w4",
                     # tf.random_normal([100, 50], mean=0, stddev=sd),
                     shape=[256, 128],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.ones([128])
L4 = tf.nn.relu(tf.matmul(L2, W4) + b4)  # 4차 히든레이어는 'Relu' 함수를 쓴다.
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# 5차 히든 레이어
W5 = tf.get_variable("w5",
                     # tf.random_normal([100, 50], mean=0, stddev=sd),
                     shape=[128, 128],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.ones([128])
L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)  # 5차 히든레이어는 'Relu' 함수를 쓴다.
L5 = tf.nn.dropout(L5, keep_prob=keep_prob)

# 6차 히든 레이어
W6 = tf.get_variable("w6",
                     # tf.random_normal([100, 50], mean=0, stddev=sd),
                     shape=[128, 128],
                     initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.ones([128])
L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)  # 6차 히든레이어는 'Relu' 함수를 쓴다.
L6 = tf.nn.dropout(L6, keep_prob=keep_prob)


# 최종 레이어
W8 = tf.get_variable("w8",
                     # tf.random_normal([50, num_output], mean=0, stddev=sd),
                     shape=[128, num_class],
                     initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.ones([num_class])


logits = tf.add(tf.matmul(L6, W8), b8)

hypothesis = tf.nn.softmax(logits)
cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='myCost')
cost = tf.reduce_mean(cost)



optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# tf.summary.scalar("cost2", cost)
#merged = tf.summary.merge_all()

############################## Model 구축 끝.

X_train, X_test, Y_train, Y_test = np.load("./data.npy", allow_pickle=True)
X_train = X_train.astype("float")
X_test = X_test.astype("float")

print(num_class,"개의 클래스!!")
print("X_train :", np.shape(X_train))  # (225, 173, 24)
print("Y_train :", np.shape(Y_train))
print("X_test :", np.shape(X_test))  # (75, 173, 24)
print("Y_test :", np.shape(Y_test))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter("./log/", sess.graph)
    min_cost = 1000
    for epoch in range(training_epochs):
        for batch in range(total_batch_size):
            # avg_cost = 0
            batch_xs = X_train[batch]  # X_train[batch:batch+1]
            batch_ys = Y_train[batch:batch+1]

            #print(batch_xs)
            #print("================")
            scaler = MinMaxScaler()
            batch_xs = scaler.fit_transform(batch_xs)
            #print(batch_xs)

            minmax_batch_xs = [batch_xs]


            feed_dict = {X: minmax_batch_xs , Y: batch_ys, keep_prob: 0.7}
            cost_, _, h= sess.run([cost, optimizer, hypothesis], feed_dict=feed_dict)
            # summary merged
            # print(sess.run(batch_ys))
            # print(h)
            print('Epoch:', '%04d' % (batch), 'cost = %.9f'%(cost_))
            #train_writer.add_summary(summary, global_step= batch)
            if (min_cost > cost_):
                min_cost = cost_
                saver.save(sess, './models/NN_cost_%.6f_epoch_%d'%(cost_, batch))

    # is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 30개 Test data 중에 30개만
    # print("Accuracy: ", sess.run(accuracy, feed_dict={X: X_test[0:31], Y: Y_test[0:31], keep_prob: 1}))  # X_Test 개수 많을 거 같은데..

    print('Learning Finished!')

    # test
    class_list = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left',
                  'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
                  'tree', 'two', 'up', 'wow', 'yes', 'zero']

    path = "./word/stop.wav"

    X_test = []
    Y_test = []

    mfcc = feature_mfcc(path)
    class_stop = class_list.index("stop")  # = 22 (python index 0부터) 번호 적기
    label = [0 for i in range(num_class)]
    label[class_stop] = 1  # class가 30개, one-hot vector 만드는 과정

    X_test.append(mfcc.T)

    for i in range(len(X_test)):
        Y_test.append(label)

    #print(np.shape(X_test))
    #print(np.shape(Y_test))

    print(sess.run(hypothesis, feed_dict={X: X_test, keep_prob: 1}))
    print("predict idx: ", end='')
    print(sess.run(tf.argmax(hypothesis, 1), feed_dict={X: X_test, keep_prob: 1}))
    # print(pd.value_counts(pd.Series(sess.run(tf.argmax(hypothesis, 1),
    #                                         feed_dict={X: X_test, keep_prob: 1}))))

    ## 여러개일 때 유용
    # correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: X_test, Y: Y_test, keep_prob: 1}))





