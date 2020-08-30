from wav_tools_SV import *
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def minmax_Batch_size(X):
    minmax_X = []
    #print(X.shape.as_list())
    #exit()
    for i in range(len(X)):
        temp = X[i]
        scaler = MinMaxScaler()
        temp = scaler.fit_transform(temp)
        minmax_X.append(temp)

    return minmax_X

def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)



keep_prob = tf.placeholder(tf.float32, name="keep_prob")


num_mfcc = 24
num_class = 30
Tx = 173

X = tf.placeholder(tf.float32, [None, Tx, num_mfcc], name="X")
Y = tf.placeholder(tf.float32, [None, num_class], name="Y")


hidden_size = 32 # 32
n_layers = 2
learning_rate = 0.001
keep_prob = tf.placeholder(tf.float32, name="keep_prob")



layers = [tf.nn.rnn_cell.GRUCell(num_units=hidden_size,activation='relu')
                        for layer in range(n_layers)]
layers_drop = [tf.nn.rnn_cell.DropoutWrapper(
                    layer, state_keep_prob= keep_prob)
                    for layer in layers]

multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers)
outputs, _state = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

outputs = tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=1  # 1로 줄이고..
                                            , activation_fn=None)  # tf.nn.relu 얘가 default 라니..

# length of sequence to
outputs_ = tf.reshape(outputs, shape=[-1, 1, Tx])
W = tf.get_variable(name="last_W", shape=[1, Tx, num_class],
                    initializer= xavier_init(Tx, num_class))
predict = tf.nn.conv1d(outputs_, filters=W, padding="VALID",)  # 1D 쉽다
logits = tf.reshape(predict, shape=[-1, num_class], name="predict")

# print(logits.shape.as_list())
# exit()

hypothesis = tf.nn.softmax(logits, name="hypothesis")
cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
cost = tf.reduce_mean(cost)



optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, name="MyOpt")

# tf.summary.scalar("cost2", cost)
#merged = tf.summary.merge_all()

############################## Model 구축 끝.

X_train, X_test, Y_train, Y_test = np.load("./data.npy", allow_pickle=True)
X_train = X_train.astype("float")
X_test = X_test.astype("float")

print(num_class,"개의 클래스!!")
print("X_train :", np.shape(X_train))  # (27000, 173, 24)
print("Y_train :", np.shape(Y_train))
print("X_test :", np.shape(X_test))  # (3000, 173, 24)
print("Y_test :", np.shape(Y_test))


learning_rate = 0.001
training_epochs = 10  #100
batch_size = 5
total_data = len(X_train)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter("./log/", sess.graph)
    min_cost = 1000
    for epoch in range(training_epochs):
        print("============ epoch", epoch, "============")
        for batch in range(total_data):
            # avg_cost = 0

            # batch_xs = X_train[batch]  # X_train[batch:batch+1]
            # batch_ys = Y_train[batch:batch+1]
            #
            # #print(batch_xs)
            # #print("================")
            # scaler = MinMaxScaler()
            # batch_xs = scaler.fit_transform(batch_xs)
            # #print(batch_xs)
            #
            # minmax_batch_xs = [batch_xs]

            batch_xs = X_train[batch:batch + batch_size]
            batch_ys = Y_train[batch:batch + batch_size]
            minmax_batch_xs = minmax_Batch_size(batch_xs)


            feed_dict = {X: minmax_batch_xs , Y: batch_ys, keep_prob: 0.7}
            # batch 5개 cost_ 주의.
            cost_, _, h= sess.run([cost, optimizer, hypothesis], feed_dict=feed_dict)
            # summary merged
            # print(sess.run(batch_ys))
            # print(h)

            print('epoch:', epoch,'data: %04d' % (batch)*batch_size, 'cost = %.9f'%(cost_))
            #train_writer.add_summary(summary, global_step= batch)
            if (min_cost > cost_):
                min_cost = cost_
                saver.save(sess, './models/NN_epoch_%d_data_%d_cost_%.6f_'%(epoch, batch*batch_size, cost_))

    print('Learning Finished!')


    # test
    class_list = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left',
                  'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
                  'tree', 'two', 'up', 'wow', 'yes', 'zero']



    print("label idx: ", end='')
    print(np.argmax(Y_test, 1))
    print("predict idx: ", end='')

    minmax_X_test = minmax_Batch_size(X_test)

    minmax_X_test = minmax_X_test[0:5]
    print(sess.run(tf.argmax(hypothesis, 1), feed_dict={X: minmax_X_test, keep_prob: 1}))


    ## 여러개일 때 유용
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: minmax_X_test, Y: Y_test, keep_prob: 1}))





