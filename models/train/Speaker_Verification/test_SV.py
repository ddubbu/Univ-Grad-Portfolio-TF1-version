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

num_class = 30


X_train, X_test, Y_train, Y_test = np.load("./data.npy", allow_pickle=True)
X_train = X_train.astype("float")
X_test = X_test.astype("float")

print(num_class, "개의 클래스!!")
print("X_train :", np.shape(X_train))  # (27000, 173, 24)
print("Y_train :", np.shape(Y_train))
print("X_test :", np.shape(X_test))  # (3000, 173, 24)
print("Y_test :", np.shape(Y_test))

# 저장하고 load 시간 줄이기~
train_npy = (X_train, Y_train)  #
test_npy = (X_test, Y_test)
np.save("./train.npy", train_npy)
np.save("./test.npy", test_npy)



with tf.Session() as sess:

    saver = tf.train.import_meta_graph('./models/NN_cost_0.000000_epoch_18.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./models/'))
    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    loss = graph.get_tensor_by_name("loss:0")
    opt = graph.get_operation_by_name("MyOpt")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    hypothesis = graph.get_tensor_by_name("hypothesis:0")


    # test
    class_list = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left',
                  'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
                  'tree', 'two', 'up', 'wow', 'yes', 'zero']

    ## my voice test
    # path = "./word/stop.wav"
    # X_test = []
    # Y_test = []
    #
    # mfcc = feature_mfcc(path)
    # class_stop = class_list.index("stop")  # = 22 (python index 0부터) 번호 적기
    # label = [0 for i in range(num_class)]
    # label[class_stop] = 1  # class가 30개, one-hot vector 만드는 과정
    #
    # X_test.append(mfcc.T)
    #
    # for i in range(len(X_test)):
    #     Y_test.append(label)
    #
    # #print(np.shape(X_test))
    # #print(np.shape(Y_test))

    # print(sess.run(hypothesis, feed_dict={X: X_test, keep_prob: 1}))


    print("label idx: ", end='')
    print(np.argmax(Y_test, 1))
    print("predict idx: ", end='')


    minmax_X_test = minmax_Batch_size(X_test)  # ★ 3000개 test 어케 하지?



    minmax_X_test = minmax_X_test[0:5]
    print(sess.run(tf.argmax(hypothesis, 1), feed_dict={X: minmax_X_test, keep_prob: 1}))


    ## 여러개일 때 유용
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: minmax_X_test, Y: Y_test, keep_prob: 1}))
