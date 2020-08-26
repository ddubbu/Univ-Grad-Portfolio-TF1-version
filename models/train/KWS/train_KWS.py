# from wav_tools import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


# after generate_train_data

sequence_length = 1723  # = Tx, Ty
n_mfcc = 24
# keep_prob = tf.placeholder(tf.float32, 1, name="keep_prob")
X = tf.placeholder(tf.float32, [None, sequence_length, n_mfcc], name='X')
Y = tf.placeholder(tf.float32, [None, sequence_length, 1], name='Y')

hidden_size = 128
n_layers = 1
learning_rate = 0.001
keep_prob = tf.placeholder(tf.float32, name="keep_prob")



layers = [tf.nn.rnn_cell.GRUCell(num_units=hidden_size,activation='relu')
                        for layer in range(n_layers)]
layers_drop = [tf.nn.rnn_cell.DropoutWrapper(
                    layer, state_keep_prob= keep_prob)
                    for layer in layers]

multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers)
outputs, _state = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

# outputs.shape = (?, sequence_length, hidden_size) = (?, 1723, 128)

outputs = tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=1  # 1로 줄이고..
                                            , activation_fn=None)  # tf.nn.relu 얘가 default 라니..
# outputs = tf.layers.dense(outputs, 1, name="fc_outputs")

# length of sequence to
outputs_ = tf.reshape(outputs, shape=[1, 1, sequence_length])
W = tf.get_variable(name="last_W", shape=[1, sequence_length, sequence_length],
                    initializer= xavier_init(sequence_length,sequence_length))  #( tf.random_normal([1, sequence_length, sequence_length]))
predict = tf.nn.conv1d(outputs_, filters=W, padding="VALID",)  # 1D 쉽다
predict = tf.reshape(predict, shape=[1, -1, 1], name="predict")


loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=predict, labels=Y)
loss = tf.reduce_mean(loss, name="loss")

tf.summary.scalar("train_loss", loss)

opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, name="MyOpt")




merged = tf.summary.merge_all()



#################### Train #######################
with tf.Session() as sess:
    # ##########################
    # # 여기 주석풀면 아래는 주석, 위에 모델 코드도 주석 달아버려~
    # saver = tf.train.import_meta_graph('./models/GRU_model_train_batch_2000/epoch-0.meta')
    # saver.restore(sess, tf.train.latest_checkpoint('./models/GRU_model_train_batch_2000/'))
    # graph = tf.get_default_graph()
    #
    # X = graph.get_tensor_by_name("X:0")
    # Y = graph.get_tensor_by_name("Y:0")
    # loss = graph.get_tensor_by_name("loss:0")
    # opt = graph.get_operation_by_name("MyOpt")
    # keep_prob = graph.get_tensor_by_name("keep_prob:0")
    # # outputs = graph.get_tensor_by_name("predict:0")
    #
    # tf.summary.scalar('train_loss', loss)
    # merged = tf.summary.merge_all()
    # ###########################


    sess.run(tf.global_variables_initializer())  # 위에 주석풀면 여긴 주석
    saver = tf.train.Saver()  # 위에 주석풀면 여긴 주석

    # training data train0.npy ~ train19.npy : 2000개 -> 1 epoch
    # test data train20.npy ~ train21.npy : 200 개

    train_epoch = 25
    total_batch_size = 100  # 1
    total_train_data_size = total_batch_size*20

    # train loss
    train_avg_losses = []
    test_avg_losses = []

    log_path = './log/train_{}_epoch_{}/'.format(total_train_data_size, train_epoch)
    os.makedirs(log_path, exist_ok=True)
    for epoch in range(train_epoch):  #

        # log_path += str(epoch)
        log_path2 = log_path + 'epoch'+ str(epoch)
        os.makedirs(log_path2, exist_ok = True)
        train_writer = tf.summary.FileWriter(log_path2, sess.graph)

        # To see Overfitting : plotting
        fig1 = plt.figure()
        plt.xlim(0, train_epoch)
        # plt.ylim(0, 1)  # 1로 봐도 될까? ★ -> 확대 가능하니깐, 노 걱정.
        plt.xlabel("epoch")
        plt.ylabel("avg_loss")

        # plt.hold(True)
        # plt.show()

        # train loss
        train_avg_loss = 0.
        for i in range(0, 20):
            folder_path = './XY_train/train' + str(i)
            # load train data
            X_data = np.load(folder_path + '_X.npy')
            Y_data = np.load(folder_path + '_Y.npy')
            # print(X.shape, Y.shape)

            for batch in range(total_batch_size):
                _loss, _, summary = sess.run([loss, opt, merged],
                                                       feed_dict={X:X_data[batch:batch+1],
                                                                  Y:Y_data[batch:batch+1],
                                                                  keep_prob:0.8})
                train_avg_loss += _loss / (total_train_data_size)  # 나중에 어차피 전체 개수로 나눌거 지금 나눠서 더해주자.
                # print
                if(batch % 10 == 0):
                    print("epoch# :", epoch, "train", i*total_batch_size + batch, ", _loss:", _loss)

                train_writer.add_summary(summary, global_step=total_batch_size*i + batch)

        print("====== epoch# :", epoch, " ||2000/2000|| Training END ======")
        print("train avg_loss:", train_avg_loss)  # (i-1)*100 + batch
        train_avg_losses.append(train_avg_loss)


        # test loss
        total_test_data_size = total_batch_size*2
        for i in range(0, 2):

            folder_path = './XY_train/train' + str(20 + i)
            # load train data
            X_test_data = np.load(folder_path + '_X.npy')
            Y_test_data = np.load(folder_path + '_Y.npy')
            # print(X.shape, Y.shape)

            # test loss
            test_avg_loss = 0.

            for batch in range(100): # 1
                # opt 실행하지마.
                _loss = sess.run(loss, feed_dict={X: X_test_data[batch:batch + 1],
                                                  Y: Y_test_data[batch:batch + 1],
                                                  keep_prob: 1.0})  # , keep_prob:0.8})
                test_avg_loss += _loss / total_test_data_size
                # print("epoch# :", epoch, "test", i * 100 + batch, ", avg_loss:", test_avg_loss)  # (i-1)*100 + batch

        test_avg_losses.append(test_avg_loss)
        print("test avg_loss:", test_avg_loss)  # (i-1)*100 + batch


        saver.save(sess, './models/GRU_model_train_batch_2000/epoch', epoch)


        np.save("train_loss.npy", arr=train_avg_losses)
        np.save("test_loss.npy", arr=test_avg_losses)

        x_axis = np.linspace(1, (epoch+1) + 1, num=epoch+1)  # stop 미포함 인듯.
        # print(np.shape(x_axis))
        # print(np.shape(train_avg_losses))

        #x_axis = np.linspace(1, (epoch+1) + 1, num=len(train_avg_losses))
        train_line = plt.scatter(x_axis, train_avg_losses, c='r', marker='o')
        test_line = plt.scatter(x_axis, test_avg_losses, c='b', marker='x')
        plt.legend(handles=(train_line, test_line), labels=('train_avg_loss', 'test_avg_loss'))

        if epoch == train_epoch - 1:  #  학습 다 끝나면 loss plot 닫지 않고 기다리기
            plt.show()
            print("학습 종료")

        else:
            plt.show(block=False)
            plt.pause(3)
            plt.close(fig1)

        # Overfitting 3회 이상 자동 종료
        if epoch > 3:  # 연쇄적으로 test_avg_loss가 3번 커지는 현상이 보일 때
            if (test_avg_loss > test_avg_losses[epoch-1]
                and test_avg_losses[epoch-1] > test_avg_losses[epoch-2]
                and test_avg_losses[epoch-2] > test_avg_losses[epoch-3]):
                print("Overfitting at epoch = ", epoch)

                plt.show()
                print("Overfitting 학습 종료")

                exit()




            


train_writer.close()




