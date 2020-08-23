# from wav_tools import *
import tensorflow as tf
import numpy as np

def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)



# after generate_train_data

Ty_sequence_length = 1723
Tx_sequence_length = 1723
n_mfcc = 24
# keep_prob = tf.placeholder(tf.float32, 1, name="keep_prob")
X = tf.placeholder(tf.float32, [None, Tx_sequence_length, n_mfcc], name='X')
Y = tf.placeholder(tf.float32, [None, Ty_sequence_length, 1], name='Y')

# # length of sequence to
# X_ = tf.reshape(X, shape=[1, n_mfcc, Tx_sequence_length])
# W = tf.get_variable(name="W", shape=[1, Tx_sequence_length, Ty_sequence_length],
#                     initializer=xavier_init(Tx_sequence_length, Ty_sequence_length))
#
# X_ = tf.nn.conv1d(X_, filters=W, padding="VALID")  # 1D 쉽다
# X_ = tf.reshape(X_, shape=[1, -1, n_mfcc])
#
# RNN_input = tf.contrib.layers.fully_connected(inputs=X_, num_outputs=1  # 1로 줄이고..
#                                             , activation_fn=None, scope="fc_inputs")  # tf.nn.relu 얘가 default 라니..

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

RNN_output = tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=1  # 1로 줄이고..
                                            , activation_fn=None, scope="fc_outputs")  # tf.nn.relu 얘가 default 라니..


loss = tf.pow((RNN_output - Y),2)
# loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=RNN_output, labels=Y)
loss = tf.reduce_mean(loss, name="MyLoss")

tf.summary.scalar('GRU_1_Tx_1372_lr_0.001_dropout0.8_label_negative', loss)

opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, name="MyOpt")



merged = tf.summary.merge_all()

#################### Train #######################
with tf.Session() as sess:
    ##########################
    # 여기 주석풀면 아래는 주석
    # # load meta graph
    # saver = tf.train.import_meta_graph('./models/GRU_model_train_epoch1300.meta')
    # saver.restore(sess, tf.train.latest_checkpoint('./models/'))
    #
    # graph = tf.get_default_graph()
    #
    # X = graph.get_tensor_by_name("X:0")
    # Y = graph.get_tensor_by_name("Y:0")
    # outputs = graph.get_tensor_by_name("fc_outputs/BiasAdd:0")  # my 추측 ,fc_outputs
    # loss = graph.get_tensor_by_name("MyLoss:0")
    # tf.summary.scalar('cost', loss)
    # merged = tf.summary.merge_all()
    #
    # opt = graph.get_operation_by_name("MyOpt")  # maybe ?
    ###########################


    sess.run(tf.global_variables_initializer())  # 위에 주석풀면 여긴 주석
    saver = tf.train.Saver()  # 위에 주석풀면 여긴 주석
    min_loss = 1000

    train_writer = tf.summary.FileWriter("./log/", sess.graph)


    for i in range(0, 50):
        folder_path = './XY_train/train' + str(i)
        # load train data
        X_data = np.load(folder_path + '_X.npy')
        Y_data = np.load(folder_path + '_Y.npy')
        # print(X.shape, Y.shape)

        for batch in range(100):
            _loss, _, _outputs, summary = sess.run([loss, opt, outputs, merged],
                                                   feed_dict={X:X_data[batch:batch+1],
                                                              Y:Y_data[batch:batch+1],
                                                              keep_prob:0.8})  #, keep_prob:0.8})
            print("train", i*100 + batch, ", loss:", _loss)  # (i-1)*100 + batch

            if _loss < 100:  # outlier 제외
                train_writer.add_summary(summary, global_step=i*100 + batch)
                # 혹은 lr 조절

            if(batch%10 == 0 and min_loss > _loss):
                saver.save(sess, './models/GRU_model_train_epoch'+str(i*100 + batch))

    train_writer.close()




