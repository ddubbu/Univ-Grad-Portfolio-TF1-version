from wav_tools import *
import tensorflow as tf


# after generate_train_data

sequence_length = 1723  # = Tx, Ty
n_mfcc = 24
# keep_prob = tf.placeholder(tf.float32, 1, name="keep_prob")
X = tf.placeholder(tf.float32, [None, sequence_length, n_mfcc], name='X')
Y = tf.placeholder(tf.float32, [None, sequence_length, 1], name='Y')

hidden_size = 128
gruCell = tf.nn.rnn_cell.GRUCell(hidden_size, activation='relu')

outputs, _state = tf.nn.dynamic_rnn(gruCell, X, dtype=tf.float32)
# outputs.shape = (?, sequence_length, hidden_size) = (?, 1723, 128)

outputs = tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=1  # 1로 줄이고..
                                            , activation_fn=None, scope="fc_outputs")  # tf.nn.relu 얘가 default 라니..


loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=Y)
loss = tf.reduce_mean(loss, name="MyLoss")

tf.summary.scalar('cost_rl=0.001', loss)

opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, name="MyOpt")



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


    for i in range(1, 41):
        folder_path = './XY_train/train' + str(i)
        # load train data
        X_data = np.load(folder_path + '_X.npy')
        Y_data = np.load(folder_path + '_Y.npy')
        # print(X.shape, Y.shape)

        for batch in range(100):
            _loss, _, _outputs, summary = sess.run([loss, opt, outputs, merged], feed_dict={X:X_data[batch:batch+1], Y:Y_data[batch:batch+1]})  #, keep_prob:0.8})
            print("train", (i-1)*100 + batch, ", loss:", _loss)  # (i-1)*100 + batch

            if _loss < 100:  # outlier 제외
                train_writer.add_summary(summary, global_step=(i-1)*100 + batch)
                # 혹은 lr 조절

            if(batch%10 == 0 and min_loss > _loss):
                saver.save(sess, './models/GRU_model_train_epoch'+str((i-1)*100 + batch))

    train_writer.close()




