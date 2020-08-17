import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from wav_tools import *


# 자르기
from pydub import AudioSegment
######################## TEST #####################

sess = tf.Session()
saver = tf.train.import_meta_graph('./models/GRU_model_train_epoch1300.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models/'))

graph = tf.get_default_graph()

# alltensors = [ tensor.name for tensor in tf.get_default_graph().get_operations()]
# # print(tensors)
# for tnsr in alltensors:
#     print(tnsr)
# exit()

X = graph.get_tensor_by_name("X:0")
Y = graph.get_tensor_by_name("Y:0")
outputs = graph.get_tensor_by_name("fc_outputs/BiasAdd:0")  # my 추측 ,fc_outputs
loss = graph.get_tensor_by_name("MyLoss:0")

opt = graph.get_operation_by_name("MyOpt")  # maybe ? oo

# keep_prob

X_test = np.load('./XY_test/test_X.npy')
Y_test = np.load('./XY_test/test_Y.npy')

batch_size = Y_test.shape[0]


for batch in range(0, batch_size):
    TEST_FILE_PATH = './XY_test/wav/test' + str(batch) + '.wav'


    _loss, _outputs = sess.run([loss, outputs], feed_dict={X: X_test[batch:batch + 1],
                                                           Y: Y_test[batch:batch + 1]})  # , keep_prob:0.8})
    result = _outputs[0]   # (1723, 1) col format for plotting
    print("test", batch, ", loss:", _loss)

    # plt.figure()
    # # plt.subplot(2, 1, 1)
    # # mfcc = X_test[batch].T
    # # librosa.display.specshow(mfcc, x_axis='time')  # y 축은 모르겠당 y_coords=13
    # # plt.subplot(2, 1, 2)
    # plt.plot(result)
    # plt.show()

    # Min-Max Normalize
    max = np.max(result)  # , axis=1
    min = np.min(result)
    result = (result - min) / (max - min)

    peak_y = np.max(result)
    peak_x = np.where(result == peak_y)[0][0]

    # 1s = 172 T_y step
    # where do you cut ?
    T_y = 1723
    mark_length = 300  # 172
    left_step = int(mark_length * 0.3)
    right_step = mark_length - left_step
    left_x = peak_x - left_step
    right_x = peak_x + right_step

    scatter_x = [peak_x]  # , peak_x, left_x, right_x
    scatter_y = [peak_y]  # , 0, 0, 0
    # plt.figure()
    # plt.ylim([0, 1.1])
    # plt.plot(result)
    # plt.scatter(scatter_x, scatter_y, c=['r'])  # , 'r', 'g', 'g'
    # plt.axvline(x=left_x, color='r', linestyle='-')
    # plt.axvline(x=right_x, color='r', linestyle='-')
    # plt.show()

    # pyaudio peak and restore
    # pydub does thing in milliseconds
    segment_left_t = int(left_x * 10000.0 / T_y) # + 2500
    segment_right_t = int(right_x * 10000.0 / T_y)  # + 2500

    print("time: ", segment_left_t, "~", segment_right_t)

    sig = AudioSegment.from_wav(TEST_FILE_PATH)

    ex_Keyword = sig[segment_left_t : segment_right_t + 1]

    ex_FILE_PATH = "./extract_keyword/" + str(batch) + ".wav"
    file_handle = ex_Keyword.export(ex_FILE_PATH, format="wav")