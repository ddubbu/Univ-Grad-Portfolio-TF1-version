import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from wav_tools import *
from pydub import AudioSegment  # for Extraction


def KWS(RECORD_FILE_PATH, feature):
    # feature = (1, 1723, 24) numpy.array

    sess = tf.Session()
    saver = tf.train.import_meta_graph('./models/GRU_model_train_epoch3990.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./models/'))
    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name("X:0")
    # Y = graph.get_tensor_by_name("Y:0")
    outputs = graph.get_tensor_by_name("fc_outputs/BiasAdd:0")  # my 추측 ,fc_outputs

    _outputs = sess.run(outputs, feed_dict={X: feature})  # , keep_prob:0.8})

    result = _outputs[0]   # (1723, 1) col format for plotting

    # Un-Normalized-result
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

    # get Extraction duration
    T_y = 1723
    mark_length = 300  # 172로 하면, Tight 하긴한데 피아노 배경 Activate 말고 학습 굿
    left_step = int(mark_length * 0.3)
    left_x = peak_x - left_step if peak_x - left_step > 0 else 0
    right_x = left_x + mark_length

    scatter_x = [peak_x]  # , peak_x, left_x, right_x
    scatter_y = [peak_y]  # , 0, 0, 0
    plt.figure()
    plt.ylim([0, 1.1])
    plt.plot(result)
    plt.scatter(scatter_x, scatter_y, c=['r'])  # , 'r', 'g', 'g'
    plt.axvline(x=left_x, color='r', linestyle='-')
    plt.axvline(x=right_x, color='r', linestyle='-')
    plt.show()

    # pyaudio peak and restore
    # pydub does thing in milliseconds
    segment_left_t = int(left_x * 10000.0 / T_y) # + 2500
    segment_right_t = int(right_x * 10000.0 / T_y)  # + 2500

    print("time: ", segment_left_t, "~", segment_right_t)

    sig = AudioSegment.from_wav(RECORD_FILE_PATH)

    ex_Keyword = sig[segment_left_t : segment_right_t + 1]

    ex_FILE_PATH = "./data_record/extraction.wav"
    file_handle = ex_Keyword.export(ex_FILE_PATH, format="wav")

    return ex_FILE_PATH


def Speaker_Verification(feature):
    print()