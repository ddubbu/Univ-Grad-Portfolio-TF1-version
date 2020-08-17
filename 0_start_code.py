import tensorflow as tf


import matplotlib.pyplot as plt

from wav_tools import *
from model import *


############ 오디오 녹음 ############
whoIs = "unknown"
# RECORD_FILE_NAME = record(whoIs)
RECORD_FILE_NAME = "data_record/test.wav"

############ Keyword Spotting ############
feature1 = feature_mfcc(RECORD_FILE_NAME)
# # plot_mfcc(mfcc)
#
Keyword_pos = KWS(feature1)
# # after extraction
# Extraction_FILE_NAME = "data_record/test.wav" # extraction
# feature2 = feature_mfcc(Extraction_FILE_NAME)
# prediction = Speaker_Verification(feature2)
#
# ############ Feature Extraction ############
#
# mfcc = feature_mfcc(RECORD_FILE_NAME)




