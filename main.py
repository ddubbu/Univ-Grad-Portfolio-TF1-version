
from wav_tools import *
from models.after_train_model import *
#import numpy as np


############ 오디오 녹음 ############

FILE_NAME = "dev_file"
RECORD_FILE_PATH = record(FILE_NAME)

# RECORD_FILE_PATH = "./data_record/dev_file.wav"

############ Keyword Spotting ############
feature1 = []  # list type != numpy
feature1.append(feature_mfcc(RECORD_FILE_PATH).T)
f1_shape = np.shape(feature1)

# must be [1, 1723, 24] = X.shape
# so, check
if f1_shape != (1, 1723, 24):
    print("KWS Model input shape Mis-Matching!!!!")
    print("feature1.shape :", f1_shape)
    exit()

ex_FILE_PATH = KWS(RECORD_FILE_PATH, feature1)
print(ex_FILE_PATH, "\n CLEAR!")
exit()

# # plot_mfcc(mfcc)
#

# # after extraction
# Extraction_FILE_NAME = "data_record/test.wav" # extraction
# feature2 = feature_mfcc(Extraction_FILE_NAME)
# prediction = Speaker_Verification(feature2)
#
# ############ Feature Extraction ############
#
# mfcc = feature_mfcc(RECORD_FILE_PATH)




