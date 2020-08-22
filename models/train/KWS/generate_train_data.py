from wav_tools import *

# make train data
Ty = 1723

for i in range(50):  # *.np file
    np_folder_path = './XY_train/train'+str(i)
    create_training_examples(Ty, 100, np_folder_path)


# folder_path = './XY_test/test'
# create_training_examples(Ty, 10, folder_path)