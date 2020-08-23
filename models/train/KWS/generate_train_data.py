from wav_tools import *

Ty = 1723

# make train data
for i in range(10):  # *.np file
    np_folder_path = './XY_train/train'+str(i)
    create_training_examples(Ty, 100, np_folder_path)
    print("======= %d 개 완료======="%(i*100))


# ## make test data
# folder_path = './XY_test/test'
# create_training_examples(Ty, 1, folder_path)





# Extract 10s noise
# PATH = "C:/0. Git/speech_commands_v0.01"
#
# for filename in os.listdir(PATH + "/_backgrounds"):
#     print(PATH + "/_backgrounds/" + filename)
#     if filename.endswith("wav"):
#         background = AudioSegment.from_wav(PATH + "/_backgrounds/" + filename)
#         print(len(background))
#         background_10s = background[:10*1000]  # pydub은 milliseconds 단위 사용
#
#         file_handle = background_10s.export(PATH + "/_backgrounds/10s_" + filename , format="wav")

