from wav_tools import *

# make train data
Ty = 1723

for i in range(50):  # *.np file
    np_folder_path = './XY_train/train'+str(i)
    create_training_examples(Ty, 100, np_folder_path)


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

