import os
import numpy as np
import matplotlib.pyplot as plt

import librosa, librosa.display
import pyaudio  # 마이크 사용
import wave  # 녹음한 데이터 wav 파일로 저장.
from pydub import AudioSegment  # vs librosa.load as np.array

from datetime import datetime  # 저장될 파일 이름


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # 비트레이트 설정 [bps]
CHUNK = int(RATE / 10)  # 버퍼 사이즈 1초당 44100비트레이트 이므로 100ms단위
RECORD_SECONDS = 10  # 녹음할 시간 설정
# WAVE_OUTPUT_FILENAME = "record.wav"

############ 1. 오디오 녹음 ############
# record 10s and store wav file
# Q. chunk, br과 sr 관계는 나중에 이해하자.
def record(FILENAME):
    print("record start!")
    # 파일명 조심 : 파일명에 콜론 들어가면 안됨
    now = datetime.today().strftime('%Y-%m-%d-%H-%M-%S-')
    WAVE_OUTPUT_FILENAME = "data_record/" + now + str(FILENAME) + ".wav"
    print(WAVE_OUTPUT_FILENAME)

    p = pyaudio.PyAudio()  # 오디오 객체 생성

    stream = p.open(format=FORMAT,  # 16비트 포맷
                    channels=CHANNELS, #  모노로 마이크 열기
                    rate=RATE, #비트레이트
                    input=True,
                    # input_device_index=1,
                    frames_per_buffer=CHUNK)
                      # CHUNK만큼 버퍼가 쌓인다.

    print("Start to record the audio.")

    frames = []  # 음성 데이터를 채우는 공간
    print_sec = 0
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        #지정한  100ms를 몇번 호출할 것인지 10 * 5 = 50  100ms 버퍼 50번채움 = 5초
        data = stream.read(CHUNK)
        frames.append(data)

        if i == RATE/CHUNK * print_sec :
            print("Recording.... ", print_sec + 1, "s")
            print_sec += 1

    print("Recording is finished.")

    stream.stop_stream() # 스트림닫기
    stream.close() # 스트림 종료
    p.terminate() # 오디오객체 종료

    # WAVE_OUTPUT_FILENAME의 파일을 열고 데이터를 쓴다.
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


    # 음성 wave plot
    spf = wave.open(WAVE_OUTPUT_FILENAME,'r')
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, dtype=np.int16)

    # 시간 흐름에 따른 그래프를 그리기 위한 부분
    Time = np.linspace(0,len(signal)/RATE, num=len(signal))

    fig1 = plt.figure()
    plt.title('Voice Signal Wave...')
    plt.plot(Time, signal)
    plt.show()
    plt.close(fig1)  # 닫아줘야하는 번거로움
    print("record end!!")

    return WAVE_OUTPUT_FILENAME

############ 2. Feature Extraction ############

def plot_mfcc(mfcc, RECORD_FILE_NAME=None):
    plt.figure()
    librosa.display.specshow(mfcc, x_axis='time')  # y 축은 모르겠당 y_coords=13
    # plt.plot(mfcc[1], mfcc[0])
    plt.title('MFCC')
    plt.xlabel('Time [s]')
    plt.ylabel('MFCC Coefficeints')
    plt.colorbar()
    plt.show()

def feature_mfcc(RECORD_FILE_NAME):

    # 조정할 수 있는 건 다 적어보자.
    hop_length=128
    n_mfcc = 24
    n_mels = 40
    n_fft = 512
    fmin = 0
    fmax = None
    sr = 16000

    # sr = 22050 = bitrate/2 -> Q. bitrate 와 어떤 관계?
    # Generate mfccs from a time series
    # t초당 sig.shape = (t*sr,)
    sig, sr = librosa.load(RECORD_FILE_NAME)  # , sr=sr

    # 만약, sr=16000, mfcc.shape = (n_mfcc,1251)
    #       sr=(default)22050, mfcc.shape = (n_mfcc, 1723)

    mfcc = librosa.feature.mfcc(y=sig, sr=sr, hop_length=hop_length, fmin=fmin, fmax = fmax,
                                  n_fft= n_fft, n_mfcc=n_mfcc)

    return mfcc


############ 3. Generate training data ############

# Load raw audio files for speech synthesis
def load_raw_audio():
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir("./raw_data/activates"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav("./raw_data/activates/"+filename)
            activates.append(activate)
    for filename in os.listdir("./raw_data/backgrounds"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav("./raw_data/backgrounds/"+filename)
            backgrounds.append(background)
    for filename in os.listdir("./raw_data/negatives"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav("./raw_data/negatives/"+filename)
            negatives.append(negative)
    return activates, negatives, backgrounds


def is_overlapping(new_segment, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    previous_segments could be many.
    """
    segment_start, segment_end = new_segment
    # Step 1: Initialize overlap as a "False" flag.
    overlap = False

    # Step 2: loop over the previous_segments start and end times.
    for previous_start, previous_end in previous_segments:  # it has a tuple data.
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True

    return overlap  # Bool

def get_random_time_segment(segment_ms):
    """
    find random position for len(segment_ms) in a 10,000 ms(=10s) audio clip.
    segment_ms -- the duration of the audio clip in ms
    """

    segment_start = np.random.randint(low=0, high=10000 - segment_ms)  # Make sure segment doesn't run past the 10sec background
    segment_end = segment_start + segment_ms - 1

    return (segment_start, segment_end)

def insert_audio_clip(background, new_audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step,
    ensuring that new audio segment does not overlap with existing segments.
    """

    # Get the duration of the audio clip in ms
    segment_ms = len(new_audio_clip)

    # Step 1: Pick a random time segment onto which to insert new audio clip.
    segment_time = get_random_time_segment(segment_ms)

    # Step 2: Check if overlap with previous segments.
    # ★★★  Warning : Infinite Loop ★★★
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    # Step 3: Add previous list not to overlap
    previous_segments.append(segment_time)

    # Step 4: Insert audio segments and background
    new_background = background.overlay(new_audio_clip, position=segment_time[0])

    return new_background, segment_time


def insert_ones(Ty, y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
    should be set to 1. By strictly we mean that the label of segment_start_y should be 0 while, the
    50 following labels should be ones.

    ★ ★ Q. Why?  50 labels??
        len(new_segment) 만큼 아님?
        그리고 왜 끝나는 지점부터 50 segment를 하냐고....


    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms

    Returns:
    y -- updated labels
    """

    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)  # = 100

    # Add 1 to the correct index in the background label (y)
    mark_length = 172
    for i in range(segment_end_y + 1, segment_end_y + 100 + 1):
        if i < Ty:
            y[0, i] = 1

    return y

def create_training_examples(Ty, num_new_data, folder_path):
    """
    It creates the whole training data
    and saves it in X.npy and Y.npy.
    """
    # Load audio segments
    activates, negatives, backgrounds = load_raw_audio()
    X = []
    Y = []
    for i in range(num_new_data):
        x, y = create_training_example(Ty, backgrounds, activates, negatives, "train"+str(i))
        X.append(x.T)
        Y.append(y.T)
    X = np.array(X)
    Y = np.array(Y)
    np.save(file=folder_path + '_X.npy', arr=X)
    np.save(file=folder_path + '_Y.npy', arr=Y)
    print("The end of making # of", num_new_data, "new training audios")



def create_training_example(Ty, backgrounds, activates, negatives, file_name_added):
    """
    Creates a training example with a given background, activates, and negatives.

    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """

    print("file num:", file_name_added)
    # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
    y = np.zeros((1, Ty))

    # Step 2: Initialize segment times as empty list (≈ 1 line)
    previous_segments = []

    # Step 3: # Select 0-1 random "background" audio clips from the entire list of "backgrounds" recordings
    random_indices = np.random.randint(len(backgrounds))
    background = backgrounds[random_indices]
    # Make background quieter
    background = background - 20  # ★ 왜?

    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = 1  # ★★★ 한개만 넣자  //np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]

    # Step 4: Loop over randomly selected "activate" clips and insert in background
    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        print("activate insert time :", segment_time)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(Ty, y, segment_start)  # 첫 지점으로
        # y = insert_ones(Ty, y, segment_end)


    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = 1  # ★★★ 얘도 한개만 넣자. //np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    # Step 5: Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives:
        # Insert the audio clip on the background
        background, n_segment_time = insert_audio_clip(background, random_negative, previous_segments)
        print("negative insert time :", n_segment_time)
    # Standardize the volume of the audio clip
    background = match_target_amplitude(background, -20.0)  # ★

    # Export new training example
    file_name = "./generate_data/" + str(file_name_added) + ".wav"
    file_handle = background.export(file_name, format="wav")
    print("File was saved in ./generate_data directory.")

    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = feature_mfcc(file_name)

    # print("=============\nx", x.shape)
    # print("=============\ny", y.shape, "\n", y)
    # plot_mfcc(x)
    # plt.figure()
    # plt.plot(y[0])
    # plt.show()

    return x, y

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)