
# 라이브러리 필요한 것만.
import pandas as pd  # 이거 필요 없어지면 없애자.
import numpy as np
import matplotlib.pyplot as plt

import pyaudio  # 마이크를 사용
import wave  # 녹음한 데이터 wav 파일로 저장.
from datetime import datetime  # 저장될 파일 이름

import librosa  # feature : MFCC 추출
import librosa.display  # 직접적으로 명시 필요


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # 비트레이트 설정 [bps]
CHUNK = int(RATE / 10)  # 버퍼 사이즈 1초당 44100비트레이트 이므로 100ms단위
RECORD_SECONDS = 1  # 녹음할 시간 설정
# WAVE_OUTPUT_FILENAME = "record.wav"

############ 1. 오디오 녹음 ############
# record 10s and store wav file
# Q. chunk, br과 sr 관계는 나중에 이해하자.
def record(who):
    print("record start!")
    # 파일명 조심 : 파일명에 콜론 들어가면 안됨
    now = datetime.today().strftime('%Y-%m-%d-%H-%M-%S-')
    WAVE_OUTPUT_FILENAME = "data_record/" + now + str(who) + ".wav"
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

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        #지정한  100ms를 몇번 호출할 것인지 10 * 5 = 50  100ms 버퍼 50번채움 = 5초
        data = stream.read(CHUNK)
        frames.append(data)

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
    n_mels=128
    hop_length=128
    n_mfcc = 13
    n_fft = 512
    fmin = 0
    fmax = None
    sr = 16000

    # sr = 22050 = bitrate/2 -> Q. bitrate 와 어떤 관계?
    # Generate mfccs from a time series
    # t초당 sig.shape = (t*sr,)
    sig, sr = librosa.load(RECORD_FILE_NAME, sr=sr)  # 직접 지정할래

    # 만약, sr=16000, mfcc.shape = (n_mfcc,1251)
    #       sr=(default)22050, mfcc.shape = (n_mfcc, 1723)

    # 0.5s 정도 딜레이되잖아.
    # sig = sig[int(0.5*sr):]

    # S=pre-computed log-power mel spectrogram
    S = librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=n_mels)
    # log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(y=sig, sr=sr, hop_length=hop_length, fmin=fmin, fmax = fmax,
                                  n_fft= n_fft, n_mfcc=n_mfcc)

    return mfcc

def feature_specgram(RECORD_FILE_NAME):
    print()
    # plt.figure()
    #rate, data = wavfile.read(RECORD_FILE_NAME)
    # pxx, freqs, bins, im = plt.specgram(data[:], NFFT=200, Fs=8000, noverlap=120)

    # plt.show()