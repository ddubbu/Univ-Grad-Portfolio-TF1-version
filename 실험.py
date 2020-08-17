# MFCC 랑 비교 필요..
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=10)  # (128, 44)
log_S = librosa.amplitude_to_db(S)
plt.figure(1)
librosa.display.specshow(log_S)
plt.title('melspecgro -> db scale, n_mels=10')
plt.colorbar(format='%+02.0f dB')
plt.show()


def feature_mfcc(RECORD_FILE_NAME):

    # 조정할 수 있는 건 다 적어보자.
    hop_length=128
    n_mfcc = 13
    n_mels = 40
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
    S = librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=n_mels)
    mfcc = librosa.feature.mfcc(S=S, sr=sr, hop_length=hop_length, fmin=fmin, fmax = fmax,
                                  n_fft= n_fft, n_mfcc=n_mfcc)  #  n_mels = n_mels
    return mfcc

# 다양한 feature
def various_feature():
    sig, sr = librosa.load(RECORD_FILE_NAME, sr=sr)  # 직접 지정할래
        # 조정할 수 있는 건 다 적어보자.
    n_mels = 128

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

    plt.figure(figsize=(10, 50))
    plt.subplot(5, 1, 1)
    librosa.display.waveplot(y=sig, sr=sr)
    plt.title("wave form")
    plt.tight_layout(pad=1)

    plt.subplot(5, 1, 2)
    S = librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=n_mels)
    librosa.display.specshow(S, x_axis='time', y_axis='mel')
    plt.title("mel-spectrogram")
    plt.colorbar()
    plt.tight_layout(pad=1)

    plt.subplot(5, 1, 3)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(log_S,
                             x_axis='time', y_axis='mel')
    plt.title("log scaled mel-spectrogram")
    plt.colorbar(format='%.1f dB')
    plt.tight_layout(pad=1)

    plt.subplot(5, 1, 4)  # 좀 이상해..
    min_level_db = -100
    def _normalize(S):
        return np.clip((S-min_level_db) / -min_level_db, 0, 1)
    norm_log_S = _normalize(log_S)
    librosa.display.specshow(norm_log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title("normalized, log scaled mel-spectrogram")
    plt.colorbar(format='%.1f dB')
    plt.tight_layout(pad=1)

    plt.subplot(5, 1, 5)
    mfcc = librosa.feature.mfcc(S=norm_log_S, sr=sr, hop_length=hop_length, fmin=fmin, fmax = fmax,
                                  n_fft= n_fft, n_mfcc=n_mfcc)  #  n_mels = n_mels
    librosa.display.specshow(mfcc, x_axis='time')
    plt.title("MFCC")
    plt.colorbar(format='%s')  # 단위를 모르겠음.
    plt.tight_layout(pad=1)

    plt.show()
