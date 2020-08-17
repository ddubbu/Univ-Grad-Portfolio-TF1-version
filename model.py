# 우선은 학습된 게 필요하잖아 그치?
# input output 크기를 모르는데... ^^
from keras.models import Model, load_model, Sequential
import numpy as np

def Speaker_Verification(feature):
    print()


def KWS(feature):
    model = load_model('./models/tr_model.h5')
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    # Training
    # X = np.load("XY_train/X_train.npy")  # X.shape = (30, 5511, 101) = (#example, Tx, n_freq)
    # Y = np.load("XY_train/Y_train.npy")  # Y.shape = (30, 1375, 1)
    # model.fit(X, Y, batch_size = 5, epochs=1)

    # TEST
    print("TEST")
    X_test = np.load("./XY_test/X_test.npy")
    Y_test = np.load("./XY_test/Y_test.npy")

    loss, acc = model.evaluate(X_test, Y_test)
    print("Test set accuracy = ", acc)