from keras.models import Sequential, clone_model
from keras.layers import Dense, Conv3D, MaxPooling3D, Flatten
import numpy as np
from threading import Thread

bNOOP          = np.array([0, 0, 0, 0, 0])
bFIRE          = np.array([1, 0, 0, 0, 0])
bUP            = np.array([0, 1, 0, 0, 0])
bRIGHT         = np.array([0, 0, 1, 0, 0])
bLEFT          = np.array([0, 0, 0, 1, 0])
bDOWN          = np.array([0, 0, 0, 0, 1])
bUPRIGHT       = np.array([0, 1, 1, 0, 0])
bUPLEFT        = np.array([0, 1, 0, 1, 0])
bDOWNRIGHT     = np.array([0, 0, 1, 0, 1])
bDOWNLEFT      = np.array([0, 0, 0, 1, 1])
bUPFIRE        = np.array([1, 1, 0, 0, 0])
bRIGHTFIRE     = np.array([1, 0, 1, 0, 0])
bLEFTFIRE      = np.array([1, 0, 0, 1, 0])
bDOWNFIRE      = np.array([1, 0, 0, 0, 1])
bUPRIGHTFIRE   = np.array([1, 1, 1, 0, 0])
bUPLEFTFIRE    = np.array([1, 1, 0, 1, 0])
bDOWNRIGHTFIRE = np.array([1, 0, 1, 0, 1])
bDOWNLEFTFIRE  = np.array([1, 0, 0, 1, 1])

class MultilayerPerceptron():
    def __init__(self, networkShape):
        # Configure model
        self.model = Sequential()
        self.model.add(Dense(units=networkShape[1], activation='sigmoid', kernel_initializer="uniform", input_dim=networkShape[0]))
        self.model.add(Dense(units=networkShape[2], activation='sigmoid', kernel_initializer="uniform"))
        self.model.add(Dense(units=networkShape[3], activation='sigmoid', kernel_initializer="uniform"))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def getWeights(self):
        return self.model.get_weights()

    def setWeights(self, weights):
        self.model.set_weights(weights)

    def actionButtons(self, observation):
        flatScreen = np.expand_dims(observation.flatten(), 0)
        output = self.model.predict(flatScreen)[0]
        buttons = np.greater(output, 0.5).astype(np.float64)
        return buttons

    def action(self, observation):
        buttons = self.actionButtons(observation)
        if np.all(buttons == bNOOP):
            return 0
        elif np.all(buttons == bFIRE):
            return 1
        elif np.all(buttons == bUP):
            return 2
        elif np.all(buttons == bRIGHT):
            return 3
        elif np.all(buttons == bLEFT):
            return 4
        elif np.all(buttons == bDOWN):
            return 5
        elif np.all(buttons == bUPRIGHT):
            return 6
        elif np.all(buttons == bUPLEFT):
            return 7
        elif np.all(buttons == bDOWNRIGHT):
            return 8
        elif np.all(buttons == bDOWNLEFT):
            return 9
        elif np.all(buttons == bUPFIRE):
            return 10
        elif np.all(buttons == bRIGHTFIRE):
            return 11
        elif np.all(buttons == bLEFTFIRE):
            return 12
        elif np.all(buttons == bDOWNFIRE):
            return 13
        elif np.all(buttons == bUPRIGHTFIRE):
            return 14
        elif np.all(buttons == bUPLEFTFIRE):
            return 15
        elif np.all(buttons == bDOWNRIGHTFIRE):
            return 16
        elif np.all(buttons == bDOWNLEFTFIRE):
            return 17
        else:
            return 0
