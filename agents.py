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

class MLP():
    def __init__(self, env):
        observation_sample = env.observation_space.sample()

        input_size = observation_sample.size

        # Configure model
        self.model = Sequential()
        self.model.add(Dense(units=256, activation='sigmoid', kernel_initializer="uniform", input_dim=input_size))
        self.model.add(Dense(units=64, activation='sigmoid', kernel_initializer="uniform"))
        self.model.add(Dense(units=18, activation='softmax', kernel_initializer="uniform"))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def getWeights(self):
        return self.model.get_weights()

    def setWeights(self, weights):
        self.model.set_weights(weights)

    def Q(self, observation):
        flatScreen = observation.reshape(1, -1)
        output = self.model.predict(flatScreen)
        return output

    def action(self, observation):
        predicted_rewards = self.Q(observation)
        return np.argmax(predicted_rewards)
