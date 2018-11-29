from keras.models import Sequential, clone_model
from keras.layers import Dense, Conv3D, MaxPooling3D, Flatten
import numpy as np
from skimage.transform import resize

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
        observation_sample = resize(observation_sample, (100, 100), anti_aliasing=True)
        self.input_size = observation_sample.size
        self.output_size = 18

        self.batch_size = 32
        self.sample_idx = 0

        self.gamma = 0.95
        self.epsilon = 1

        self.batch_inputs = np.zeros((self.batch_size, self.input_size))
        self.batch_targets = np.zeros((self.batch_size, self.output_size))

        # Configure model
        self.model = Sequential()
        self.model.add(Dense(units=256, activation='sigmoid', kernel_initializer="uniform", input_dim=self.input_size))
        self.model.add(Dense(units=64, activation='sigmoid', kernel_initializer="uniform"))
        self.model.add(Dense(units=self.output_size, activation='softmax', kernel_initializer="uniform"))
        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    def getWeights(self):
        return self.model.get_weights()

    def setWeights(self, weights):
        self.model.set_weights(weights)

    def Q(self, observation):
        observation = resize(observation, (100, 100), anti_aliasing=True)
        flatScreen = observation.reshape(1, -1)
        output = self.model.predict(flatScreen)
        return output

    def action(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.output_size)

        predicted_rewards = self.Q(observation)
        return np.argmax(predicted_rewards)

    def train(self, observation, new_observation, reward, done):
        observation_resized = resize(observation, (100, 100), anti_aliasing=True)
        self.batch_inputs[self.sample_idx] = observation_resized.flatten()
        target = self.Q(observation)[0]
        action = np.argmax(target)
        if done:
            target[action] = reward
        else:
            new_observation = resize(new_observation, (100, 100), anti_aliasing=True)
            Q_new = self.model.predict(new_observation.reshape(1, -1))
            target[action] = reward + self.gamma * np.max(Q_new)
        self.batch_targets[self.sample_idx] = target

        self.sample_idx += 1

        if self.sample_idx >= self.batch_size:
            print('Epsilon: {}'.format(self.epsilon))
            self.model.train_on_batch(self.batch_inputs, self.batch_targets)
            self.sample_idx = 0
            if self.epsilon > 0.1:
                self.epsilon = 0.95*self.epsilon
