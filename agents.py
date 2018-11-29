from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Conv3D, MaxPooling3D, Flatten
import numpy as np
from skimage.transform import resize
import os

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

def load(game, topology):
    modelName = '{}_{}'.format(game, topology)
    if topology == "MLP":
        # load json and create model
        json_path = os.path.join(modelName, 'model.json')
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
            json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        weightsPath = os.path.join(modelName, 'weights.h5')
        model.load_weights(weightsPath)
        print("Loaded model from disk")
    else:
        print("Available topologies:")
        print("  - MLP")
        exit()
    return model

class MLP():
    def __init__(self, env):
        self.frame_size = (128, 128)
        observation_sample = env.observation_space.sample()
        observation_sample = resize(observation_sample, self.frame_size, anti_aliasing=True, mode='constant')
        self.input_size = observation_sample.size
        self.output_size = env.action_space.n

        self.batch_size = 32
        self.sample_idx = 0

        self.gamma = 0.99
        self.epsilon = 1

        self.batch_inputs = np.zeros((self.batch_size, self.input_size))
        self.batch_targets = np.zeros((self.batch_size, self.output_size))

        self.history_x = [0]
        self.history_y = [0]

        # Configure model
        self.model = Sequential()
        self.model.add(Dense(units=512, activation='sigmoid', kernel_initializer="uniform", input_dim=self.input_size))
        self.model.add(Dense(units=64, activation='sigmoid', kernel_initializer="uniform"))
        self.model.add(Dense(units=self.output_size, activation='softmax', kernel_initializer="uniform"))
        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    def getWeights(self):
        return self.model.get_weights()

    def setWeights(self, weights):
        self.model.set_weights(weights)

    def Q(self, observation):
        observation = resize(observation, self.frame_size, anti_aliasing=True, mode='constant')
        flatScreen = observation.reshape(1, -1)
        output = self.model.predict(flatScreen)
        return output

    def action(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.output_size)

        predicted_rewards = self.Q(observation)
        return np.argmax(predicted_rewards)

    def train(self, observation, new_observation, reward, done):
        observation_resized = resize(observation, self.frame_size, anti_aliasing=True, mode='constant')
        self.batch_inputs[self.sample_idx] = observation_resized.flatten()
        target = self.Q(observation)[0]
        action = np.argmax(target)
        if done:
            target[action] = reward
        else:
            new_observation = resize(new_observation, self.frame_size, anti_aliasing=True, mode='constant')
            Q_new = self.model.predict(new_observation.reshape(1, -1))
            target[action] = reward + self.gamma * np.max(Q_new)
        self.batch_targets[self.sample_idx] = target

        self.sample_idx += 1

        if self.sample_idx >= self.batch_size:
            # print('Epsilon: {}'.format(self.epsilon))
            self.model.train_on_batch(self.batch_inputs, self.batch_targets)
            self.sample_idx = 0
            if self.epsilon > 0.2:
                self.epsilon = 0.99*self.epsilon

    def add_history(self, time, reward):
        print('After training for {0:.2f} hours, got {1} reward in the last episode'.format(time, reward))
        self.history_x.append(time)
        self.history_y.append(reward)

    def save(self, game):
        modelName = '{}_MLP'.format(game)
        os.makedirs(modelName, exist_ok=True)
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(os.path.join(modelName, 'model.json'), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(os.path.join(modelName, 'weights.h5'))
        np.save(os.path.join(modelName, 'history_x.npy'), self.history_x)
        np.save(os.path.join(modelName, 'history_y.npy'), self.history_y)
