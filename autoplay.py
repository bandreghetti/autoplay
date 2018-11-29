#!/usr/bin/env python3

import gym
import time
from games import gameList
import agents
import numpy as np
import sys
from skimage.transform import resize

FPS = 20.0
framePeriod = 1.0/FPS

def main():
    if len(sys.argv) <= 1:
        print("Usage: ./autoplay.py <topology>")
        print("Example: ./autoplay.py MLP")
        exit()

    topology = sys.argv[1]

    game = gameList[0]
    env = gym.make(game)

    model = agents.load(game, topology)

    env.reset()
    observation, _, done, _ = env.step(env.action_space.sample())
    lastRender = time.time()

    while True:
        currPeriod = time.time() - lastRender

        if currPeriod > framePeriod:
            lastRender = time.time()
            print('FPS: {}'.format(np.round(1/currPeriod)))
            env.render()

            observation = resize(observation, (128, 128), anti_aliasing=True, mode='constant')

            if np.random.rand() < 0.01:
                action = np.random.randint(env.action_space.n)
            else:
                if topology == 'MLP':
                    flatScreen = observation.reshape(1, -1)
                    output = model.predict(flatScreen)
                elif topology == 'Conv':
                    screen = observation.reshape((1, 128, 128, 3))
                    output = model.predict(screen)
                action = np.argmax(output)



            observation, _, done, _ = env.step(action)

            if done:
                env.reset()

    env.close()

main()
