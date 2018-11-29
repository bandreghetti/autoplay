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

            if topology == 'MLP':
                observation = resize(observation, (100, 100), anti_aliasing=True, mode='constant')
                flatScreen = observation.reshape(1, -1)
                output = model.predict(flatScreen)
                action = np.argmax(output)

            observation, _, done, _ = env.step(action)

            if done:
                env.reset()

    env.close()

main()
