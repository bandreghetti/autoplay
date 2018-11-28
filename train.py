#!/usr/bin/env python3

import gym
import time
from games import gameList
import agents
import numpy as np
import sys

FPS = 20.0
framePeriod = 1.0/FPS

def main():
    render_opt = False
    if len(sys.argv) > 2:
        topology = sys.argv[1]
        trainTime = int(sys.argv[2])
        if len(sys.argv) > 3:
            if sys.argv[3] == "render":
                render_opt = True
    else:
        print("Usage: ./train.py <topology> <train-hours> [render]")
        print("Example: ./train.py MLP 24 render")
        exit()

    env = gym.make(gameList[0])

    if topology == "MLP":
        agent = agents.MLP(env)
    else:
        print("Available topologies:")
        print("  - MLP")
        exit()

    env.reset()
    observation, _, done, _ = env.step(env.action_space.sample())
    env.render()
    lastRender = time.time()

    nFrames = trainTime*3600*FPS
    if render_opt:
        print("Rendering")
    print("Training for {} hour(s) ({} frames)".format(trainTime, int(nFrames)))
    framesPlayed = 0
    episodesPlayed = 0
    while framesPlayed < nFrames:
        if render_opt == True:
            currPeriod = time.time() - lastRender

        if render_opt == False or currPeriod > framePeriod:
            if render_opt:
                lastRender = time.time()
                print('FPS: {}'.format(np.round(1/currPeriod)))
                env.render()

            framesPlayed += 1
            action = agent.action(observation)
            print(action)
            observation, _, done, _ = env.step(action)

            if done:
                episodesPlayed += 1
                env.reset()

    env.close()

main()
