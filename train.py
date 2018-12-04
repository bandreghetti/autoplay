#!/usr/bin/env python3

import gym
import time
from games import gameList
import agents
import numpy as np
import sys

FPS = 20.0
framePeriod = 1.0/FPS
batch_size = 32
train_chance = 0.2

def main():
    render_opt = False
    if len(sys.argv) > 2:
        topology = sys.argv[1]
        trainTime = float(sys.argv[2])
        if len(sys.argv) > 3:
            if sys.argv[3] == "render":
                render_opt = True
    else:
        print("Usage: ./train.py <topology> <train-hours> [render]")
        print("Example: ./train.py MLP 24 render")
        exit()

    game = gameList[0]
    env = gym.make(game)

    if topology == "MLP":
        agent = agents.MLP(env)
    elif topology == "Conv":
        agent = agents.Conv(env)
    else:
        print("Available topologies:")
        print("  - MLP")
        exit()

    env.reset()
    observation, _, done, _ = env.step(env.action_space.sample())
    lastRender = time.time()

    nFrames = trainTime*3600*FPS
    print("Training for {} hour(s) ({} frames)".format(trainTime, int(nFrames)))
    framesPlayed = 0
    episodesPlayed = 0
    episodeFrames = 0
    total_reward = 0
    while framesPlayed < nFrames:
        if render_opt == True:
            currPeriod = time.time() - lastRender

        if render_opt == False or currPeriod > framePeriod:
            if render_opt:
                lastRender = time.time()
                print('FPS: {}'.format(np.round(1/currPeriod)))
                env.render()

            framesPlayed += 1
            episodeFrames += 1
            action = agent.action(observation)
            new_observation, reward, done, info_lives = env.step(action)
         
            """
            time_factor = 0.99 + 0.01*episodeFrames
         	
            lives=info_lives['ale.lives']
            reward = reward*time_factor*(lives/4)
            """
         
            total_reward += reward
            #print(info_lives)
            #print("Reward: " + str(reward))
            if np.random.rand() < train_chance:
                agent.train(observation, new_observation, reward, done)

            observation = new_observation

            if done:
                agent.add_history(float(framesPlayed)/(3600*FPS), total_reward)
                total_reward = 0
                episodesPlayed += 1
                episodeFrames = 0
                env.reset()

    env.close()
    agent.save(game)

main()
