#!/usr/bin/env python3

import gym
import time
from games import gameList
from agents import MultilayerPerceptron
import numpy as np
from BADGE import Lineage

FPS = 30.0
framePeriod = 1.0/FPS
nGens = 100
render = True
nAgents = 10
networkShape = [512, 64, 5]

def main():
    env = gym.make(gameList[0])
    env.reset()
    observation, _, done, _ = env.step(env.action_space.sample())
    env.render()
    lastRender = time.time()
    lin = Lineage(nAgents, [observation.size] + networkShape)
    agent = MultilayerPerceptron([observation.size] + networkShape)
    try:
        for _ in range(nGens):
            for creature in lin.current_gen.genome:
                agent.setWeights(creature.Theta)
                while not done:
                    if not render or time.time() - lastRender > framePeriod:
                        lastRender = time.time()
                        action = agent.action(observation)
                        observation, reward, done, _ = env.step(action)
                        if render:
                            env.render()
                creature.updateFitness(reward)
            lin.nextGeneration()

    except KeyboardInterrupt:
        pass
    env.close()

main()
