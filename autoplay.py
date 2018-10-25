#!/usr/bin/env python3

import gym
import time
import pynput.keyboard as kb
from joypad import Joypad
from games import gameList
from agents import MultilayerPerceptron
import numpy as np

FPS = 20.0
framePeriod = 1.0/FPS

joypad = Joypad()

def main():
    kbListener = kb.Listener(on_press=joypad.on_press, on_release=joypad.on_release)
    kbListener.start()

    env = gym.make(gameList[0])
    agent = MultilayerPerceptron(env.observation_space)
    
    env.reset()
    observation, _, done, _ = env.step(env.action_space.sample())
    env.render()
    lastRender = time.time()
    try:
        while(True):
            action = joypad.action
            if action == -1:
                break
            currPeriod = time.time() - lastRender
            if currPeriod > framePeriod:
                lastRender = time.time()
                observation, _, done, _ = env.step(action)
                agent.train(observation, joypad.actionButtons())
                print('{} Autoplay:{}, FPS:{}'.format(joypad.actionButtons(),
                                                 joypad.autoplay,
                                                 np.round(1/currPeriod)))
                print(agent.action(observation))
                if done:
                    env.reset()
                env.render()
    except KeyboardInterrupt:
        pass
    env.close()
    kbListener.stop()

main()