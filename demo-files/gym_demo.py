#!/usr/bin/env python3

import gym
import time
import pynput.keyboard as kb
from joypad import Joypad
from games import gameList

FPS = 30.0
framePeriod = 1.0/30.0

joypad = Joypad()

def main():
    kbListener = kb.Listener(on_press=joypad.on_press, on_release=joypad.on_release)
    kbListener.start()

    env = gym.make(gameList[1])
    env.reset()
    env.render()
    lastRender = time.time()
    try:
        while(True):
            action = joypad.action
            if action == -1:
                break
            if time.time() - lastRender > framePeriod:
                print('{} {} {}'.format(joypad.actionString(), joypad.action, joypad.lastKey))
                _, _, done, _ = env.step(action)
                if done:
                    env.reset()
                env.render()
                lastRender = time.time()
    except KeyboardInterrupt:
        pass
    env.close()
    kbListener.stop()



main()
