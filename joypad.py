from pynput.keyboard import Key
import numpy as np

EXIT          = -1
NOOP          = 0
FIRE          = 1
UP            = 2
RIGHT         = 3
LEFT          = 4
DOWN          = 5
UPRIGHT       = 6
UPLEFT        = 7
DOWNRIGHT     = 8
DOWNLEFT      = 9
UPFIRE        = 10
RIGHTFIRE     = 11
LEFTFIRE      = 12
DOWNFIRE      = 13
UPRIGHTFIRE   = 14
UPLEFTFIRE    = 15
DOWNRIGHTFIRE = 16
DOWNLEFTFIRE  = 17

class Joypad():
    def __init__(self, env):
        self.action2num = {}
        self.action_meanings = env.unwrapped.get_action_meanings()
        self.n_actions = env.action_space.n
        for idx, action in enumerate(self.action_meanings):
            self.action2num[action] = idx
        self.num2action = {v: k for k, v in self.action2num.items()}
        self.action = 0
        self.env_action = 0
        self.lastKey = None
        self.autoplay = False

    def actionString(self):
        return self.num2action[self.env_action]

    def actionOneHot(self):
        ret = np.zeros(self.n_actions, dtype=np.float64)
        np.put(ret, self.env_action, 1)
        return ret

    def actionButtons(self):
        bFIRE  = 0
        bUP    = 1
        bRIGHT = 2
        bLEFT  = 3
        bDOWN  = 4
        ret = np.zeros(5, dtype=np.float64)
        if(self.action == FIRE):
            np.put(ret, bFIRE, 1)
        elif self.action == UP:
            np.put(ret, bUP, 1)
        elif self.action == RIGHT:
            np.put(ret, bRIGHT, 1)
        elif self.action == LEFT:
            np.put(ret, bLEFT, 1)
        elif self.action == DOWN:
            np.put(ret, bDOWN, 1)
        elif self.action == UPRIGHT:
            np.put(ret, bUP, 1)
            np.put(ret, bRIGHT, 1)
        elif self.action == UPLEFT:
            np.put(ret, bUP, 1)
            np.put(ret, bLEFT, 1)
        elif self.action == DOWNRIGHT:
            np.put(ret, bDOWN, 1)
            np.put(ret, bRIGHT, 1)
        elif self.action == DOWNLEFT:
            np.put(ret, bDOWN, 1)
            np.put(ret, bLEFT, 1)
        elif self.action == UPFIRE:
            np.put(ret, bUP, 1)
            np.put(ret, bFIRE, 1)
        elif self.action == RIGHTFIRE:
            np.put(ret, bRIGHT, 1)
            np.put(ret, bFIRE, 1)
        elif self.action == LEFTFIRE:
            np.put(ret, bLEFT, 1)
            np.put(ret, bFIRE, 1)
        elif self.action == DOWNFIRE:
            np.put(ret, bDOWN, 1)
            np.put(ret, bFIRE, 1)
        elif self.action == UPRIGHTFIRE:
            np.put(ret, bUP, 1)
            np.put(ret, bRIGHT, 1)
            np.put(ret, bFIRE, 1)
        elif self.action == UPLEFTFIRE:
            np.put(ret, bUP, 1)
            np.put(ret, bLEFT, 1)
            np.put(ret, bFIRE, 1)
        elif self.action == DOWNRIGHTFIRE:
            np.put(ret, bDOWN, 1)
            np.put(ret, bRIGHT, 1)
            np.put(ret, bFIRE, 1)
        elif self.action == DOWNLEFTFIRE:
            np.put(ret, bDOWN, 1)
            np.put(ret, bLEFT, 1)
            np.put(ret, bFIRE, 1)

        return ret

    def on_press(self, key):
        self.lastKey = key
        if(key == Key.space):
            if  (self.action == NOOP):
                self.action = FIRE
            elif(self.action == UP):
                self.action = UPFIRE
            elif(self.action == RIGHT):
                self.action = RIGHTFIRE
            elif(self.action == LEFT):
                self.action = LEFTFIRE
            elif(self.action == DOWN):
                self.action = DOWNFIRE
            elif(self.action == UPRIGHT):
                self.action = UPRIGHTFIRE
            elif(self.action == UPLEFT):
                self.action = UPLEFTFIRE
            elif(self.action == DOWNRIGHT):
                self.action = DOWNRIGHTFIRE
            elif(self.action == DOWNLEFT):
                self.action = DOWNLEFTFIRE
        elif(key == Key.up):
            if  (self.action == NOOP):
                self.action = UP
            elif(self.action == FIRE):
                self.action = UPFIRE
            elif(self.action == RIGHT):
                self.action = UPRIGHT
            elif(self.action == LEFT):
                self.action = UPLEFT
            elif(self.action == RIGHTFIRE):
                self.action = UPRIGHTFIRE
            elif(self.action == LEFTFIRE):
                self.action = UPLEFTFIRE
        elif(key == Key.right):
            if  (self.action == NOOP):
                self.action = RIGHT
            elif(self.action == FIRE):
                self.action = RIGHTFIRE
            elif(self.action == UP):
                self.action = UPRIGHT
            elif(self.action == DOWN):
                self.action = DOWNRIGHT
            elif(self.action == UPFIRE):
                self.action = UPRIGHTFIRE
            elif(self.action == DOWNFIRE):
                self.action = DOWNRIGHTFIRE
        elif(key == Key.left):
            if  (self.action == NOOP):
                self.action = LEFT
            elif(self.action == FIRE):
                self.action = LEFTFIRE
            elif(self.action == UP):
                self.action = UPLEFT
            elif(self.action == DOWN):
                self.action = DOWNLEFT
            elif(self.action == UPFIRE):
                self.action = UPLEFTFIRE
            elif(self.action == DOWNFIRE):
                self.action = DOWNLEFTFIRE
        elif(key == Key.down):
            if  (self.action == NOOP):
                self.action = DOWN
            elif(self.action == FIRE):
                self.action = DOWNFIRE
            elif(self.action == RIGHT):
                self.action = DOWNRIGHT
            elif(self.action == LEFT):
                self.action = DOWNLEFT
            elif(self.action == RIGHTFIRE):
                self.action = DOWNRIGHTFIRE
            elif(self.action == LEFTFIRE):
                self.action = DOWNLEFTFIRE
        elif(key == Key.tab):
            self.autoplay = not self.autoplay
        elif(key == Key.esc):
            self.action = EXIT

    def on_release(self, key):
        if(key == Key.space):
            if  (self.action == FIRE):
                self.action = NOOP
            elif(self.action == UPFIRE):
                self.action = UP
            elif(self.action == RIGHTFIRE):
                self.action = RIGHT
            elif(self.action == LEFTFIRE):
                self.action = LEFT
            elif(self.action == DOWNFIRE):
                self.action = DOWN
            elif(self.action == UPRIGHTFIRE):
                self.action = UPRIGHT
            elif(self.action == UPLEFTFIRE):
                self.action = UPLEFT
            elif(self.action == DOWNRIGHTFIRE):
                self.action = DOWNRIGHT
            elif(self.action == DOWNLEFTFIRE):
                self.action = DOWNLEFT
        elif(key == Key.up):
            if  (self.action == UP):
                self.action = NOOP
            elif(self.action == UPFIRE):
                self.action = FIRE
            elif(self.action == UPRIGHT):
                self.action = RIGHT
            elif(self.action == UPLEFT):
                self.action = LEFT
            elif(self.action == UPRIGHTFIRE):
                self.action = RIGHTFIRE
            elif(self.action == UPLEFTFIRE):
                self.action = LEFTFIRE
        elif(key == Key.right):
            if  (self.action == RIGHT):
                self.action = NOOP
            elif(self.action == RIGHTFIRE):
                self.action = FIRE
            elif(self.action == UPRIGHT):
                self.action = UP
            elif(self.action == DOWNRIGHT):
                self.action = DOWN
            elif(self.action == UPRIGHTFIRE):
                self.action = UPFIRE
            elif(self.action == DOWNRIGHTFIRE):
                self.action = DOWNFIRE
        elif(key == Key.left):
            if  (self.action == LEFT):
                self.action = NOOP
            elif(self.action == LEFTFIRE):
                self.action = FIRE
            elif(self.action == UPLEFT):
                self.action = UP
            elif(self.action == DOWNLEFT):
                self.action = DOWN
            elif(self.action == UPLEFTFIRE):
                self.action = UPFIRE
            elif(self.action == DOWNLEFTFIRE):
                self.action = DOWNFIRE
        elif(key == Key.down):
            if  (self.action == DOWN):
                self.action = NOOP
            elif(self.action == DOWNFIRE):
                self.action = FIRE
            elif(self.action == DOWNRIGHT):
                self.action = RIGHT
            elif(self.action == DOWNLEFT):
                self.action = LEFT
            elif(self.action == DOWNRIGHTFIRE):
                self.action = RIGHTFIRE
            elif(self.action == DOWNLEFTFIRE):
                self.action = LEFTFIRE

    def set_env_action(self):
        if self.action == NOOP and "NOOP" in self.action2num:
            self.env_action = self.action2num["NOOP"]
        elif self.action == FIRE and "FIRE" in self.action2num:
            self.env_action = self.action2num["FIRE"]
        elif self.action == UP and "UP" in self.action2num:
            self.env_action = self.action2num["UP"]
        elif self.action == RIGHT and "RIGHT" in self.action2num:
            self.env_action = self.action2num["RIGHT"]
        elif self.action == LEFT and "LEFT" in self.action2num:
            self.env_action = self.action2num["LEFT"]
        elif self.action == DOWN and "DOWN" in self.action2num:
            self.env_action = self.action2num["DOWN"]
        elif self.action == UPRIGHT and "UPRIGHT" in self.action2num:
            self.env_action = self.action2num["UPRIGHT"]
        elif self.action == UPLEFT and "UPLEFT" in self.action2num:
            self.env_action = self.action2num["UPLEFT"]
        elif self.action == DOWNRIGHT and "DOWNRIGHT" in self.action2num:
            self.env_action = self.action2num["DOWNRIGHT"]
        elif self.action == DOWNLEFT and "DOWNLEFT" in self.action2num:
            self.env_action = self.action2num["DOWNLEFT"]
        elif self.action == UPFIRE and "UPFIRE" in self.action2num:
            self.env_action = self.action2num["UPFIRE"]
        elif self.action == RIGHTFIRE and "RIGHTFIRE" in self.action2num:
            self.env_action = self.action2num["RIGHTFIRE"]
        elif self.action == LEFTFIRE and "LEFTFIRE" in self.action2num:
            self.env_action = self.action2num["LEFTFIRE"]
        elif self.action == DOWNFIRE and "DOWNFIRE" in self.action2num:
            self.env_action = self.action2num["DOWNFIRE"]
        elif self.action == UPRIGHTFIRE and "UPRIGHTFIRE" in self.action2num:
            self.env_action = self.action2num["UPRIGHTFIRE"]
        elif self.action == UPLEFTFIRE and "UPLEFTFIRE" in self.action2num:
            self.env_action = self.action2num["UPLEFTFIRE"]
        elif self.action == DOWNRIGHTFIRE and "DOWNRIGHTFIRE" in self.action2num:
            self.env_action = self.action2num["DOWNRIGHTFIRE"]
        elif self.action == DOWNLEFTFIRE and "DOWNLEFTFIRE" in self.action2num:
            self.env_action = self.action2num["DOWNLEFTFIRE"]
        else:
            self.env_action = 0
