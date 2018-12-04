#!/usr/bin/env python3

from games import gameList
import numpy as np
import sys
import os
from matplotlib import pyplot as plt

def main():
    if len(sys.argv) <= 1:
        print("Usage: ./plot.py <foldername>")
        print("Example: ./plot.py MLP")
        exit()

    foldername = sys.argv[1]

    history_x = np.load(os.path.join(foldername, 'history_x.npy'))
    history_y = np.load(os.path.join(foldername, 'history_y.npy'))

    plt.plot(history_x, history_y)
    plt.show()

main()
