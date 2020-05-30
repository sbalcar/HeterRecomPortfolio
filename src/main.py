#!/usr/bin/python3

import time
import sys

import random
import numpy as np

from jobs.ml1m import ml1m #function

def main():

  np.random.seed(42)
  random.seed(42)

  start = time.time()

  #ml1m("stc08", 50, 1)
  #ml1m("stc08", 60, 1)
  #ml1m("stc08", 70, 1)
  #ml1m("stc08", 80, 1)
  ml1m("stc08", 90, 1)

  #ml1m("stc08", 50, 2)
  #ml1m("stc08", 60, 2)
  #ml1m("stc08", 70, 2)
  #ml1m("stc08", 80, 2)
  #ml1m("stc08", 90, 2)

  #ml1m("stc08", 50, 3)
  #ml1m("stc08", 60, 3)
  #ml1m("stc08", 70, 3)
  #ml1m("stc08", 80, 3)
  #ml1m("stc08", 90, 3)

  end = time.time()

  print()
  print("Time: " + format(end - start, '.5f') + " s")


if __name__ == "__main__":

  main()

