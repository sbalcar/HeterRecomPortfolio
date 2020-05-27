#!/usr/bin/python3

import time
import sys

import random
from numpy import np

from jobs.ml1m import ml1mDiv50 #function
from jobs.ml1m import ml1mDiv60 #function
from jobs.ml1m import ml1mDiv70 #function
from jobs.ml1m import ml1mDiv80 #function
from jobs.ml1m import ml1mDiv90 #function

def main():

  np.random.seed(42)
  random.seed(42)

  start = time.time()

  ml1mDiv50()
  #ml1mDiv60()
  #ml1mDiv70()
  #ml1mDiv80()
  #ml1mDiv90()

  end = time.time()

  print()
  print("Time: " + format(end - start, '.5f') + " s")


if __name__ == "__main__":

  main()

