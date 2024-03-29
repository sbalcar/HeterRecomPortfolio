#!/usr/bin/python3

import time
import sys
import os

import random
import numpy as np

from execute.sequentialEvaluation import sequentialEvaluation #function
from execute.generateBatches import generateBatches #function
from execute.generateBehaviour import generateBehaviour #function
from execute.startHttpServer import startHttpServer #function
from execute.resultsVerification import resultsVerification #function

from batch.batch import Batch #class
from batchDefinition.aBatchDefinition import ABatchDefinition #class


def main2():

  start = time.time()

  sequentialEvaluation()

  end = time.time()

  print()
  print("Time: " + format(end - start, '.5f') + " s")


if __name__ == "__main__":
  #generateBatches()

  if len(sys.argv) == 2 and sys.argv[1] == "-generateBatches":
      generateBatches()

  if len(sys.argv) == 2 and sys.argv[1] == "-generateBehaviours":
      generateBehaviour()

  if len(sys.argv) == 2 and sys.argv[1] == "-startHttpServer":
      startHttpServer()

  if len(sys.argv) == 2 and sys.argv[1] == "-resultVerification":
      resultsVerification()

  if len(sys.argv) == 1:
      main2()
