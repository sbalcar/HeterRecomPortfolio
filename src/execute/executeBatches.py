#!/usr/bin/python3

import time
import sys
import os

import random
import numpy as np

from typing import List

from input.batchesML1m.batchBanditTS import BatchBanditTS #class
from input.batchesML1m.batchDHondt import BatchDHondt #class
from input.batchesML1m.batchDHondtThompsonSampling import BatchDHondtThompsonSampling #class
from input.batchesML1m.batchNegDHondt import BatchNegDHondt #class
from input.batchesML1m.batchNegDHondtThompsonSampling import BatchNegDHondtThompsonSampling #class
from input.batchesML1m.batchSingle import BatchSingle #class


def executeBatches():

  np.random.seed(42)
  random.seed(42)

  print("ExecuteBatch")

  batchesDir:str = ".." + os.sep + "inputs"

  batches:List[str] = [batchesDir + os.sep + f.name for f in os.scandir(batchesDir) if f.is_dir() if f.name != "__pycache__"]
  for batchIdI in batches:
      print(batchIdI)

      jobs:List[str] = [batchIdI + os.sep + f.name for f in os.scandir(batchesDir + os.sep + batchIdI)
                        if f.is_file() and not f.name.startswith(".")]
      for jobI in jobs:
          print(jobI)

          try:
              file = open(jobI, "r")
              command: str = file.read()

              print("Removing job: " + command)
              os.remove(jobI)
              print("Removed job: " + command)

              print("Executing job: " + command)
              exec(command)
              print("Finishing job: " + command)
              return
          except:
              print("Skiped job: " + command)