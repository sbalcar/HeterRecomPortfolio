#!/usr/bin/python3

import time
import sys
import os

import random
import traceback
import numpy as np

from typing import List

from input.batchesML1m.batchMLBanditTS import BatchMLBanditTS #class
from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class
from input.batchesML1m.batchMLDHondtThompsonSampling import BatchMLDHondtThompsonSampling #class
from input.batchesML1m.batchMLFuzzyDHondtINF import BatchFuzzyDHondtINF #class
from input.batchesML1m.batchMLDHondtThompsonSamplingINF import BatchMLDHondtThompsonSamplingINF #class
from input.batchesML1m.batchMLFuzzyDHondtDirectOptimize import BatchMLFuzzyDHondtDirectOptimize #class
from input.batchesML1m.batchMLFuzzyDHondtDirectOptimizeINF import BatchFuzzyDHondtDirectOptimizeINF #class
from input.batchesML1m.batchMLSingle import BatchMLSingle #class
from input.batchesML1m.batchMLSingleINF import BatchMLSingleINF #class
from input.batchesRetailrocket.batchRRSingle import BatchRRSingle #class

from input.batchesML1m.batchMLSingle2 import BatchMLSingle2 #class
from input.batchesML1m.batchMLSingleBPRMFHT import BatchMLSingleBPRMFHT #class
from input.batchesML1m.batchMLSingleW2VHT import BatchMLSingleW2VHT #class
from input.batchesML1m.batchMLSingleCosineCBHT import BatchMLSingleCosineCBHT #class

from input.batchesRetailrocket.batchRRSingle import BatchRRSingle #class
from input.batchesRetailrocket.batchRRSingleW2VHT import BatchRRSingleW2VHT #class

from input.batchSlanTour.batchSTFuzzyDHondt import BatchSTFuzzyDHondt #class

from input.batchSlanTour.batchSTSingle import BatchSTSingle #class
from input.batchSlanTour.batchSTSingleBPRMFHT import BatchSTSingleBPRMFHT #class
from input.batchSlanTour.batchSTSingleW2VHT import BatchSTSingleW2VHT #class
from input.batchSlanTour.batchSTSingleCosineCBHT import BatchSTSingleCosineCBHT #class


def sequentialEvaluation():

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
          except Exception:
              traceback.print_exc()
              print("Skiped job: " + command)