#!/usr/bin/python3

import time
import sys
import os

import random
import traceback
import numpy as np

from typing import List

from input.batchesML1m.batchMLFuzzyDHondtDirectOptimizeThompsonSamplingINF import BatchMLFuzzyDHondtDirectOptimizeThompsonSamplingINF #class
from input.batchesML1m.batchMLFuzzyDHondtDirectOptimizeThompsonSampling import BatchMLFuzzyDHondtDirectOptimizeThompsonSampling #class
from input.batchesML1m.batchMLContextDHondtINF import BatchMLContextDHondtINF #class
from input.batchesML1m.batchMLContextDHondt import BatchMLContextDHondt #class
from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class
from input.batchesML1m.batchMLFuzzyDHondtThompsonSampling import BatchMLFuzzyDHondtThompsonSampling #class
from input.batchesML1m.batchMLFuzzyDHondtINF import BatchMLFuzzyDHondtINF #class
from input.batchesML1m.batchMLFuzzyDHondtThompsonSamplingINF import BatchMLFuzzyDHondtThompsonSamplingINF #class
from input.batchesML1m.batchMLFuzzyDHondtDirectOptimize import BatchMLFuzzyDHondtDirectOptimize #class
from input.batchesML1m.batchMLFuzzyDHondtDirectOptimizeINF import BatchMLFuzzyDHondtDirectOptimizeINF #class
from input.batchesML1m.batchMLSingle import BatchMLSingle #class
from input.batchesML1m.batchMLSingleINF import BatchMLSingleINF #class
from input.batchesML1m.batchMLBanditTS import BatchMLBanditTS #class
from input.batchesML1m.batchMLWeightedAVG import BatchMLWeightedAVG #class
from input.batchesML1m.batchMLRandomRecsSwitching import BatchMLRandomRecsSwitching #class
from input.batchesML1m.batchMLRandomKfromN import BatchMLRandomKfromN #class

from input.batchesML1m.batchMLSingle import BatchMLSingle #class
from input.batchesML1m.batchMLSingleW2VHT import BatchMLSingleW2VHT #class
from input.batchesML1m.batchMLSingleCosineCBHT import BatchMLSingleCosineCBHT #class
from input.batchesML1m.batchMLSingleBPRMFHT import BatchMLSingleBPRMFHT  #class
from input.batchesML1m.batchMLSingleVMContextKNNHT import BatchMLVMContextKNNHT #class

from input.batchesRetailrocket.batchRRSingle import BatchRRSingle #class
from input.batchesRetailrocket.batchRRSingleW2VHT import BatchRRSingleW2VHT #class
from input.batchesRetailrocket.batchRRSingleVMContextKNNHT import BatchRRVMContextKNNHT #class

from input.batchesSlanTour.batchSTFuzzyDHondtDirectOptimizeThompsonSamplingINF import BatchSTFuzzyDHondtDirectOptimizeThompsonSamplingINF #class
from input.batchesSlanTour.batchSTFuzzyDHondtDirectOptimizeThompsonSampling import BatchSTFuzzyDHondtDirectOptimizeThompsonSampling #class
from input.batchesSlanTour.batchSTContextDHondtINF import BatchSTContextDHondtINF #class
from input.batchesSlanTour.batchSTContextDHondt import BatchSTContextDHondt #class
from input.batchesSlanTour.batchSTDHondtThompsonSamplingINF import BatchSTDHondtThompsonSamplingINF #class
from input.batchesSlanTour.batchSTDHondtThompsonSampling import BatchSTDHondtThompsonSampling #class
from input.batchesSlanTour.batchSTDHondtThompsonSamplingINF import BatchSTDHondtThompsonSamplingINF #class
from input.batchesSlanTour.batchSTFuzzyDHondtDirectOptimize import BatchSTFuzzyDHondtDirectOptimize #class
from input.batchesSlanTour.batchSTFuzzyDHondt import BatchSTFuzzyDHondt #class
from input.batchesSlanTour.batchSTFuzzyDHondtINF import BatchSTFuzzyDHondtINF #class
from input.batchesSlanTour.batchSTWeightedAVG import BatchSTWeightedAVG #class
from input.batchesSlanTour.batchSTBanditTS import BatchSTBanditTS #class
from input.batchesSlanTour.batchSTRandomRecsSwitching import BatchSTRandomRecsSwitching #class
from input.batchesSlanTour.batchSTRandomKfromN import BatchSTRandomKfromN #class


from input.batchesSlanTour.batchSTSingleVMContextKNNHT import BatchSTVMContextKNNHT #class
from input.batchesSlanTour.batchSTSingle import BatchSTSingle #class
from input.batchesSlanTour.batchSTSingleBPRMFHT import BatchSTSingleBPRMFHT #class
from input.batchesSlanTour.batchSTSingleW2VHT import BatchSTSingleW2VHT #class
from input.batchesSlanTour.batchSTSingleCosineCBHT import BatchSTSingleCosineCBHT #class


def sequentialEvaluation():

  timeDelays:float = random.uniform(0.0, 5.0)
  print("TimeDelays: " + str(timeDelays))
  time.sleep(timeDelays)

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
              return