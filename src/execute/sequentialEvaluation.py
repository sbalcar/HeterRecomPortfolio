#!/usr/bin/python3

import time
import sys
import os

import random
import traceback
import numpy as np

from typing import List

from batchDefinition.ml1m.batchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR import BatchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingINF import BatchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingINF #class
from batchDefinition.ml1m.batchDefMLFAI import BatchDefMLFAI #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtDirectOptimizeThompsonSampling import BatchDefMLFuzzyDHondtDirectOptimizeThompsonSampling #class
from batchDefinition.ml1m.batchDefMLContextDHondtINF import BatchDefMLContextDHondtINF #class
from batchDefinition.ml1m.batchDefMLContextFuzzyDHondtDirectOptimize import BatchDefMLContextFuzzyDHondtDirectOptimize #class
from batchDefinition.ml1m.batchDefMLContextFuzzyDHondtDirectOptimizeINF import BatchDefMLContextFuzzyDHondtDirectOptimizeINF #class
from batchDefinition.ml1m.batchDefMLContextDHondt import BatchDefMLContextDHondt #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtThompsonSampling import BatchDefMLFuzzyDHondtThompsonSampling #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtINF import BatchMLFuzzyDHondtINF #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtThompsonSamplingINF import BatchDefMLFuzzyDHondtThompsonSamplingINF #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtDirectOptimize import BatchDefMLFuzzyDHondtDirectOptimize #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtDirectOptimizeINF import BatchDefMLFuzzyDHondtDirectOptimizeINF #class
from batchDefinition.ml1m.batchDefMLSingle import BatchDefMLSingle #class
from batchDefinition.ml1m.batchDefMLSingleINF import BatchDefMLSingleINF #class
from batchDefinition.ml1m.batchDefMLBanditTS import BatchDefMLBanditTS #class
from batchDefinition.ml1m.batchDefMLWeightedAVG import BatchDefMLWeightedAVG #class
from batchDefinition.ml1m.batchDefMLWeightedAVGMMR import BatchDefMLWeightedAVGMMR #class
from batchDefinition.ml1m.batchDefMLRandomRecsSwitching import BatchDefMLRandomRecsSwitching #class
from batchDefinition.ml1m.batchDefMLRandomKfromN import BatchDefMLRandomKfromN #class

from batchDefinition.ml1m.batchDefMLSingle import BatchDefMLSingle #class
from batchDefinition.ml1m.batchDefMLSingleW2VHT import BatchDefMLSingleW2VHT #class
from batchDefinition.ml1m.batchDefMLSingleCosineCBHT import BatchDefMLSingleCosineCBHT #class
from batchDefinition.ml1m.batchDefMLSingleBPRMFHT import BatchDefMLSingleBPRMFHT  #class
from batchDefinition.ml1m.batchDefMLSingleVMContextKNNHT import BatchDefMLSingleVMContextKNNHT #class

from batchDefinition.retailrocket.batchDefRRSingle import BatchDefRRSingle #class
from batchDefinition.retailrocket.batchDefRRSingleW2VHT import BatchDefRRSingleW2VHT #class
from batchDefinition.retailrocket.batchDefRRSingleBPRMFHT import BatchDefRRSingleBPRMFHT #class
from batchDefinition.retailrocket.batchDefRRSingleVMContextKNNHT import BatchDefRRSingleVMContextKNNHT #class
from batchDefinition.retailrocket.batchDefRRSingleCosineCBHT import BatchDefRRSingleCosineCBHT #class
from batchDefinition.retailrocket.batchDefRRBanditTS import BatchDefRRBanditTS #class
from batchDefinition.retailrocket.batchDefRRFuzzyDHondt import BatchDefRRFuzzyDHondt #class
from batchDefinition.retailrocket.batchDefRRFAI import BatchDefRRFAI #class
from batchDefinition.retailrocket.batchDefRRFuzzyDHondtDirectOptimizeThompsonSampling import BatchDefRRFuzzyDHondtDirectOptimizeThompsonSampling #class

from batchDefinition.slanTour.batchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingINF import BatchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingINF #class
from batchDefinition.slanTour.batchDefSTFuzzyDHondtDirectOptimizeThompsonSampling import BatchDefSTFuzzyDHondtDirectOptimizeThompsonSampling #class
from batchDefinition.slanTour.batchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingMMR import BatchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingMMR #class
from batchDefinition.slanTour.batchDefSTContextDHondtINF import BatchDefSTContextDHondtINF #class
from batchDefinition.slanTour.batchDefSTContextFuzzyDHondtDirectOptimize import BatchDefSTContextFuzzyDHondtDirectOptimize #class
from batchDefinition.slanTour.batchDefSTContextFuzzyDHondtDirectOptimizeINF import BatchDefSTContextFuzzyDHondtDirectOptimizeINF #class

from batchDefinition.slanTour.batchDefSTContextDHondt import BatchDefSTContextDHondt #class
from batchDefinition.slanTour.batchDefSTFuzzyDHondtThompsonSamplingINF import BatchDefSTDFuzzyHondtThompsonSamplingINF #class
from batchDefinition.slanTour.batchDefSTFuzzyDHondtThompsonSampling import BatchDefSTFuzzyDHondtThompsonSampling #class
from batchDefinition.slanTour.batchDefSTFuzzyDHondtThompsonSamplingINF import BatchDefSTDFuzzyHondtThompsonSamplingINF #class
from batchDefinition.slanTour.batchDefSTFuzzyDHondtDirectOptimize import BatchDefSTFuzzyDHondtDirectOptimize #class
from batchDefinition.slanTour.batchDefSTFuzzyDHondt import BatchDefSTFuzzyDHondt #class
from batchDefinition.slanTour.batchDefSTFuzzyDHondtINF import BatchDefSTFuzzyDHondtINF #class
from batchDefinition.slanTour.batchDefSTWeightedAVG import BatchDefSTWeightedAVG #class
from batchDefinition.slanTour.batchDefSTWeightedAVGMMR import BatchDefSTWeightedAVGMMR #class
from batchDefinition.slanTour.batchDefSTFAI import BatchDefSTFAI #cass
from batchDefinition.slanTour.batchDefSTBanditTS import BatchDefSTBanditTS #class
from batchDefinition.slanTour.batchDefSTRandomRecsSwitching import BatchDefSTRandomRecsSwitching #class
from batchDefinition.slanTour.batchDefSTRandomKfromN import BatchDefSTRandomKfromN #class


from batchDefinition.slanTour.batchDefSTSingleVMContextKNNHT import BatchDefSTSingleVMContextKNNHT #class
from batchDefinition.slanTour.batchDefSTSingle import BatchDefSTSingle #class
from batchDefinition.slanTour.batchDefSTSingleBPRMFHT import BatchDefSTSingleBPRMFHT #class
from batchDefinition.slanTour.batchDefSTSingleW2VHT import BatchDefSTSingleW2VHT #class
from batchDefinition.slanTour.batchDefSTSingleCosineCBHT import BatchDefSTSingleCosineCBHT #class


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