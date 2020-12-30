#!/usr/bin/python3

import os
import random
import numpy as np

from input.batchesML1m.batchMLBanditTS import BatchMLBanditTS #class

from input.batchesML1m.batchMLDHondtThompsonSampling import BatchMLDHondtThompsonSampling #class
from input.batchesML1m.batchMLDHondtThompsonSamplingINF import BatchMLDHondtThompsonSamplingINF #class

from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class
from input.batchesML1m.batchFuzzyDHondtINF import BatchFuzzyDHondtINF #class

from input.batchesML1m.batchMLFuzzyDHondtDirectOptimize import BatchMLFuzzyDHondtDirectOptimize #class
from input.batchesML1m.batchFuzzyDHondtDirectOptimizeINF import BatchFuzzyDHondtDirectOptimizeINF #class

from input.batchesML1m.batchMLSingle2 import BatchMLSingle2 #class
from input.batchesML1m.batchMLSingle import BatchMLSingle #class
from input.batchesML1m.batchMLSingleINF import BatchMLSingleINF #class

from input.batchesRetailrocket.batchRRSingle import BatchRRSingle #class

from input.batchSlanTour.batchSTSingle import BatchSTSingle #class


def generateBatches():
    print("Generate Batches")

    # ML
#    BatchMLBanditTS.generateBatches()

#    BatchMLFuzzyDHondt.generateBatches()
#    BatchFuzzyDHondtINF.generateBatches()

#    BatchMLDHondtThompsonSampling.generateBatches()
#    BatchMLDHondtThompsonSamplingINF.generateBatches()

#    BatchMLFuzzyDHondtDirectOptimize.generateBatches()
#    BatchFuzzyDHondtDirectOptimizeINF.generateBatches()

#    BatchMLSingle.generateBatches()
#    BatchMLSingleINF.generateBatches()

    BatchMLSingle2.generateBatches()

    # RR
#    BatchRRSingle.generateBatches()

    #ST
#    BatchSTSingle.generateBatches()


if __name__ == "__main__":

  np.random.seed(42)
  random.seed(42)

  os.chdir("..")
  print(os.getcwd())
  generateBatches()
