#!/usr/bin/python3

import os
import random
import numpy as np

from input.batchesML1m.batchBanditTS import BatchBanditTS #class

from input.batchesML1m.batchDHondtThompsonSampling import BatchDHondtThompsonSampling #class
from input.batchesML1m.batchDHondtThompsonSamplingINF import BatchDHondtThompsonSamplingINF #class

from input.batchesML1m.batchFuzzyDHondt import BatchFuzzyDHondt #class
from input.batchesML1m.batchFuzzyDHondtINF import BatchFuzzyDHondtINF #class

from input.batchesML1m.batchFuzzyDHondtDirectOptimize import BatchFuzzyDHondtDirectOptimize #class
from input.batchesML1m.batchFuzzyDHondtDirectOptimizeINF import BatchFuzzyDHondtDirectOptimizeINF #class

from input.batchesML1m.batchSingle import BatchSingle #class
from input.batchesML1m.batchSingleINF import BatchSingleINF #class

from input.batchesRetailrocket.batchRRSingle import BatchRRSingle #class

from input.batchSlanTour.batchSTSingle import BatchSTSingle #class


def generateBatches():
    print("Generate Batches")

    # ML
#    BatchBanditTS.generateBatches()

    BatchFuzzyDHondt.generateBatches()
#    BatchFuzzyDHondtINF.generateBatches()

#    BatchDHondtThompsonSampling.generateBatches()
#    BatchDHondtThompsonSamplingINF.generateBatches()

#    BatchFuzzyDHondtDirectOptimize.generateBatches()
#    BatchFuzzyDHondtDirectOptimizeINF.generateBatches()

    BatchSingle.generateBatches()
#    BatchSingleINF.generateBatches()

    # RR
    BatchRRSingle.generateBatches()

    #ST
    BatchSTSingle.generateBatches()


if __name__ == "__main__":

  np.random.seed(42)
  random.seed(42)

  os.chdir("..")
  print(os.getcwd())
  generateBatches()
