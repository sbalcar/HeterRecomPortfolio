#!/usr/bin/python3

import os
import random
import numpy as np

from input.batchesML1m.batchMLBanditTS import BatchMLBanditTS #class

from input.batchesML1m.batchMLContextDHondt import BatchMLContextDHondt #class

from input.batchesML1m.batchMLDHondtThompsonSampling import BatchMLDHondtThompsonSampling #class
from input.batchesML1m.batchMLDHondtThompsonSamplingINF import BatchMLDHondtThompsonSamplingINF #class

from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class
from input.batchesML1m.batchMLFuzzyDHondtINF import BatchMLFuzzyDHondtINF #class

from input.batchesML1m.batchMLFuzzyDHondtDirectOptimize import BatchMLFuzzyDHondtDirectOptimize #class
from input.batchesML1m.batchMLFuzzyDHondtDirectOptimizeINF import BatchMLFuzzyDHondtDirectOptimizeINF #class

from input.batchesML1m.batchMLWeightedAVG import BatchMLWeightedAVG #class

from input.batchesML1m.batchMLSingle import BatchMLSingle #class
from input.batchesML1m.batchMLSingleINF import BatchMLSingleINF #class

from input.batchesML1m.batchMLSingle2 import BatchMLSingle2 #class
from input.batchesML1m.batchMLSingleBPRMFHT import BatchMLSingleBPRMFHT #class
from input.batchesML1m.batchMLSingleW2VHT import BatchMLSingleW2VHT #class
from input.batchesML1m.batchMLSingleCosineCBHT import BatchMLSingleCosineCBHT #class

from input.batchesRetailrocket.batchRRSingle import BatchRRSingle #class
from input.batchesRetailrocket.batchRRSingleW2VHT import BatchRRSingleW2VHT #class

from input.batchesSlanTour.batchSTContextDHondt import BatchSTContextDHondt #class
from input.batchesSlanTour.batchSTContextDHondtINF import BatchSTContextDHondtINF #class
from input.batchesSlanTour.batchSTDHondtThompsonSampling import BatchSTDHondtThompsonSampling #class
from input.batchesSlanTour.batchSTDHondtThompsonSamplingINF import BatchSTDHondtThompsonSamplingINF #class
from input.batchesSlanTour.batchSTFuzzyDHondtDirectOptimize import BatchSTFuzzyDHondtDirectOptimize #class
from input.batchesSlanTour.batchSTFuzzyDHondtINF import BatchSTFuzzyDHondtINF #class
from input.batchesSlanTour.batchSTFuzzyDHondt import BatchSTFuzzyDHondt #class
from input.batchesSlanTour.batchSTWeightedAVG import BatchSTWeightedAVG #class
from input.batchesSlanTour.batchSTBanditTS import BatchSTBanditTS #class

from input.batchesSlanTour.batchSTSingleVMContextKNNHT import BatchSTVMContextKNNHT #class
from input.batchesSlanTour.batchSTSingleBPRMFHT import BatchSTSingleBPRMFHT #class
from input.batchesSlanTour.batchSTSingle import BatchSTSingle #class
from input.batchesSlanTour.batchSTSingleW2VHT import BatchSTSingleW2VHT #class
from input.batchesSlanTour.batchSTSingleCosineCBHT import BatchSTSingleCosineCBHT #class


def generateBatches():
    print("Generate Batches")

    # ML
#    BatchMLBanditTS.generateBatches()

#    BatchMLFuzzyDHondt.generateBatches()
#    BatchMLFuzzyDHondtINF.generateBatches()

#    BatchMLDHondtThompsonSampling.generateBatches()
#    BatchMLDHondtThompsonSamplingINF.generateBatches()

#    BatchMLFuzzyDHondtDirectOptimize.generateBatches()
#    BatchMLFuzzyDHondtDirectOptimizeINF.generateBatches()

#    BatchMLContextDHondt.generateBatches()

#    BatchMLWeightedAVG.generateBatches()

#    BatchMLSingle.generateBatches()
#    BatchMLSingleINF.generateBatches()

#    BatchMLSingle2.generateBatches()
#    BatchMLSingleBPRMFHT.generateBatches()
#    BatchMLSingleW2VHT.generateBatches()
#    BatchMLSingleCosineCBHT.generateBatches()

    # RR
#    BatchRRSingle.generateBatches()
#    BatchRRSingleW2VHT.generateBatches()

    #ST
    BatchSTContextDHondtINF.generateBatches()
#    BatchSTContextDHondt.generateBatches()
#    BatchSTDHondtThompsonSampling.generateBatches()
#    BatchSTFuzzyDHondtDirectOptimize.generateBatches()
#    BatchSTFuzzyDHondt.generateBatches()
#    BatchSTFuzzyDHondtINF.generateBatches()
#    BatchSTWeightedAVG.generateBatches()
#    BatchSTBanditTS.generateBatches()

#    BatchSTVMContextKNNHT.generateBatches()
#    BatchSTSingle.generateBatches()
#    BatchSTSingleBPRMFHT.generateBatches()
#    BatchSTSingleW2VHT.generateBatches()
#    BatchSTSingleCosineCBHT.generateBatches()


if __name__ == "__main__":

  np.random.seed(42)
  random.seed(42)

  os.chdir("..")
  print(os.getcwd())
  generateBatches()
