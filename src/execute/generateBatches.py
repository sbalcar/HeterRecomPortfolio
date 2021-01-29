#!/usr/bin/python3

import os
import random
import numpy as np

from typing import List #class

from input.batchesML1m.batchMLBanditTS import BatchMLBanditTS #class

from input.batchesML1m.batchMLContextDHondt import BatchMLContextDHondt #class

from input.batchesML1m.batchMLFuzzyDHondtThompsonSampling import BatchMLFuzzyDHondtThompsonSampling #class
from input.batchesML1m.batchMLFuzzyDHondtThompsonSamplingINF import BatchMLFuzzyDHondtThompsonSamplingINF #class

from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class
from input.batchesML1m.batchMLFuzzyDHondtINF import BatchMLFuzzyDHondtINF #class

from input.batchesML1m.batchMLFuzzyDHondtDirectOptimize import BatchMLFuzzyDHondtDirectOptimize #class
from input.batchesML1m.batchMLFuzzyDHondtDirectOptimizeINF import BatchMLFuzzyDHondtDirectOptimizeINF #class

from input.batchesML1m.batchMLWeightedAVG import BatchMLWeightedAVG #class
from input.batchesML1m.batchMLRandomRecsSwitching import BatchMLRandomRecsSwitching #class
from input.batchesML1m.batchMLRandomKfromN import BatchMLRandomKfromN #class

from input.batchesML1m.batchMLFuzzyDHondtDirectOptimizeThompsonSampling import BatchMLFuzzyDHondtDirectOptimizeThompsonSampling #class

from input.batchesML1m.batchMLSingle import BatchMLSingle #class
from input.batchesML1m.batchMLSingleINF import BatchMLSingleINF #class

from input.batchesML1m.batchMLSingleBPRMFHT import BatchMLSingleBPRMFHT #class
from input.batchesML1m.batchMLSingleW2VHT import BatchMLSingleW2VHT #class
from input.batchesML1m.batchMLSingleCosineCBHT import BatchMLSingleCosineCBHT #class
from input.batchesML1m.batchMLSingleBPRMFHT import BatchMLSingleBPRMFHT  #class
from input.batchesML1m.batchMLSingleVMContextKNNHT import BatchMLVMContextKNNHT #class

from input.batchesRetailrocket.batchRRSingle import BatchRRSingle #class
from input.batchesRetailrocket.batchRRSingleW2VHT import BatchRRSingleW2VHT #class

from input.batchesSlanTour.batchSTFuzzyDHondtDirectOptimizeThompsonSampling import BatchSTFuzzyDHondtDirectOptimizeThompsonSampling #class
from input.batchesSlanTour.batchSTContextDHondt import BatchSTContextDHondt #class
from input.batchesSlanTour.batchSTContextDHondtINF import BatchSTContextDHondtINF #class
from input.batchesSlanTour.batchSTDHondtThompsonSampling import BatchSTDHondtThompsonSampling #class
from input.batchesSlanTour.batchSTDHondtThompsonSamplingINF import BatchSTDHondtThompsonSamplingINF #class
from input.batchesSlanTour.batchSTFuzzyDHondtDirectOptimize import BatchSTFuzzyDHondtDirectOptimize #class
from input.batchesSlanTour.batchSTFuzzyDHondtINF import BatchSTFuzzyDHondtINF #class
from input.batchesSlanTour.batchSTFuzzyDHondt import BatchSTFuzzyDHondt #class
from input.batchesSlanTour.batchSTWeightedAVG import BatchSTWeightedAVG #class
from input.batchesSlanTour.batchSTBanditTS import BatchSTBanditTS #class
from input.batchesSlanTour.batchSTRandomRecsSwitching import BatchSTRandomRecsSwitching #class
from input.batchesSlanTour.batchSTRandomKfromN import BatchSTRandomKfromN #class

from input.batchesSlanTour.batchSTSingleVMContextKNNHT import BatchSTVMContextKNNHT #class
from input.batchesSlanTour.batchSTSingleBPRMFHT import BatchSTSingleBPRMFHT #class
from input.batchesSlanTour.batchSTSingle import BatchSTSingle #class
from input.batchesSlanTour.batchSTSingleW2VHT import BatchSTSingleW2VHT #class
from input.batchesSlanTour.batchSTSingleCosineCBHT import BatchSTSingleCosineCBHT #class


def generateAllBatches():
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

    BatchMLRandomRecsSwitching.generateBatches()
    BatchMLRandomKfromN.generateBatches()

#    BatchMLSingle.generateBatches()
#    BatchMLSingleINF.generateBatches()

#    BatchMLSingleBPRMFHT.generateBatches()
#    BatchMLSingleW2VHT.generateBatches()
#    BatchMLSingleCosineCBHT.generateBatches()
#    BatchMLVMContextKNNHT.generateBatches()

    # RR
#    BatchRRSingle.generateBatches()
#    BatchRRSingleW2VHT.generateBatches()

    #ST
#    BatchSTContextDHondtINF.generateBatches()
#    BatchSTContextDHondt.generateBatches()
#    BatchSTDHondtThompsonSampling.generateBatches()
#    BatchSTDHondtThompsonSamplingINF.generateBatches()
#    BatchSTFuzzyDHondtDirectOptimize.generateBatches()
#    BatchSTFuzzyDHondt.generateBatches()
#    BatchSTFuzzyDHondtINF.generateBatches()
#    BatchSTWeightedAVG.generateBatches()
#    BatchSTBanditTS.generateBatches()
    BatchSTRandomRecsSwitching.generateBatches()
    BatchSTRandomKfromN.generateBatches()

#    BatchSTSingle.generateBatches()
#    BatchSTSingleBPRMFHT.generateBatches()
#    BatchSTSingleW2VHT.generateBatches()
#    BatchSTSingleCosineCBHT.generateBatches()
#    BatchSTVMContextKNNHT.generateBatches()



def generateBatchesJournal():
    print("Generate Batches")

    # ML
    BatchMLSingle.generateBatches()

    BatchMLFuzzyDHondt.lrClicks:List[float] = [0.03]
    BatchMLFuzzyDHondt.lrViewDivisors:List[float] = [250]
    BatchMLFuzzyDHondt.selectorIds = [BatchMLFuzzyDHondt.SLCTR_FIXED]
    BatchMLBanditTS.generateBatches()

    BatchMLWeightedAVG.lrClicks:List[float] = [0.03]
    BatchMLWeightedAVG.lrViewDivisors:List[float] = [250]
    BatchMLWeightedAVG.generateBatches()

    BatchMLRandomRecsSwitching.generateBatches()
    BatchMLRandomKfromN.generateBatches()

    BatchMLFuzzyDHondt.generateBatches()

    BatchMLFuzzyDHondtThompsonSampling.generateBatches()
    BatchMLContextDHondt.generateBatches()

    BatchMLFuzzyDHondtDirectOptimize.lrClicks:List[float] = [0.03]
    BatchMLFuzzyDHondtDirectOptimize.lrViewDivisors:List[float] = [250]
    BatchMLFuzzyDHondtDirectOptimize.selectorIds = [BatchMLFuzzyDHondtDirectOptimize.SLCTR_FIXED]
    BatchMLFuzzyDHondtDirectOptimize.generateBatches()

    BatchMLFuzzyDHondtDirectOptimizeThompsonSampling.lrClicks:List[float] = [0.03]
    BatchMLFuzzyDHondtDirectOptimizeThompsonSampling.lrViewDivisors:List[float] = [250]
    BatchMLFuzzyDHondtDirectOptimizeThompsonSampling.selectorIds = [BatchMLFuzzyDHondtDirectOptimizeThompsonSampling.SLCTR_FIXED]
    BatchMLFuzzyDHondtDirectOptimizeThompsonSampling.generateBatches()

    # RR
    BatchRRSingle.generateBatches()
    #BatchRRSingleW2VHT.generateBatches()


    #ST
    BatchSTSingle.generateBatches()

    BatchSTBanditTS.generateBatches()  # only Fixed selector
    BatchSTWeightedAVG.generateBatches()

    BatchSTRandomRecsSwitching.generateBatches()
    BatchSTRandomKfromN.generateBatches()

    BatchSTFuzzyDHondt.generateBatches()

    BatchSTDHondtThompsonSampling.generateBatches()
    BatchSTContextDHondt.generateBatches()

    BatchSTFuzzyDHondtDirectOptimize.generateBatches()

    BatchSTFuzzyDHondtDirectOptimizeThompsonSampling.generateBatches()



def generateBatches():

    np.random.seed(42)
    random.seed(42)

    #generateAllBatches()
    generateBatchesJournal()




if __name__ == "__main__":

  os.chdir("..")
  print(os.getcwd())

  #generateBatches()
  generateBatchesJournal()
