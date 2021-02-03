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
from input.batchesML1m.batchMLFuzzyDHondtDirectOptimizeThompsonSamplingINF import BatchMLFuzzyDHondtDirectOptimizeThompsonSamplingINF #class

from input.batchesML1m.batchMLSingle import BatchMLSingle #class
from input.batchesML1m.batchMLSingleINF import BatchMLSingleINF #class

from input.batchesML1m.batchMLSingleBPRMFHT import BatchMLSingleBPRMFHT #class
from input.batchesML1m.batchMLSingleW2VHT import BatchMLSingleW2VHT #class
from input.batchesML1m.batchMLSingleCosineCBHT import BatchMLSingleCosineCBHT #class
from input.batchesML1m.batchMLSingleBPRMFHT import BatchMLSingleBPRMFHT  #class
from input.batchesML1m.batchMLSingleVMContextKNNHT import BatchMLVMContextKNNHT #class

from input.batchesRetailrocket.batchRRSingle import BatchRRSingle #class
from input.batchesRetailrocket.batchRRSingleW2VHT import BatchRRSingleW2VHT #class
from input.batchesRetailrocket.batchRRSingleVMContextKNNHT import BatchRRVMContextKNNHT #class

from input.batchesSlanTour.batchSTFuzzyDHondtDirectOptimizeThompsonSampling import BatchSTFuzzyDHondtDirectOptimizeThompsonSampling #class
from input.batchesSlanTour.batchSTFuzzyDHondtDirectOptimizeThompsonSamplingINF import BatchSTFuzzyDHondtDirectOptimizeThompsonSamplingINF #class

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
    BatchMLBanditTS.generateAllBatches()

    BatchMLFuzzyDHondt.generateAllBatches()
    BatchMLFuzzyDHondtINF.generateAllBatches()

    BatchMLFuzzyDHondtThompsonSampling.generateAllBatches()
    BatchMLFuzzyDHondtThompsonSamplingINF.generateAllBatches()

    BatchMLFuzzyDHondtDirectOptimize.generateAllBatches()
    BatchMLFuzzyDHondtDirectOptimizeINF.generateAllBatches()

    BatchMLContextDHondt.generateAllBatches()

    BatchMLWeightedAVG.generateAllBatches()

    BatchMLRandomRecsSwitching.generateAllBatches()
    BatchMLRandomKfromN.generateAllBatches()

    BatchMLSingle.generateAllBatches()
    BatchMLSingleINF.generateAllBatches()

    BatchMLSingleBPRMFHT.generateAllBatches()
    BatchMLSingleW2VHT.generateAllBatches()
    BatchMLSingleCosineCBHT.generateAllBatches()
    BatchMLVMContextKNNHT.generateAllBatches()

    # RR
    BatchRRSingle.generateAllBatches()
    BatchRRSingleW2VHT.generateAllBatches()

    #ST
    BatchSTContextDHondtINF.generateAllBatches()
    BatchSTContextDHondt.generateAllBatches()
    BatchSTDHondtThompsonSampling.generateAllBatches()
    BatchSTDHondtThompsonSamplingINF.generateAllBatches()
    BatchSTFuzzyDHondtDirectOptimize.generateAllBatches()
    BatchSTFuzzyDHondt.generateAllBatches()
    BatchSTFuzzyDHondtINF.generateAllBatches()
    BatchSTWeightedAVG.generateAllBatches()
    BatchSTBanditTS.generateAllBatches()
    BatchSTRandomRecsSwitching.generateAllBatches()
    BatchSTRandomKfromN.generateAllBatches()

    BatchSTSingle.generateAllBatches()
    BatchSTSingleBPRMFHT.generateAllBatches()
    BatchSTSingleW2VHT.generateAllBatches()
    BatchSTSingleCosineCBHT.generateAllBatches()
    BatchSTVMContextKNNHT.generateAllBatches()



def generateBatchesJournal():
    print("Generate Batches")

    jobIdML01:str = "FixedReduceOLin07500HLin07500"
    jobIdML02:str = "FixedReduceOLin10075HLin0500"
    jobIdML03:str = "FixedReduceOLin1000HLin1005"
    jobIdML04:str = "FixedReduceOLin1005HLin1005"

    jobIdML05:str = "FixedReduceProbOLin07500HLin07500"
    jobIdML06:str = "FixedReduceProbOLin075025HLin05025"
    jobIdML07:str = "FixedReduceProbOLin1000HLin1005"
    jobIdML08:str = "FixedReduceProbOLin1005HLin1005"


    jobIdST01:str = "FixedReduceOLin07500HLin05025"
    jobIdST02:str = "FixedReduceOLin0500HLin1005"
    jobIdST03:str = "FixedReduceOLin1000HLin1005"
    jobIdST04:str = "FixedReduceOLin1005HLin1005"

    jobIdST05:str = "FixedReduceProbOLin07500HLin07500"
    jobIdST06:str = "FixedReduceProbOLin07500HLin075025"
    jobIdST07:str = "FixedReduceProbOLin1000HLin1005"
    jobIdST08:str = "FixedReduceProbOLin1005HLin1005"


#    BatchMLFuzzyDHondtINF.lrClicks:List[float] = [0.03]
#    BatchMLFuzzyDHondtINF.lrViewDivisors:List[float] = [250]
#    BatchMLFuzzyDHondt.selectorIds = [BatchMLFuzzyDHondt.SLCTR_FIXED]
#    BatchSTFuzzyDHondtINF.generateSelectedBatches([jobIdST01.replace("Reduce", "Clk003ViewDivisor250Reduce"),
#                                                   jobIdST02.replace("Reduce", "Clk003ViewDivisor250Reduce"),
#                                                   jobIdST03.replace("Reduce", "Clk003ViewDivisor250Reduce"),
#                                                   jobIdST04.replace("Reduce", "Clk003ViewDivisor250Reduce"),
#                                                   jobIdST05.replace("Reduce", "Clk003ViewDivisor250Reduce"),
#                                                   jobIdST06.replace("Reduce", "Clk003ViewDivisor250Reduce"),
#                                                   jobIdST07.replace("Reduce", "Clk003ViewDivisor250Reduce"),
#                                                   jobIdST08.replace("Reduce", "Clk003ViewDivisor250Reduce")])


    # ML #############################################################################
#    BatchMLSingle.generateAllBatches()

#    BatchMLFuzzyDHondt.lrClicks:List[float] = [0.03]
#    BatchMLFuzzyDHondt.lrViewDivisors:List[float] = [250]
#    BatchMLFuzzyDHondt.selectorIds = [BatchMLFuzzyDHondt.SLCTR_FIXED]
#    BatchMLBanditTS.generateAllBatches()

#    BatchMLWeightedAVG.lrClicks:List[float] = [0.03]
#    BatchMLWeightedAVG.lrViewDivisors:List[float] = [250]
#    BatchMLWeightedAVG.generateAllBatches()

#    BatchMLRandomRecsSwitching.generateAllBatches()
#    BatchMLRandomKfromN.generateAllBatches()

#    BatchMLFuzzyDHondt.generateAllBatches()

#    BatchMLFuzzyDHondtThompsonSampling.generateAllBatches()
#    BatchMLContextDHondt.generateAllBatches()

#    BatchMLFuzzyDHondtDirectOptimize.lrClicks:List[float] = [0.03]
#    BatchMLFuzzyDHondtDirectOptimize.lrViewDivisors:List[float] = [250]
#    BatchMLFuzzyDHondtDirectOptimize.selectorIds = [BatchMLFuzzyDHondtDirectOptimize.SLCTR_FIXED]
#    BatchMLFuzzyDHondtDirectOptimize.generateAllBatches()

#    BatchMLFuzzyDHondtDirectOptimizeThompsonSampling.generateAllBatches()

    # INF
    BatchMLFuzzyDHondtDirectOptimizeThompsonSamplingINF.generateSelectedBatches([
        jobIdML01, jobIdML02, jobIdML03, jobIdML04, jobIdML05, jobIdML06, jobIdML07, jobIdML08])


    # RR #############################################################################
#    BatchRRSingle.generateAllBatches()
#    BatchRRSingleW2VHT.generateAllBatches()
#    BatchRRVMContextKNNHT.generateAllBatches()


    # ST #############################################################################
#    BatchSTSingle.generateAllBatches()

#    BatchSTBanditTS.generateAllBatches()  # only Fixed selector
#    BatchSTWeightedAVG.generateAllBatches()

#    BatchSTRandomRecsSwitching.generateAllBatches()
#    BatchSTRandomKfromN.generateAllBatches()

#    BatchSTFuzzyDHondt.generateAllBatches()

#    BatchSTDHondtThompsonSampling.generateAllBatches()
#    BatchSTContextDHondt.generateAllBatches()

#    BatchSTFuzzyDHondtDirectOptimize.generateAllBatches()

#    BatchSTFuzzyDHondtDirectOptimizeThompsonSampling.generateAllBatches()

    BatchSTFuzzyDHondtDirectOptimizeThompsonSamplingINF.generateSelectedBatches([
        jobIdST01, jobIdST02, jobIdST03, jobIdST04, jobIdST05, jobIdST06, jobIdST07, jobIdST08])



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
