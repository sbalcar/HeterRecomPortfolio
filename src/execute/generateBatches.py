#!/usr/bin/python3

import os
import random
import numpy as np

from typing import List #class

from batchDefinition.aBatchDefinition import ABatchDefinition #class
from batchDefinition.ml1m.batchDefMLBanditTS import BatchDefMLBanditTS #class

from batchDefinition.ml1m.batchDefMLContextDHondt import BatchDefMLContextDHondt #class
from batchDefinition.ml1m.batchDefMLContextDHondtINF import BatchDefMLContextDHondtINF #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondtThompsonSampling import BatchDefMLFuzzyDHondtThompsonSampling #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtThompsonSamplingINF import BatchDefMLFuzzyDHondtThompsonSamplingINF #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtINF import BatchMLFuzzyDHondtINF #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondtDirectOptimize import BatchDefMLFuzzyDHondtDirectOptimize #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtDirectOptimizeINF import BatchDefMLFuzzyDHondtDirectOptimizeINF #class

from batchDefinition.ml1m.batchDefMLWeightedAVG import BatchDefMLWeightedAVG #class
from batchDefinition.ml1m.batchDefMLWeightedAVGMMR import BatchDefMLWeightedAVGMMR #class
from batchDefinition.ml1m.batchDefMLFAI import BatchDefMLFAI #cass

from batchDefinition.ml1m.batchDefMLRandomRecsSwitching import BatchDefMLRandomRecsSwitching #class
from batchDefinition.ml1m.batchDefMLRandomKfromN import BatchDefMLRandomKfromN #class

from batchDefinition.ml1m.batchDefMLContextFuzzyDHondtDirectOptimize import BatchDefMLContextFuzzyDHondtDirectOptimize #class
from batchDefinition.ml1m.batchDefMLContextFuzzyDHondtDirectOptimizeINF import BatchDefMLContextFuzzyDHondtDirectOptimizeINF #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondtDirectOptimizeThompsonSampling import BatchDefMLFuzzyDHondtDirectOptimizeThompsonSampling #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingINF import BatchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingINF #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR import BatchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondtThompsonSamplingINF import BatchDefMLFuzzyDHondtThompsonSamplingINF #class

from batchDefinition.ml1m.batchDefMLSingle import BatchDefMLSingle #class
from batchDefinition.ml1m.batchDefMLSingleINF import BatchDefMLSingleINF #class

from batchDefinition.ml1m.batchDefMLSingleBPRMFHT import BatchDefMLSingleBPRMFHT #class
from batchDefinition.ml1m.batchDefMLSingleW2VHT import BatchDefMLSingleW2VHT #class
from batchDefinition.ml1m.batchDefMLSingleCosineCBHT import BatchDefMLSingleCosineCBHT #class
from batchDefinition.ml1m.batchDefMLSingleBPRMFHT import BatchDefMLSingleBPRMFHT  #class
from batchDefinition.ml1m.batchDefMLSingleVMContextKNNHT import BatchDefMLSingleVMContextKNNHT #class

from batchDefinition.retailrocket.batchDefRRSingle import BatchDefRRSingle #class
from batchDefinition.retailrocket.batchDefRRSingleW2VHT import BatchDefRRSingleW2VHT #class
from batchDefinition.retailrocket.batchDefRRSingleVMContextKNNHT import BatchDefRRSingleVMContextKNNHT #class
from batchDefinition.retailrocket.batchDefRRSingleCosineCBHT import BatchDefRRSingleCosineCBHT #class

from batchDefinition.slanTour.batchDefSTFuzzyDHondtDirectOptimizeThompsonSampling import BatchDefSTFuzzyDHondtDirectOptimizeThompsonSampling #class
from batchDefinition.slanTour.batchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingINF import BatchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingINF #class
from batchDefinition.slanTour.batchDefSTContextFuzzyDHondtDirectOptimizeINF import BatchDefSTContextFuzzyDHondtDirectOptimizeINF #class

from batchDefinition.slanTour.batchDefSTContextDHondt import BatchDefSTContextDHondt #class
from batchDefinition.slanTour.batchDefSTContextDHondtINF import BatchDefSTContextDHondtINF #class
from batchDefinition.slanTour.batchDefSTFuzzyDHondtThompsonSampling import BatchDefSTFuzzyDHondtThompsonSampling #class
from batchDefinition.slanTour.batchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingMMR import BatchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingMMR #class
from batchDefinition.slanTour.batchDefSTFuzzyDHondtThompsonSamplingINF import BatchDefSTDFuzzyHondtThompsonSamplingINF #class
from batchDefinition.slanTour.batchDefSTFuzzyDHondtDirectOptimize import BatchDefSTFuzzyDHondtDirectOptimize #class
from batchDefinition.slanTour.batchDefSTContextFuzzyDHondtDirectOptimize import BatchDefSTContextFuzzyDHondtDirectOptimize #class

from batchDefinition.slanTour.batchDefSTFuzzyDHondtINF import BatchDefSTFuzzyDHondtINF #class
from batchDefinition.slanTour.batchDefSTFuzzyDHondt import BatchDefSTFuzzyDHondt #class
from batchDefinition.slanTour.batchDefSTWeightedAVG import BatchDefSTWeightedAVG #class
from batchDefinition.slanTour.batchDefSTWeightedAVGMMR import BatchDefSTWeightedAVGMMR #class
from batchDefinition.slanTour.batchDefSTFAI import BatchDefSTFAI #cass

from batchDefinition.slanTour.batchDefSTBanditTS import BatchDefSTBanditTS #class
from batchDefinition.slanTour.batchDefSTRandomRecsSwitching import BatchDefSTRandomRecsSwitching #class
from batchDefinition.slanTour.batchDefSTRandomKfromN import BatchDefSTRandomKfromN #class

from batchDefinition.slanTour.batchDefSTSingleVMContextKNNHT import BatchDefSTSingleVMContextKNNHT #class
from batchDefinition.slanTour.batchDefSTSingleBPRMFHT import BatchDefSTSingleBPRMFHT #class
from batchDefinition.slanTour.batchDefSTSingle import BatchDefSTSingle #class
from batchDefinition.slanTour.batchDefSTSingleW2VHT import BatchDefSTSingleW2VHT #class
from batchDefinition.slanTour.batchDefSTSingleCosineCBHT import BatchDefSTSingleCosineCBHT #class


def getBatchInstance(batchStr):
    return eval(batchStr)()


def getAllBatches():
    print("Get All Batches")

    batchesDef:List[ABatchDefinition] = []

    # ML
    batchDefMLBanditTS = BatchDefMLBanditTS()
    batchDefMLFuzzyDHondt = BatchDefMLFuzzyDHondt()
    batchMLFuzzyDHondtINF = BatchMLFuzzyDHondtINF()
    batchDefMLFuzzyDHondtThompsonSampling = BatchDefMLFuzzyDHondtThompsonSampling()
    batchDefMLFuzzyDHondtThompsonSamplingINF = BatchDefMLFuzzyDHondtThompsonSamplingINF()

    batchDefMLFuzzyDHondtDirectOptimize = BatchDefMLFuzzyDHondtDirectOptimize()
    batchDefMLFuzzyDHondtDirectOptimizeINF = BatchDefMLFuzzyDHondtDirectOptimizeINF()

    batchDefMLContextDHondt = BatchDefMLContextDHondt()

    batchDefMLWeightedAVG = BatchDefMLWeightedAVG()
    batchDefMLWeightedAVGMMR = BatchDefMLWeightedAVGMMR()
    batchDefMLFAI = BatchDefMLFAI()

    batchDefMLRandomRecsSwitching = BatchDefMLRandomRecsSwitching()
    batchDefMLRandomKfromN = BatchDefMLRandomKfromN()

    batchDefMLSingle = BatchDefMLSingle()
    batchDefMLSingleINF = BatchDefMLSingleINF()

    batchDefMLSingleBPRMFHT = BatchDefMLSingleBPRMFHT()
    batchDefMLSingleW2VHT = BatchDefMLSingleW2VHT()
    batchDefMLSingleCosineCBHT = BatchDefMLSingleCosineCBHT()
    batchDefMLSingleVMContextKNNHT = BatchDefMLSingleVMContextKNNHT()

    batchDefMLContextFuzzyDHondtDirectOptimize = BatchDefMLContextFuzzyDHondtDirectOptimize()


    batchesDef.append(batchDefMLBanditTS)
    batchesDef.append(batchDefMLFuzzyDHondt)
    batchesDef.append(batchMLFuzzyDHondtINF)
    batchesDef.append(batchDefMLFuzzyDHondtThompsonSampling)
    batchesDef.append(batchDefMLFuzzyDHondtThompsonSamplingINF)
    batchesDef.append(batchDefMLFuzzyDHondtDirectOptimize)
    batchesDef.append(batchDefMLFuzzyDHondtDirectOptimizeINF)
    batchesDef.append(batchDefMLContextDHondt)
    batchesDef.append(batchDefMLWeightedAVG)
    batchesDef.append(batchDefMLWeightedAVGMMR)
    batchesDef.append(batchDefMLFAI)
    batchesDef.append(batchDefMLRandomRecsSwitching)
    batchesDef.append(batchDefMLRandomKfromN)
    batchesDef.append(batchDefMLSingle)
    batchesDef.append(batchDefMLSingleINF)
    batchesDef.append(batchDefMLSingleBPRMFHT)
    batchesDef.append(batchDefMLSingleW2VHT)
    batchesDef.append(batchDefMLSingleCosineCBHT)
    batchesDef.append(batchDefMLSingleVMContextKNNHT)
    batchesDef.append(batchDefMLContextFuzzyDHondtDirectOptimize)

    # RR
    batchDefRRSingle = BatchDefRRSingle()
    batchDefRRSingleW2VHT = BatchDefRRSingleW2VHT()
    batchDefRRSingleCosineCBHT = BatchDefRRSingleCosineCBHT()

    batchesDef.append(batchDefRRSingle)
    batchesDef.append(batchDefRRSingleW2VHT)
    batchesDef.append(batchDefRRSingleCosineCBHT)


    #ST
    batchDefSTContextDHondtINF = BatchDefSTContextDHondtINF()
    batchDefSTContextDHondt = BatchDefSTContextDHondt()
    batchDefSTFuzzyDHondtThompsonSampling = BatchDefSTFuzzyDHondtThompsonSampling()
    batchDefSTDFuzzyHondtThompsonSamplingINF = BatchDefSTDFuzzyHondtThompsonSamplingINF()
    batchDefSTFuzzyDHondtDirectOptimize = BatchDefSTFuzzyDHondtDirectOptimize()
    batchDefSTFuzzyDHondt = BatchDefSTFuzzyDHondt()
    batchDefSTFuzzyDHondtINF = BatchDefSTFuzzyDHondtINF()
    batchDefSTWeightedAVG = BatchDefSTWeightedAVG()
    batchDefSTFAI = BatchDefSTFAI()

    batchDefSTBanditTS = BatchDefSTBanditTS()
    batchDefSTRandomRecsSwitching = BatchDefSTRandomRecsSwitching()
    batchDefSTRandomKfromN = BatchDefSTRandomKfromN()

    batchDefSTSingle = BatchDefSTSingle()
    batchDefSTSingleBPRMFHT = BatchDefSTSingleBPRMFHT()
    batchDefSTSingleW2VHT = BatchDefSTSingleW2VHT()
    batchDefSTSingleCosineCBHT = BatchDefSTSingleCosineCBHT()
    batchDefSTSingleVMContextKNNHT = BatchDefSTSingleVMContextKNNHT()

    batchDefSTContextFuzzyDHondtDirectOptimize = BatchDefSTContextFuzzyDHondtDirectOptimize()


    batchesDef.append(batchDefSTContextDHondtINF)
    batchesDef.append(batchDefSTContextDHondt)
    batchesDef.append(batchDefSTFuzzyDHondtThompsonSampling)
    batchesDef.append(batchDefSTDFuzzyHondtThompsonSamplingINF)
    batchesDef.append(batchDefSTFuzzyDHondtDirectOptimize)
    batchesDef.append(batchDefSTFuzzyDHondt)
    batchesDef.append(batchDefSTFuzzyDHondtINF)
    batchesDef.append(batchDefSTWeightedAVG)
    batchesDef.append(batchDefSTFAI)
    batchesDef.append(batchDefSTBanditTS)
    batchesDef.append(batchDefSTRandomRecsSwitching)
    batchesDef.append(batchDefSTRandomKfromN)
    batchesDef.append(batchDefSTSingle)
    batchesDef.append(batchDefSTSingleBPRMFHT)
    batchesDef.append(batchDefSTSingleW2VHT)
    batchesDef.append(batchDefSTSingleCosineCBHT)
    batchesDef.append(batchDefSTSingleVMContextKNNHT)
    batchesDef.append(batchDefSTContextFuzzyDHondtDirectOptimize)

    return batchesDef


def getBatchesJournal():
    print("Get Journal Batches")

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

    batchesDef:List[ABatchDefinition] = []

    # ML #############################################################################
    batchDefMLSingle = BatchDefMLSingle()
#
    batchDefMLBanditTS = BatchDefMLBanditTS()
    batchDefMLBanditTS.selectorIDs = [BatchDefMLFuzzyDHondt.SLCTR_FIXED]
#
    batchDefMLWeightedAVG = BatchDefMLWeightedAVG()
    batchDefMLWeightedAVG.lrClicks:List[float] = [0.03]
    batchDefMLWeightedAVG.lrViewDivisors:List[float] = [250]
#
    batchDefMLWeightedAVGMMR = BatchDefMLWeightedAVGMMR()
    batchDefMLWeightedAVGMMR.lrClicks:List[float] = [0.03]
    batchDefMLWeightedAVGMMR.lrViewDivisors:List[float] = [250]

    batchDefMLFAI = BatchDefMLFAI()
#
    batchDefMLRandomRecsSwitching = BatchDefMLRandomRecsSwitching()
#
    batchDefMLRandomKfromN = BatchDefMLRandomKfromN()
#
    batchDefMLFuzzyDHondt = BatchDefMLFuzzyDHondt()
    batchDefMLFuzzyDHondt.lrClicks:List[float] = [0.03]
    batchDefMLFuzzyDHondt.lrViewDivisors:List[float] = [250]
    batchDefMLFuzzyDHondt.selectorIDs:List[str] = [BatchDefMLFuzzyDHondt.SLCTR_FIXED]

#
    batchDefMLFuzzyDHondtThompsonSampling = BatchDefMLFuzzyDHondtThompsonSampling()
    batchDefMLFuzzyDHondtThompsonSampling.selectorIDs = [BatchDefMLFuzzyDHondt.SLCTR_FIXED]
#
    batchDefMLContextDHondt = BatchDefMLContextDHondt()
    batchDefMLContextDHondt.selectorIDs:List[str] = [BatchDefMLFuzzyDHondt.SLCTR_FIXED]
#
    batchDefMLFuzzyDHondtDirectOptimize = BatchDefMLFuzzyDHondtDirectOptimize()
    batchDefMLFuzzyDHondtDirectOptimize.lrClicks:List[float] = [0.03]
    batchDefMLFuzzyDHondtDirectOptimize.lrViewDivisors:List[float] = [250]
    batchDefMLFuzzyDHondtDirectOptimize.selectorIds = [
        BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_FIXED, BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE1,
        BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE2, BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE3,
        BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE4, BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE5]
#
    batchDefMLFuzzyDHondtDirectOptimizeThompsonSampling = BatchDefMLFuzzyDHondtDirectOptimizeThompsonSampling()
    batchDefMLFuzzyDHondtDirectOptimizeThompsonSampling.selectorIDs = [BatchDefMLFuzzyDHondt.SLCTR_FIXED]
#
    batchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR = BatchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR()
    batchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR.selectorIDs = [BatchDefMLFuzzyDHondt.SLCTR_FIXED]
#
    batchDefMLContextFuzzyDHondtDirectOptimize = BatchDefMLContextFuzzyDHondtDirectOptimize()

#
    # INF
    batchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingINF = BatchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingINF()
    batchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingINF.selectorIDs = [BatchDefMLFuzzyDHondt.SLCTR_FIXED]
    batchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingINF.jobIDs = [
        jobIdML01, jobIdML02, jobIdML03, jobIdML04, jobIdML05, jobIdML06, jobIdML07, jobIdML08]
#
    batchDefMLContextDHondtINF = BatchDefMLContextDHondtINF()
    batchDefMLContextDHondtINF.jobIDs = [
        jobIdML01, jobIdML02, jobIdML03, jobIdML04, jobIdML05, jobIdML06, jobIdML07, jobIdML08]
#
    batchDefMLContextFuzzyDHondtDirectOptimizeINF = BatchDefMLContextFuzzyDHondtDirectOptimizeINF()
    batchDefMLContextFuzzyDHondtDirectOptimizeINF.jobIDs = [
        jobIdML01, jobIdML02, jobIdML03, jobIdML04, jobIdML05, jobIdML06, jobIdML07, jobIdML08]

    batchDefMLFuzzyDHondtThompsonSamplingINF = BatchDefMLFuzzyDHondtThompsonSamplingINF()
    batchDefMLFuzzyDHondtThompsonSamplingINF.jobIDs = [
        jobIdML01, jobIdML02, jobIdML03, jobIdML04, jobIdML05, jobIdML06, jobIdML07, jobIdML08]
#
    batchesDef.append(batchDefMLSingle)
    batchesDef.append(batchDefMLBanditTS)
    batchesDef.append(batchDefMLWeightedAVG)
    batchesDef.append(batchDefMLWeightedAVGMMR)
    batchesDef.append(batchDefMLWeightedAVGMMR)
    batchesDef.append(batchDefMLFAI)
    batchesDef.append(batchDefMLRandomRecsSwitching)
    batchesDef.append(batchDefMLRandomKfromN)
    batchesDef.append(batchDefMLFuzzyDHondt)
    batchesDef.append(batchDefMLFuzzyDHondtThompsonSampling)
    batchesDef.append(batchDefMLContextDHondt)
    batchesDef.append(batchDefMLFuzzyDHondtDirectOptimize)
    batchesDef.append(batchDefMLFuzzyDHondtDirectOptimizeThompsonSampling)
    batchesDef.append(batchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR)
    batchesDef.append(batchDefMLContextFuzzyDHondtDirectOptimize)

    batchesDef.append(batchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingINF)
    batchesDef.append(batchDefMLContextDHondtINF)
    batchesDef.append(batchDefMLContextFuzzyDHondtDirectOptimizeINF)
    batchesDef.append(batchDefMLFuzzyDHondtThompsonSamplingINF)

    # RR #############################################################################
    batchDefRRSingle = BatchDefRRSingle()
    batchDefRRSingleW2VHT = BatchDefRRSingleW2VHT()
    batchDefRRSingleVMContextKNNHT = BatchDefRRSingleVMContextKNNHT()
    batchDefRRSingleCosineCBHT = BatchDefRRSingleCosineCBHT()

    #batchesDef.append(batchDefRRSingle)
    #batchesDef.append(batchDefRRSingleW2VHT)
    #batchesDef.append(batchDefRRSingleVMContextKNNHT)
    #batchesDef.append(batchDefRRSingleCosineCBHT)


    # ST #############################################################################
    batchDefSTSingle = BatchDefSTSingle()
#
    batchDefSTBanditTS = BatchDefSTBanditTS()
    batchDefSTBanditTS.selectorIDs = [BatchDefMLFuzzyDHondt.SLCTR_FIXED]

    batchDefSTWeightedAVG = BatchDefSTWeightedAVG()
    batchDefSTWeightedAVG.lrClicks:List[float] = [0.03]
    batchDefSTWeightedAVG.lrViewDivisors:List[float] = [250]

    batchDefSTFAI = BatchDefSTFAI()

#
    batchDefSTRandomRecsSwitching = BatchDefSTRandomRecsSwitching()
    batchDefSTRandomKfromN = BatchDefSTRandomKfromN()
#
    batchDefSTFuzzyDHondt = BatchDefSTFuzzyDHondt()
    batchDefSTFuzzyDHondt.lrClicks:List[float] = [0.03]
    batchDefSTFuzzyDHondt.lrViewDivisors:List[float] = [250]
    batchDefSTFuzzyDHondt.selectorIDs:List[str] = [BatchDefMLFuzzyDHondt.SLCTR_FIXED]

#
    batchDefSTFuzzyDHondtThompsonSampling = BatchDefSTFuzzyDHondtThompsonSampling()
    batchDefSTContextDHondt = BatchDefSTContextDHondt()
    batchDefSTContextDHondt.selectorIDs:List[str] = [BatchDefMLFuzzyDHondt.SLCTR_FIXED]
#
    batchDefSTContextFuzzyDHondtDirectOptimize = BatchDefSTContextFuzzyDHondtDirectOptimize()
#
    batchDefSTFuzzyDHondtDirectOptimize = BatchDefSTFuzzyDHondtDirectOptimize()
    batchDefSTFuzzyDHondtDirectOptimize.lrClicks:List[float] = [0.03]
    batchDefSTFuzzyDHondtDirectOptimize.lrViewDivisors:List[float] = [250]
    batchDefSTFuzzyDHondtDirectOptimize.selectorIDs = [
        BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_FIXED, BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE1,
        BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE2, BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE3,
        BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE4, BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE5]
#
    batchDefSTFuzzyDHondtDirectOptimizeThompsonSampling = BatchDefSTFuzzyDHondtDirectOptimizeThompsonSampling()
    batchDefSTFuzzyDHondtDirectOptimizeThompsonSampling.selectorIDs = [BatchDefMLFuzzyDHondt.SLCTR_FIXED]
#
    batchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingMMR = BatchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingMMR()
    batchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingMMR.selectorIDs = [BatchDefMLFuzzyDHondt.SLCTR_FIXED]


#
    # INF
    batchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingINF = BatchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingINF()
    batchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingINF.selectorIDs = [BatchDefMLFuzzyDHondt.SLCTR_FIXED]
    batchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingINF.jobIDs = [
        jobIdST01, jobIdST02, jobIdST03, jobIdST04, jobIdST05, jobIdST06, jobIdST07, jobIdST08]

    batchDefSTContextDHondtINF = BatchDefSTContextDHondtINF()
    batchDefSTContextDHondtINF.jobIDs = [
        jobIdST01, jobIdST02, jobIdST03, jobIdST04, jobIdST05, jobIdST06, jobIdST07, jobIdST08]

    batchDefSTContextFuzzyDHondtDirectOptimizeINF = BatchDefSTContextFuzzyDHondtDirectOptimizeINF()
    batchDefSTContextFuzzyDHondtDirectOptimizeINF.jobIDs = [
        jobIdST01, jobIdST02, jobIdST03, jobIdST04, jobIdST05, jobIdST06, jobIdST07, jobIdST08]

    batchDefSTDFuzzyHondtThompsonSamplingINF = BatchDefSTDFuzzyHondtThompsonSamplingINF()
    batchDefSTDFuzzyHondtThompsonSamplingINF.jobIDs = [
        jobIdST01, jobIdST02, jobIdST03, jobIdST04, jobIdST05, jobIdST06, jobIdST07, jobIdST08]



    batchesDef.append(batchDefSTSingle)
    batchesDef.append(batchDefSTBanditTS)
    batchesDef.append(batchDefSTWeightedAVG)
    batchesDef.append(batchDefSTFAI)
    batchesDef.append(batchDefSTRandomRecsSwitching)
    batchesDef.append(batchDefSTRandomKfromN)
    batchesDef.append(batchDefSTFuzzyDHondt)

    batchesDef.append(batchDefSTFuzzyDHondtThompsonSampling)
    batchesDef.append(batchDefSTContextDHondt)
    batchesDef.append(batchDefSTContextFuzzyDHondtDirectOptimize)

    batchesDef.append(batchDefSTFuzzyDHondtDirectOptimize)
    batchesDef.append(batchDefSTFuzzyDHondtDirectOptimizeThompsonSampling)
    batchesDef.append(batchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingMMR)

    batchesDef.append(batchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingINF)
    batchesDef.append(batchDefSTContextDHondtINF)
    batchesDef.append(batchDefSTContextFuzzyDHondtDirectOptimizeINF)
    batchesDef.append(batchDefSTDFuzzyHondtThompsonSamplingINF)

    return batchesDef


def getBatchesJournal2():
    print("Get Journal Batches 2")

    batchesDef:List[ABatchDefinition] = []

    # ML #############################################################################
    batchDefRRSingle = BatchDefRRSingle()

    batchesDef.append(batchDefRRSingle)
    return batchesDef


def generateBatches():
    print("Generate Batches")

    np.random.seed(42)
    random.seed(42)

    #batchesDef: List = getAllBatches()
    batchesDef:List = getBatchesJournal()
    #batchesDef:List = getBatchesJournal2()

    for batchDefI in batchesDef:
        batchDefI.generateAllBatches()



if __name__ == "__main__":

  os.chdir("..")
  print(os.getcwd())

  generateBatches()

