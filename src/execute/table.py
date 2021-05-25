#!/usr/bin/python3

import time
import sys
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import random
import numpy as np

from typing import List
from batchDefinition.aBatchDefinition import ABatchDefinition

from execute.generateBatches import getBatchesJournal #function

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

from batchDefinition.ml1m.batchDefMLRandomRecsSwitching import BatchDefMLRandomRecsSwitching #class
from batchDefinition.ml1m.batchDefMLRandomKfromN import BatchDefMLRandomKfromN #class

from batchDefinition.ml1m.batchDefMLContextFuzzyDHondtDirectOptimize import BatchDefMLContextFuzzyDHondtDirectOptimize #class
from batchDefinition.ml1m.batchDefMLContextFuzzyDHondtDirectOptimizeINF import BatchDefMLContextFuzzyDHondtDirectOptimizeINF #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondtThompsonSamplingINF import BatchDefMLFuzzyDHondtThompsonSamplingINF #class


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
from batchDefinition.slanTour.batchDefSTBanditTS import BatchDefSTBanditTS #class
from batchDefinition.slanTour.batchDefSTRandomRecsSwitching import BatchDefSTRandomRecsSwitching #class
from batchDefinition.slanTour.batchDefSTRandomKfromN import BatchDefSTRandomKfromN #class

from batchDefinition.slanTour.batchDefSTSingleVMContextKNNHT import BatchDefSTSingleVMContextKNNHT #class
from batchDefinition.slanTour.batchDefSTSingleBPRMFHT import BatchDefSTSingleBPRMFHT #class
from batchDefinition.slanTour.batchDefSTSingle import BatchDefSTSingle #class
from batchDefinition.slanTour.batchDefSTSingleW2VHT import BatchDefSTSingleW2VHT #class
from batchDefinition.slanTour.batchDefSTSingleCosineCBHT import BatchDefSTSingleCosineCBHT #class

from execute.resultsVerification import readTheLastComputationResult #class
from batch.batch import Batch #class



def generateTables():
    print("Generate Table")

    batchesDef:List[ABatchDefinition] = getBatchesJournal()

    numbersR:List[int] = 3

    batchDefClasses:List[str] = [BatchDefMLSingle,

        BatchDefMLBanditTS, BatchDefMLWeightedAVG, BatchDefMLWeightedAVGMMR,
        BatchDefMLRandomRecsSwitching, BatchDefMLRandomKfromN, BatchDefMLFuzzyDHondt,
        BatchDefMLFuzzyDHondtThompsonSampling, BatchDefMLContextDHondt, BatchDefMLFuzzyDHondtDirectOptimize,
        BatchDefMLFuzzyDHondtDirectOptimizeThompsonSampling, BatchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR,
        BatchDefMLContextFuzzyDHondtDirectOptimize,

        BatchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingINF, BatchDefMLContextDHondtINF,
        BatchDefMLContextFuzzyDHondtDirectOptimizeINF, BatchDefMLFuzzyDHondtThompsonSamplingINF]


    behaviourIDs:List[str] = ["linear0109", "static08", "powerlaw054min048"]

    for behaviourIDI in behaviourIDs:
        generateTable(batchesDef, batchDefClasses, behaviourIDI, numbersR)


    batchDefClasses: List[str] = [BatchDefSTSingle,

        #BatchDefSTBanditTS, BatchDefSTWeightedAVG, BatchDefSTWeightedAVGMMR,
        BatchDefSTBanditTS, BatchDefSTWeightedAVG,
        BatchDefSTRandomRecsSwitching, BatchDefSTRandomKfromN, BatchDefSTFuzzyDHondt,
        #BatchDefSTFuzzyDHondtThompsonSampling, BatchDefSTContextDHondt, BatchDefSTFuzzyDHondtDirectOptimize,
        BatchDefSTFuzzyDHondtThompsonSampling, BatchDefSTContextDHondt,
        #BatchDefSTFuzzyDHondtDirectOptimizeThompsonSampling, BatchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingMMR,
        BatchDefSTContextFuzzyDHondtDirectOptimize,

        BatchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingINF, BatchDefSTContextDHondtINF,
        BatchDefSTContextFuzzyDHondtDirectOptimizeINF, BatchDefSTDFuzzyHondtThompsonSamplingINF]

    behaviourIDs:List[str] = ["linear0109", "static08", "powerlaw054min048"]

    for behaviourIDI in behaviourIDs:
        generateTable(batchesDef, batchDefClasses, behaviourIDI, numbersR)



def generateTable(batchesDef, batchDefClasses, behaviourID, numbersR):

    string:str = "\\begin{tabular}" + "\n" \
        + "  {l | c | c | c}" + "\n" \
        + "  Name " + behaviourID + " & R1 & R2 & R3" + "\n" \
        + "  \\hline" + "\n"

    for batchDefClassI in batchDefClasses:
        resultsDict:dict = getResults(batchesDef, batchDefClassI, behaviourID, numbersR)
        for nameI, resultsI in zip(resultsDict.keys(), resultsDict.values()):
            string += convert(nameI, resultsI)

    string += "\\end{tabular}"

    print("\n")
    print(string)


def convert(nameI:str, results:List[int]):
   return "  " + nameI + " & " + results[0] + " & " + results[1] + " & " + results[2] + "\n"


def getResults(batchesDef:List[ABatchDefinition], batchDefMLClass, behavID:str, countR:int):

    batchDef:ABatchDefinition = [batchDefI for batchDefI in batchesDef if isinstance(batchDefI, batchDefMLClass)][0]

    batches:List[Batch] = [batchI for batchI in batchDef.getAllBatches() if behavID in batchI.getFileName()]

    numberOfJobs:int = int(len(batches) / countR)
    resultsDict:dict = {}

    for rNumberI in range(countR):
        for batchIndex in range(numberOfJobs):
            indexI:int = rNumberI*numberOfJobs + batchIndex
            batchI:Batch = batches[indexI]
            resultI = readTheLastComputationResult(batchI.getFileName())
            #print(resultI)
            nameI:str = resultI[1]
            resultValueI:int = resultI[2]

            resultsI = resultsDict.get(nameI, [])
            resultsI.append(resultValueI)
            resultsDict[nameI] = resultsI

    return resultsDict



if __name__ == "__main__":

  os.chdir("..")
  print(os.getcwd())

  generateTables()