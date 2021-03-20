#!/usr/bin/python3

import time
import sys
import os

import random
import traceback
import numpy as np

from typing import List

from input.batchesML1m.batchDefMLBanditTS import BatchDefMLBanditTS #class
from input.batchesML1m.batchDefMLContextDHondt import BatchDefMLContextDHondt #class
from input.batchesML1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class
from input.batchesML1m.batchDefMLFuzzyDHondtThompsonSampling import BatchDefMLFuzzyDHondtThompsonSampling #class
from input.batchesML1m.batchDefMLFuzzyDHondtINF import BatchMLFuzzyDHondtINF #class
from input.batchesML1m.batchDefMLFuzzyDHondtThompsonSamplingINF import BatchDefMLFuzzyDHondtThompsonSamplingINF #class
from input.batchesML1m.batchDefMLFuzzyDHondtDirectOptimize import BatchDefMLFuzzyDHondtDirectOptimize #class
from input.batchesML1m.batchDefMLFuzzyDHondtDirectOptimizeINF import BatchDefMLFuzzyDHondtDirectOptimizeINF #class
from input.batchesML1m.batchDefMLSingle import BatchDefMLSingle #class
from input.batchesML1m.batchDefMLSingleINF import BatchDefMLSingleINF #class

from input.batchesML1m.batchDefMLWeightedAVG import BatchDefMLWeightedAVG #class

from input.batchesML1m.batchDefMLSingle import BatchDefMLSingle #class
from input.batchesML1m.batchDefMLSingleW2VHT import BatchDefMLSingleW2VHT #class
from input.batchesML1m.batchDefMLSingleCosineCBHT import BatchDefMLSingleCosineCBHT #class
from input.batchesML1m.batchDefMLSingleBPRMFHT import BatchDefMLSingleBPRMFHT  #class
from input.batchesML1m.batchDefMLSingleVMContextKNNHT import BatchDefMLSingleVMContextKNNHT #class

from input.batchesRetailrocket.batchDefRRSingle import BatchDefRRSingle #class
from input.batchesRetailrocket.batchDefRRSingleW2VHT import BatchDefRRSingleW2VHT #class

from input.batchesSlanTour.batchDefSTContextDHondtINF import BatchDefSTContextDHondtINF #class
from input.batchesSlanTour.batchDefSTContextDHondt import BatchDefSTContextDHondt #class
from input.batchesSlanTour.batchDefSTFuzzyDHondtThompsonSamplingINF import BatchDefSTDFuzzyHondtThompsonSamplingINF #class
from input.batchesSlanTour.batchDefSTFuzzyDHondtThompsonSampling import BatchDefSTFuzzyDHondtThompsonSampling #class
from input.batchesSlanTour.batchDefSTFuzzyDHondtThompsonSamplingINF import BatchDefSTDFuzzyHondtThompsonSamplingINF #class
from input.batchesSlanTour.batchDefSTFuzzyDHondtDirectOptimize import BatchDefSTFuzzyDHondtDirectOptimize #class
from input.batchesSlanTour.batchDefSTFuzzyDHondt import BatchDefSTFuzzyDHondt #class
from input.batchesSlanTour.batchDefSTFuzzyDHondtINF import BatchDefSTFuzzyDHondtINF #class
from input.batchesSlanTour.batchDefSTWeightedAVG import BatchDefSTWeightedAVG #class
from input.batchesSlanTour.batchDefSTBanditTS import BatchDefSTBanditTS #class

from input.batchesSlanTour.batchDefSTSingleVMContextKNNHT import BatchDefSTSingleVMContextKNNHT #class
from input.batchesSlanTour.batchDefSTSingle import BatchDefSTSingle #class
from input.batchesSlanTour.batchDefSTSingleBPRMFHT import BatchDefSTSingleBPRMFHT #class
from input.batchesSlanTour.batchDefSTSingleW2VHT import BatchDefSTSingleW2VHT #class
from input.batchesSlanTour.batchDefSTSingleCosineCBHT import BatchDefSTSingleCosineCBHT #class


def filter():
    # ML
    #iJobIds:List[str] = list(BatchMLSingle.getParameters().keys()) +\
    #                 ["RecommenderW2V" + rIdI for rIdI in BatchMLSingleW2VHT.getParameters().keys()] + \
    #                 ["RecommenderBPRMF" + rIdI for rIdI in BatchMLSingleBPRMFHT.getParameters().keys()] +\
    #                 ["RecommendervmContextKNN" + rIdI for rIdI in BatchMLVMContextKNNHT.getParameters().keys()] +\
    #                 ["RecommenderCosineCB" + rIdI for rIdI in BatchMLSingleCosineCBHT.getParameters().keys()]

    # ST
    iJobIds:List[str] = list(BatchDefSTSingle.getParameters().keys()) + \
                        ["RecommenderW2V" + rIdI for rIdI in BatchDefSTSingleW2VHT.getParameters().keys()] + \
                        ["RecommenderBPRMF" + rIdI for rIdI in BatchDefSTSingleBPRMFHT.getParameters().keys()] + \
                        ["RecommendervmContextKNN" + rIdI for rIdI in BatchDefSTSingleVMContextKNNHT.getParameters().keys()] + \
                        ["RecommenderCosineCB" + rIdI for rIdI in BatchDefSTSingleCosineCBHT.getParameters().keys()]


    iJobIds:List[str] = ["FuzzyDHondtThompsonSamplingINF" + rIdI for rIdI in BatchDefMLFuzzyDHondtThompsonSamplingINF.getParameters().keys()]
    #iJobIds:List[str] = ["DHondtThompsonSamplingINF" + rIdI for rIdI in BatchSTDHondtThompsonSamplingINF.getParameters().keys()]
    #print(iJobIds)


    # RR
    #iJobIds:List[str] = list([]) +\
    #                 ["FuzzyDHondtThompsonSamplingINF" + rIdI for rIdI in BatchRRSingleW2VHT.getParameters().keys()]



    #dir:str = "/home/stepan/aaa/ml1mDiv80Ulinear0109R1"
    #dir:str = "/home/stepan/aaa/stDiv80Ulinear0109R1"
    #dir:str = "/home/stepan/aaa/stDiv80Ulinear0109R2"
    #dir:str = "/home/stepan/aaa/ddd/rrDiv80Ulinear0109R2"
    dir:str = "/home/stepan/aaa/eee"

    # reading evaluation
    evaluationFile = open(dir + os.sep + 'evaluation.txt', 'r')
    eLines:List[str] = [eLineI.replace("\n", "") for eLineI in evaluationFile.readlines()]
    eRows:List[int] = [(
         eLines[i*3].replace("ids: ['", "").replace("']", ""),
         eLines[i*3+1].replace("[[{'clicks': ", "").replace("}]]", ""))
         for i in range(int(len(eLines)/3.0))]

    eJobIds:List[str] = [fileIdI for fileIdI, clicksI in eRows]
    #print(eJobIds[200])

    print(eJobIds)
    #print(len(eJobIds))

    selectedFiles:List[str] = []
    for fileNameI in iJobIds:
        #print("ffffffffffffffffffff")
        print(fileNameI)
        #print("ffffffffffffffffffff")
        if not fileNameI in eJobIds:
            selectedFiles.append(fileNameI)
            #print(fileNameI)

    jobFiles:List[str] = [jobIdI for jobIdI in selectedFiles]
    print("Chybejici: " + str(len(jobFiles)) + " / " + str(len(iJobIds)))

    for jobFileI in jobFiles:
        print("cp " + jobFileI.replace("FuzzyDHondt","BatchMLFuzzyDHondt") + ".txt ../chybi/")

def rename():
    dir:str = "/home/stepan/aaa/stDiv80Ulinear0109R2_"

    # reading evaluation
    evaluationFile = open(dir + os.sep + 'evaluation.txt', 'r')
    eLines:List[str] = [eLineI.replace("\n", "") for eLineI in evaluationFile.readlines()]
    eRows:List[int] = [(
         eLines[i*3].replace("ids: ['", "").replace("']", ""),
         eLines[i*3+1].replace("[[{'clicks': ", "").replace("}]]", ""))
         for i in range(int(len(eLines)/3.0))]

    fileNames:List[str] = [fileIdI + ".txt" for fileIdI, clicksI in eRows]
    fileNames.remove("ContextDHondtFixed.txt")
    fileNames.remove("ContextDHondtRoulette1.txt")
    fileNames.remove("ContextDHondtRoulette3.txt")

    print(len(fileNames))

    for fileNameI in fileNames:
        fileNameNewI = fileNameI.replace("ReduceOLin", "ReduceProbOLin")
        fileNameNewI = fileNameNewI.replace("ReduceOStat", "ReduceProbOStat")

        cmd1I:str = "mv " + "computation-" + fileNameI + " " + "computation-" + fileNameNewI
        cmd2I:str = "mv " + "portfModelTimeEvolution-" + fileNameI + " " + "portfModelTimeEvolution-" + fileNameNewI
        cmd3I:str = "mv " + "historyOfRecommendation-" + fileNameI + " " + "historyOfRecommendation-" + fileNameNewI
        print(cmd1I)
        print(cmd2I)
        print(cmd3I)


def rename2():

    iJobIds:List[str] = ["FuzzyDHondtThompsonSamplingINF" + rIdI for rIdI in BatchDefMLFuzzyDHondtThompsonSamplingINF.getParameters().keys()]

    for iJobIdI in iJobIds:
       fileNameI:str = iJobIdI.replace("FuzzyDHondt", "DHondt") + ".txt"
       fileNameNewI:str = iJobIdI + ".txt"

       cmd1I: str = "mv " + "computation-" + fileNameI + " " + "computation-" + fileNameNewI
       cmd2I: str = "mv " + "portfModelTimeEvolution-" + fileNameI + " " + "portfModelTimeEvolution-" + fileNameNewI
       cmd3I: str = "mv " + "historyOfRecommendation-" + fileNameI + " " + "historyOfRecommendation-" + fileNameNewI
       print(cmd1I)
       print(cmd2I)
       print(cmd3I)

if __name__ == "__main__":
   os.chdir("..")

   filter()
   #rename2()