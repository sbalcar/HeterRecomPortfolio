#!/usr/bin/python3

import time
import sys
import os

import random
import traceback
import numpy as np

from typing import List

from input.batchesML1m.batchMLBanditTS import BatchMLBanditTS #class
from input.batchesML1m.batchMLContextDHondt import BatchMLContextDHondt #class
from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class
from input.batchesML1m.batchMLFuzzyDHondtThompsonSampling import BatchMLFuzzyDHondtThompsonSampling #class
from input.batchesML1m.batchMLFuzzyDHondtINF import BatchMLFuzzyDHondtINF #class
from input.batchesML1m.batchMLFuzzyDHondtThompsonSamplingINF import BatchMLFuzzyDHondtThompsonSamplingINF #class
from input.batchesML1m.batchMLFuzzyDHondtDirectOptimize import BatchMLFuzzyDHondtDirectOptimize #class
from input.batchesML1m.batchMLFuzzyDHondtDirectOptimizeINF import BatchMLFuzzyDHondtDirectOptimizeINF #class
from input.batchesML1m.batchMLSingle import BatchMLSingle #class
from input.batchesML1m.batchMLSingleINF import BatchMLSingleINF #class

from input.batchesML1m.batchMLWeightedAVG import BatchMLWeightedAVG #class

from input.batchesML1m.batchMLSingle import BatchMLSingle #class
from input.batchesML1m.batchMLSingleW2VHT import BatchMLSingleW2VHT #class
from input.batchesML1m.batchMLSingleCosineCBHT import BatchMLSingleCosineCBHT #class
from input.batchesML1m.batchMLSingleBPRMFHT import BatchMLSingleBPRMFHT  #class
from input.batchesML1m.batchMLSingleVMContextKNNHT import BatchMLVMContextKNNHT #class

from input.batchesRetailrocket.batchRRSingle import BatchRRSingle #class
from input.batchesRetailrocket.batchRRSingleW2VHT import BatchRRSingleW2VHT #class

from input.batchesSlanTour.batchSTContextDHondtINF import BatchSTContextDHondtINF #class
from input.batchesSlanTour.batchSTContextDHondt import BatchSTContextDHondt #class
from input.batchesSlanTour.batchSTFuzzyDHondtThompsonSamplingINF import BatchSTDHondtThompsonSamplingINF #class
from input.batchesSlanTour.batchSTFuzzyDHondtThompsonSampling import BatchSTDHondtThompsonSampling #class
from input.batchesSlanTour.batchSTFuzzyDHondtThompsonSamplingINF import BatchSTDHondtThompsonSamplingINF #class
from input.batchesSlanTour.batchSTFuzzyDHondtDirectOptimize import BatchSTFuzzyDHondtDirectOptimize #class
from input.batchesSlanTour.batchSTFuzzyDHondt import BatchSTFuzzyDHondt #class
from input.batchesSlanTour.batchSTFuzzyDHondtINF import BatchSTFuzzyDHondtINF #class
from input.batchesSlanTour.batchSTWeightedAVG import BatchSTWeightedAVG #class
from input.batchesSlanTour.batchSTBanditTS import BatchSTBanditTS #class

from input.batchesSlanTour.batchSTSingleVMContextKNNHT import BatchSTVMContextKNNHT #class
from input.batchesSlanTour.batchSTSingle import BatchSTSingle #class
from input.batchesSlanTour.batchSTSingleBPRMFHT import BatchSTSingleBPRMFHT #class
from input.batchesSlanTour.batchSTSingleW2VHT import BatchSTSingleW2VHT #class
from input.batchesSlanTour.batchSTSingleCosineCBHT import BatchSTSingleCosineCBHT #class


def filter():
    # ML
    #iJobIds:List[str] = list(BatchMLSingle.getParameters().keys()) +\
    #                 ["RecommenderW2V" + rIdI for rIdI in BatchMLSingleW2VHT.getParameters().keys()] + \
    #                 ["RecommenderBPRMF" + rIdI for rIdI in BatchMLSingleBPRMFHT.getParameters().keys()] +\
    #                 ["RecommendervmContextKNN" + rIdI for rIdI in BatchMLVMContextKNNHT.getParameters().keys()] +\
    #                 ["RecommenderCosineCB" + rIdI for rIdI in BatchMLSingleCosineCBHT.getParameters().keys()]

    # ST
    iJobIds:List[str] = list(BatchSTSingle.getParameters().keys()) +\
                     ["RecommenderW2V" + rIdI for rIdI in BatchSTSingleW2VHT.getParameters().keys()] + \
                     ["RecommenderBPRMF" + rIdI for rIdI in BatchSTSingleBPRMFHT.getParameters().keys()] +\
                     ["RecommendervmContextKNN" + rIdI for rIdI in BatchSTVMContextKNNHT.getParameters().keys()] +\
                     ["RecommenderCosineCB" + rIdI for rIdI in BatchSTSingleCosineCBHT.getParameters().keys()]


    iJobIds:List[str] = ["FuzzyDHondtThompsonSamplingINF" + rIdI for rIdI in BatchMLFuzzyDHondtThompsonSamplingINF.getParameters().keys()]
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

    iJobIds:List[str] = ["FuzzyDHondtThompsonSamplingINF" + rIdI for rIdI in BatchMLFuzzyDHondtThompsonSamplingINF.getParameters().keys()]

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