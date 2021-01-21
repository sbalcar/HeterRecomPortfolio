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
from input.batchesML1m.batchMLDHondtThompsonSampling import BatchMLDHondtThompsonSampling #class
from input.batchesML1m.batchMLFuzzyDHondtINF import BatchMLFuzzyDHondtINF #class
from input.batchesML1m.batchMLDHondtThompsonSamplingINF import BatchMLDHondtThompsonSamplingINF #class
from input.batchesML1m.batchMLFuzzyDHondtDirectOptimize import BatchMLFuzzyDHondtDirectOptimize #class
from input.batchesML1m.batchMLFuzzyDHondtDirectOptimizeINF import BatchMLFuzzyDHondtDirectOptimizeINF #class
from input.batchesML1m.batchMLSingle import BatchMLSingle #class
from input.batchesML1m.batchMLSingleINF import BatchMLSingleINF #class

from input.batchesML1m.batchMLWeightedAVG import BatchMLWeightedAVG #class

from input.batchesML1m.batchMLSingle import BatchMLSingle #class
from input.batchesML1m.batchMLSingle2 import BatchMLSingle2 #class
from input.batchesML1m.batchMLSingleW2VHT import BatchMLSingleW2VHT #class
from input.batchesML1m.batchMLSingleCosineCBHT import BatchMLSingleCosineCBHT #class
from input.batchesML1m.batchMLSingleBPRMFHT import BatchMLSingleBPRMFHT  #class
from input.batchesML1m.batchMLSingleVMContextKNNHT import BatchMLVMContextKNNHT #class

from input.batchesRetailrocket.batchRRSingle import BatchRRSingle #class
from input.batchesRetailrocket.batchRRSingleW2VHT import BatchRRSingleW2VHT #class

from input.batchesSlanTour.batchSTContextDHondtINF import BatchSTContextDHondtINF #class
from input.batchesSlanTour.batchSTContextDHondt import BatchSTContextDHondt #class
from input.batchesSlanTour.batchSTDHondtThompsonSamplingINF import BatchSTDHondtThompsonSamplingINF #class
from input.batchesSlanTour.batchSTDHondtThompsonSampling import BatchSTDHondtThompsonSampling #class
from input.batchesSlanTour.batchSTDHondtThompsonSamplingINF import BatchSTDHondtThompsonSamplingINF #class
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




if __name__ == "__main__":
   os.chdir("..")

   iJobIds:List[str] = list(BatchMLSingle.getParameters().keys()) +\
                     ["RecommenderW2V" + rIdI for rIdI in BatchMLSingleW2VHT.getParameters().keys()] + \
                     ["RecommenderBPRMF" + rIdI for rIdI in BatchSTSingleBPRMFHT.getParameters().keys()] +\
                     ["RecommendervmContextKNN" + rIdI for rIdI in BatchSTVMContextKNNHT.getParameters().keys()] +\
                     ["RecommenderCosineCB" + rIdI for rIdI in BatchSTSingleCosineCBHT.getParameters().keys()]


   dir:str = "/home/stepan/aaa/ml1mDiv80Ulinear0109R1"

   # reading evaluation
   evaluationFile = open(dir + os.sep + 'evaluation.txt', 'r')
   eLines:List[str] = [eLineI.replace("\n", "") for eLineI in evaluationFile.readlines()]
   eRows:List[int] = [(
         eLines[i*3].replace("ids: ['", "").replace("']", ""),
         eLines[i*3+1].replace("[[{'clicks': ", "").replace("}]]", ""))
         for i in range(int(len(eLines)/3.0))]

   eJobIds:List[str] = [fileIdI for fileIdI, clicksI in eRows]
   #print(eJobIds[200])

   selectedFiles:List[str] = []
   for fileNameI in iJobIds:
      if not fileNameI in eJobIds:
         selectedFiles.append(fileNameI)
         #print(fileNameI)


   jobFiles:List[str] = ["BatchMLSingle" + jobIdI + ".txt" for jobIdI in selectedFiles]
   print(len(jobFiles))

   for jobFileI in jobFiles:
    print(jobFileI)