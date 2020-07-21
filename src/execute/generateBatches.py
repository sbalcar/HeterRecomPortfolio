#!/usr/bin/python3

import os
import random
import numpy as np

from typing import List

from pandas.core.frame import DataFrame #class

from datasets.behaviours import Behaviours #class

#from input.batchML1m.jobBanditTS import jobBanditTS #function
#from input.batchML1m.jobDHontFixed import jobDHontFixed #function
#from input.batchML1m.jobDHontRoulette import jobDHontRoulette #function
#from input.batchML1m.jobDHontRoulette3 import jobDHontRoulette3 #function
#from input.batchML1m.jobNegDHontFixedOLin0802HLin1002 import jobNegDHontFixedOLin0802HLin1002 #function
#from input.batchML1m.jobNegDHontFixedOStat08HLin1002 import jobNegDHontFixedOStat08HLin1002 #function

#from input.batchML1m.jobSingleCBmax import jobSingleML1mCBmax #class
#from input.batchML1m.jobSingleCBwindow10 import jobSingleML1mCBwindow10 #class
#from input.batchML1m.jobSingleTheMostPopular import jobSingleML1mTheMostPopular #class
#from input.batchML1m.jobSingleW2vPosnegMean import jobSingleW2vPosnegMean #class
#from input.batchML1m.jobSingleW2vPosnegWindow3 import jobSingleW2vPosnegWindow3 #class


def generateBatches():
   print("Generate Batches")

   uBehaviours:List[str] = [Behaviours.COL_LINEAR0109, Behaviours.COL_STATIC08]

   repetitions:List[int] = [1, 2, 3, 5, 8]

   uBehaviourI:str
   repetitionI:int
   for uBehaviourI in uBehaviours:
      for repetitionI in repetitions:
         __generateBatch(90, uBehaviourI, repetitionI)



def __generateBatch(divisionDatasetPercentualSize:int, uBehaviour:str, repetition:int):

   batchID:str = "ml1mDiv" + str(divisionDatasetPercentualSize) + "U" + uBehaviour + "R" + str(repetition)
   batchDir:str = ".." + os.sep + "batches" + os.sep + batchID
   os.mkdir(batchDir)

   __writeToFile(batchDir + os.sep + "banditTS.job", "jobBanditTS('" + batchID + "', " + str(divisionDatasetPercentualSize) + ", '" + uBehaviour + "', " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "dHontFixed.job", "jobDHontFixed('" + batchID + "', " + str(divisionDatasetPercentualSize) + ", '" + uBehaviour + "', " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "dHontRoulette.job", "jobDHontRoulette('" + batchID + "', " + str(divisionDatasetPercentualSize) + ", '" + uBehaviour + "', " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "dHontRoulette3.job", "jobDHontRoulette3('" + batchID + "', " + str(divisionDatasetPercentualSize) + ", '" + uBehaviour + "', " + str(repetition) + ")")

   __writeToFile(batchDir + os.sep + "negDHontFixedOLin0802HLin1002.job", "jobNegDHontFixedOLin0802HLin1002('" + batchID + "', " + str(divisionDatasetPercentualSize) + ", '" + uBehaviour + "', " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "negDHontFixedOStat08HLin1002.job", "jobNegDHontFixedOStat08HLin1002('" + batchID + "', " + str(divisionDatasetPercentualSize) + ", '" + uBehaviour + "', " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "negDHontRouletteOLin0802HLin1002.job", "jobNegDHontRouletteOLin0802HLin1002('" + batchID + "', " + str(divisionDatasetPercentualSize) + ", '" + uBehaviour + "', " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "negDHontRouletteOStat08HLin1002.job", "jobNegDHontRouletteOStat08HLin1002('" + batchID + "', " + str(divisionDatasetPercentualSize) + ", '" + uBehaviour + "', " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "negDHontRoulette3OLin0802HLin1002.job", "jobNegDHontRoulette3OLin0802HLin1002('" + batchID + "', " + str(divisionDatasetPercentualSize) + ", '" + uBehaviour + "', " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "negDHontRoulette3OStat08HLin1002.job", "jobNegDHontRoulette3OStat08HLin1002('" + batchID + "', " + str(divisionDatasetPercentualSize) + ", '" + uBehaviour + "', " + str(repetition) + ")")


   __writeToFile(batchDir + os.sep + "singleML1mCBmax.job", "jobSingleML1mCBmax('" + batchID + "', " + str(divisionDatasetPercentualSize) + ", '" + uBehaviour + "', " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "singleML1mCBwindow10.job", "jobSingleML1mCBwindow10('" + batchID + "', " + str(divisionDatasetPercentualSize) + ", '" + uBehaviour + "', " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "singleML1mTheMostPopular.job", "jobSingleML1mTheMostPopular('" + batchID + "', " + str(divisionDatasetPercentualSize) + ", '" + uBehaviour + "', " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "singleW2vPosnegMean.job", "jobSingleW2vPosnegMean('" + batchID + "', " + str(divisionDatasetPercentualSize) + ", '" + uBehaviour + "', " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "singleW2vPosnegWindow3.job", "jobSingleW2vPosnegWindow3('" + batchID + "', " + str(divisionDatasetPercentualSize) + ", '" + uBehaviour + "', " + str(repetition) + ")")


def __writeToFile(fileName:str, text:str):
   f = open(fileName, "w")
   f.write(text)
   f.close()



if __name__ == "__main__":

  np.random.seed(42)
  random.seed(42)

  os.chdir("..")
  generateBatches()