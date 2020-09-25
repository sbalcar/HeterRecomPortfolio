#!/usr/bin/python3

import os
import random
import numpy as np

from typing import List

from pandas.core.frame import DataFrame #class

from datasets.behaviours import Behaviours #class


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

   data:List[tuple] = [
      ("singleML1mCBmax.job", "jobSingleML1mCBmax"),
      ("singleML1mCBwindow3.job", "jobSingleML1mCBwindow3"),
      ("singleML1mTheMostPopular.job", "jobSingleML1mTheMostPopular"),
      ("singleW2vPosnegMean.job", "jobSingleW2vPosnegMean"),
      ("singleW2vPosnegWindow3.job", "jobSingleW2vPosnegWindow3"),

      ("banditTS.job", "jobBanditTS"),
      ("dHontFixedClk01View00002.job", "jobDHontFixedClk01View00002"),
      ("dHontRoulette1Clk01View00002.job", "jobDHontRoulette1Clk01View00002"),
      ("dHontRoulette3Clk01View00002.job", "jobDHontRoulette3Clk01View00002"),

      ("negDHontFixedClk01View00002OLin0802HLin1002.job", "jobNegDHontFixedClk01View00002OLin0802HLin1002"),
      ("negDHontFixedClk01View00002OStat08HLin1002.job", "jobNegDHontFixedClk01View00002OStat08HLin1002"),
      ("negDHontRoulette1Clk01View00002OLin0802HLin1002.job", "jobNegDHontRoulette1Clk01View00002OLin0802HLin1002"),
      ("negDHontRoulette1Clk01View00002OStat08HLin1002.job", "jobNegDHontRoulette1Clk01View00002OStat08HLin1002"),
      ("negDHontRoulette3Clk01View00002OLin0802HLin1002.job", "jobNegDHontRoulette3Clk01View00002OLin0802HLin1002"),
      ("negDHontRoulette3Clk01View00002OStat08HLin1002.job", "jobNegDHontRoulette3Clk01View00002OStat08HLin1002")
   ]

   batchID:str = "ml1mDiv" + str(divisionDatasetPercentualSize) + "U" + uBehaviour + "R" + str(repetition)
   batchDir:str = ".." + os.sep + "batches" + os.sep + batchID
   os.mkdir(batchDir)

   for jobI, genFunctionI in data:
      __writeToFile(batchDir + os.sep + jobI, genFunctionI + "('" + batchID + "', " + str(
         divisionDatasetPercentualSize) + ", '" + uBehaviour + "', " + str(repetition) + ")")


def __writeToFile(fileName:str, text:str):
   f = open(fileName, "w")
   f.write(text)
   f.close()



if __name__ == "__main__":

  np.random.seed(42)
  random.seed(42)

  os.chdir("..")
  generateBatches()