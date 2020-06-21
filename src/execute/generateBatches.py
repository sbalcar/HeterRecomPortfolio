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
#from input.batchML1m.jobNegDHontOLin0802HLin1002 import jobNegDHontOLin0802HLin1002 #function
#from input.batchML1m.jobNegDHontOStat08HLin1002 import jobNegDHontOStat08HLin1002 #function

#from input.batchML1m.jobSingleCBmax import jobSingleML1mCBmax #class
#from input.batchML1m.jobSingleCBwindow10 import jobSingleML1mCBwindow10 #class
#from input.batchML1m.jobSingleTheMostPopular import jobSingleML1mTheMostPopular #class
#from input.batchML1m.jobSingleW2vPosnegMean import jobSingleW2vPosnegMean #class
#from input.batchML1m.jobSingleW2vPosnegWindow3 import jobSingleW2vPosnegWindow3 #class


def generateBatches():
   print("Generate Batches")

   __generateBatch(90, 1)
   __generateBatch(90, 3)
   __generateBatch(90, 5)
   __generateBatch(90, 8)



def __generateBatch(divisionDatasetPercentualSize:int, repetition:int):

   batchDir:str = ".." + os.sep + "batches" + os.sep + "ml1mDiv" + str(divisionDatasetPercentualSize) + "R" + str(repetition)
   os.mkdir(batchDir)

   __writeToFile(batchDir + os.sep + "banditTS.job", "jobBanditTS(" + str(divisionDatasetPercentualSize) + ", " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "dHontFixed.job", "jobDHontFixed(" + str(divisionDatasetPercentualSize) + ", " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "dHontRoulette.job", "jobDHontRoulette(" + str(divisionDatasetPercentualSize) + ", " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "dHontRoulette3.job", "jobDHontRoulette3(" + str(divisionDatasetPercentualSize) + ", " + str(repetition) + ")")

   __writeToFile(batchDir + os.sep + "negDHontOLin0802HLin1002.job", "jobNegDHontOLin0802HLin1002(" + str(divisionDatasetPercentualSize) + ", " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "negDHontOStat08HLin1002.job", "jobNegDHontOStat08HLin1002(" + str(divisionDatasetPercentualSize) + ", " + str(repetition) + ")")

   __writeToFile(batchDir + os.sep + "singleML1mCBmax.job", "jobSingleML1mCBmax(" + str(divisionDatasetPercentualSize) + ", " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "singleML1mCBwindow10.job", "jobSingleML1mCBwindow10(" + str(divisionDatasetPercentualSize) + ", " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "singleML1mTheMostPopular.job", "jobSingleML1mTheMostPopular(" + str(divisionDatasetPercentualSize) + ", " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "singleW2vPosnegMean.job", "jobSingleW2vPosnegMean(" + str(divisionDatasetPercentualSize) + ", " + str(repetition) + ")")
   __writeToFile(batchDir + os.sep + "singleW2vPosnegWindow3.job", "jobSingleW2vPosnegWindow3(" + str(divisionDatasetPercentualSize) + ", " + str(repetition) + ")")


def __writeToFile(fileName:str, text:str):
   f = open(fileName, "w")
   f.write(text)
   f.close()
