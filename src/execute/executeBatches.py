#!/usr/bin/python3

import time
import sys
import os

import random
import numpy as np

from typing import List

from input.batchML1m.jobSingleTheMostPopular import jobSingleML1mTheMostPopular #function
from input.batchML1m.jobSingleCBmax import jobSingleML1mCBmax #function
from input.batchML1m.jobSingleCBwindow3 import jobSingleML1mCBwindow3 #function
from input.batchML1m.jobSingleW2vPosnegMean import jobSingleW2vPosnegMean #function
from input.batchML1m.jobSingleW2vPosnegWindow3 import jobSingleW2vPosnegWindow3 #function

from input.batchML1m.jobBanditTS import jobBanditTS #function
from input.batchML1m.jobDHontFixedClk01View00002 import jobDHontFixedClk01View00002 #function
from input.batchML1m.jobDHontRoulette1Clk01View00002 import jobDHontRoulette1Clk01View00002 #function
from input.batchML1m.jobDHontRoulette3Clk01View00002 import jobDHontRoulette3Clk01View00002 #function

from input.batchML1m.jobNegDHontFixedClk01View00002OStat08HLin1002 import jobNegDHontFixedClk01View00002OStat08HLin1002 #function
from input.batchML1m.jobNegDHontFixedClk01View00002OLin0802HLin1002 import jobNegDHontFixedClk01View00002OLin0802HLin1002 #function
from input.batchML1m.jobNegDHontRoulette1Clk01View00002OStat08HLin1002 import jobNegDHontRoulette1Clk01View00002OStat08HLin1002 #function
from input.batchML1m.jobNegDHontRoulette1Clk01View00002OLin0802HLin1002 import jobNegDHontRoulette1Clk01View00002OLin0802HLin1002 #function
from input.batchML1m.jobNegDHontRoulette3Clk01View00002OStat08HLin1002 import jobNegDHontRoulette3Clk01View00002OStat08HLin1002 #function
from input.batchML1m.jobNegDHontRoulette3Clk01View00002OLin0802HLin1002 import jobNegDHontRoulette3Clk01View00002OLin0802HLin1002 #function


def executeBatches():

  np.random.seed(42)
  random.seed(42)


  batchesDir:str = ".." + os.sep + "inputs"

  batches:List[str] = [batchesDir + os.sep + f.name for f in os.scandir(batchesDir) if f.is_dir() if f.name != "__pycache__"]
  for batchIdI in batches:
      print(batchIdI)

      jobs:List[str] = [batchIdI + os.sep + f.name for f in os.scandir(batchesDir + os.sep + batchIdI)
                        if f.is_file() and not f.name.startswith(".")]
      for jobI in jobs:
          print(jobI)

          file = open(jobI, "r")
          command:str = file.read()
          os.remove(jobI)
          print(command)

          exec(command)
