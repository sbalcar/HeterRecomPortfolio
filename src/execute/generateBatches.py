#!/usr/bin/python3

import os
import random
import numpy as np

from typing import List

from pandas.core.frame import DataFrame #class

from datasets.behaviours import Behaviours #class

from input.batchesML1m.batchBanditTS import BatchBanditTS #class
from input.batchesML1m.batchDHondt import BatchDHondt #class
from input.batchesML1m.batchDHondtBanditsVotesRoulette import BatchDHondtBanditsVotesRoulette #class
from input.batchesML1m.batchNegDHondt import BatchNegDHondt #class

from input.batchesML1m.batchSingle import BatchSingle #class

def generateBatches():
   print("Generate Batches")
   
   BatchDHondtBanditsVotesRoulette.generateBatches()

   BatchBanditTS.generateBatches()

   BatchDHondt.generateBatches()
   BatchNegDHondt.generateBatches()

   BatchSingle.generateBatches()


if __name__ == "__main__":

  np.random.seed(42)
  random.seed(42)

  os.chdir("..")
  print(os.getcwd())
  generateBatches()
