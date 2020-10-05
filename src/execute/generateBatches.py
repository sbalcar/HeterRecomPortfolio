#!/usr/bin/python3

import os
import random
import numpy as np

from typing import List

from pandas.core.frame import DataFrame #class

from datasets.behaviours import Behaviours #class

from input.batchesML1m.batchBanditTS import BatchBanditTS #class
from input.batchesML1m.batchDHondtFixed import BatchDHondtFixed #class
from input.batchesML1m.batchDHondtRoulette import BatchDHondtRoulette #class
from input.batchesML1m.batchDHondtBanditsVotesRoulette import BatchDHondtBanditsVotesRoulette #class
from input.batchesML1m.batchNegDHondtRoulette import BatchNegDHondtRoulette #class
from input.batchesML1m.batchNegDHondtFixed import BatchNegDHondtFixed #class

from input.batchesML1m.batchSingle import BatchSingle #class

def generateBatches():
   print("Generate Batches")

   
   BatchDHondtBanditsVotesRoulette.generateBatches()
   BatchBanditTS.generateBatches()
   BatchDHondtFixed.generateBatches()
   BatchDHondtRoulette.generateBatches()

   BatchNegDHondtFixed.generateBatches()
   BatchNegDHondtRoulette.generateBatches()

   BatchSingle.generateBatches()


if __name__ == "__main__":

  np.random.seed(42)
  random.seed(42)

  os.chdir("..")
  print(os.getcwd())
  generateBatches()
