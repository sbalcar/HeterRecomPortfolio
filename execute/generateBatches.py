#!/usr/bin/python3

import os
import random
import numpy as np

from typing import List

from pandas.core.frame import DataFrame #class

from datasets.behaviours import Behaviours #class

from input.batchesML1m.batchBanditTS import BatchBanditTS #class
from input.batchesML1m.batchDHontFixed import BatchDHontFixed #class
from input.batchesML1m.batchDHontRoulette import BatchDHontRoulette #class
from input.batchesML1m.batchDHondtBanditsVotesRoulette import BatchDHondtBanditsVotesRoulette #class
from input.batchesML1m.batchNegDHontRoulette import BatchNegDHontRoulette #class
from input.batchesML1m.batchNegDHontFixed import BatchNegDHontFixed #class

from input.batchesML1m.batchSingle import BatchSingle #class

def generateBatches():
   print("Generate Batches")

   
   BatchDHondtBanditsVotesRoulette.generateBatches()
   #BatchBanditTS.generateBatches()
   #BatchDHontFixed.generateBatches()
   #BatchDHontRoulette.generateBatches()

   #BatchNegDHontFixed.generateBatches()
   #BatchNegDHontRoulette.generateBatches()

   #BatchSingle.generateBatches()


if __name__ == "__main__":

  np.random.seed(42)
  random.seed(42)

  os.chdir("..")
  print(os.getcwd())
  generateBatches()
