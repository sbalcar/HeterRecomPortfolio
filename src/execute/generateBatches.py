#!/usr/bin/python3

import os
import random
import numpy as np

from typing import List #class

from pandas.core.frame import DataFrame #class

from datasets.behaviours import Behaviours #class

from input.batchesML1m.batchBanditTS import BatchBanditTS #class
from input.batchesML1m.batchDHondt import BatchDHondt #class
from input.batchesML1m.batchDHondtThompsonSampling import BatchDHondtThompsonSampling #class
from input.batchesML1m.batchNegDHondtThompsonSampling import BatchNegDHondtThompsonSampling #class

from input.batchesML1m.batchNegDHondt import BatchNegDHondt #class

from input.batchesML1m.batchSingle import BatchSingle #class

class BatchParameters:

    @staticmethod
    def getBatchParameters():

        divisionsDatasetPercentualSize:List[int] = [90]
        uBehaviours:List[str] = [Behaviours.BHVR_LINEAR0109, Behaviours.BHVR_STATIC08,
                                  Behaviours.BHVR_STATIC06, Behaviours.BHVR_STATIC04,
                                  Behaviours.BHVR_STATIC02]
        repetitions:List[int] = [1, 2, 3, 5]

        aDict:dict = {}

        for divisionDatasetPercentualSizeI in divisionsDatasetPercentualSize:
            for uBehaviourJ in uBehaviours:
                for repetitionK in repetitions:
                    batchID:str = "ml1mDiv" + str(divisionDatasetPercentualSizeI) + "U" + uBehaviourJ + "R" + str(repetitionK)

                    aDict[batchID] = (divisionDatasetPercentualSizeI, uBehaviourJ, repetitionK)

        return aDict


def generateBatches():
   print("Generate Batches")

   BatchSingle().generateBatches()

   BatchBanditTS().generateBatches()

   BatchDHondt().generateBatches()
   BatchDHondtThompsonSampling().generateBatches()

   BatchNegDHondt().generateBatches()
   BatchNegDHondtThompsonSampling().generateBatches()


if __name__ == "__main__":

  np.random.seed(42)
  random.seed(42)

  os.chdir("..")
  print(os.getcwd())
  generateBatches()
