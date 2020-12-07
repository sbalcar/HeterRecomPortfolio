#!/usr/bin/python3

import os
import random
import numpy as np

from typing import List #class

from datasets.ml.behaviours import Behaviours #class

from input.batchesML1m.batchFuzzyDHondtDirectOptimize import BatchFuzzyDHondtDirectOptimize #class


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

   #BatchBanditTS().generateBatches()

   #BatchDHondtThompsonSampling().generateBatches()
   #BatchDHondtThompsonSamplingINF().generateBatches()

   #BatchFuzzyDHondt().generateBatches()
   #BatchFuzzyDHondtINF().generateBatches()

   BatchFuzzyDHondtDirectOptimize().generateBatches()

   #BatchSingle().generateBatches()
   #BatchSingleINF().generateBatches()

if __name__ == "__main__":

  np.random.seed(42)
  random.seed(42)

  os.chdir("..")
  print(os.getcwd())
  generateBatches()
