#!/usr/bin/python3

import time
import sys
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

from configuration.configuration import Configuration #class

import random
import numpy as np

from typing import List
from batch.batch import Batch #class

from execute.generateBatches import getBatchesJournal #function

def verificationJournal():
    print("Verification Journal")

    batchesDef:List = getBatchesJournal()

    batches:List[Batch] = []
    for batcheDefI in batchesDef:
        batches.extend(batcheDefI.getAllBatches())

    print("Batches: " + str(len(batches)))
    for batchI in batches:
        #print(batchI)
        if (not batchI.exists()):
            print(batchI.batchID)
            print(batchI.jobID)
            print("KO")


def resultsVerification():
    verificationJournal()


if __name__ == "__main__":

  np.random.seed(42)
  random.seed(42)

  os.chdir("..")

  resultsVerification()