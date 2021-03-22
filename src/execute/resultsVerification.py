#!/usr/bin/python3

import time
import sys
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import random
import numpy as np

from typing import List
import batch.batch

from execute.generateBatches import getBatchesJournal #function



def resultsVerification():
    verificationJournal()

def verificationJournal():
    print("Verification Journal")

    batchesDef:List = getBatchesJournal()

    batches:List[batch.batch.Batch] = []
    for batcheDefI in batchesDef:
        batches.extend(batcheDefI.getAllBatches())

    print("Batches: " + str(len(batches)))
    for batchI in batches:
        #print(batchI)
        if (not os.path.isfile(batchI.getFileName())):
            pass
            print("KO")
            #print(batchI.batchID)
            #print(batchI.jobID)
        else:
            pass
            #print("OK")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    os.chdir("..")

    resultsVerification()