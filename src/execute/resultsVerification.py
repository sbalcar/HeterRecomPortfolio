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
        if not os.path.isfile(batchI.getFileName()):
            pass
            print("KO  " + batchI.getFileName())
        else:
            pass
            #print("OK")
            results = readTheLastComputationResult(batchI.getFileName())
            if int(results[0]) != 54400 and int(results[0]) != 9100:
                print(results[0] + "  " + batchI.batchID + "/" + results[1] + "  " + results[2])


def readTheLastComputationResult(fileName:str):
    with open(fileName, 'r') as f:
        lines = f.readlines()
        iterationLine = lines[-3]
        methods = lines[-2]
        lastLine = lines[-1]

    iterationLine = iterationLine.replace('RatingI: ', '')
    iterationLine = iterationLine[:iterationLine.index(' /')]

    lastLine = lastLine.replace("Evaluations: [{'clicks': ", "")
    lastLine = lastLine.replace("}]\n", "")

    methods = methods.replace("PortfolioIds: ['", "").replace("']\n", "")

    return (iterationLine, methods, lastLine)


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    os.chdir("..")

    resultsVerification()