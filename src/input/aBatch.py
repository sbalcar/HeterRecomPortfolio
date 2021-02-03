#!/usr/bin/python3

import os

from typing import List

from abc import ABC, abstractmethod

from input.inputABatchDefinition import InputABatchDefinition


class ABatch(ABC):

    @staticmethod
    def getParameters():
        pass

    @abstractmethod
    def run(self, batchID:str, jobID:str):
        pass

    @classmethod
    def generateAllBatches(cls):

        batchIDs:List[str] = InputABatchDefinition.getBatchParameters(cls.datasetID).keys()

        jobIDs:List[str] = list(cls.getParameters().keys())

        for batchIDI in batchIDs:
            for jobIDL in jobIDs:
                cls.__writeToFile(batchIDI, jobIDL)


    @classmethod
    def generateSelectedBatches(cls, selectedJobIDs:List[str]):

        batchIDs:List[str] = InputABatchDefinition.getBatchParameters(cls.datasetID).keys()

        jobIDs:List[str] = list(cls.getParameters().keys())

        for batchIDI in batchIDs:
            for jobIDL in jobIDs:
                print(jobIDL)
                if not jobIDL in selectedJobIDs:
                    continue
                cls.__writeToFile(batchIDI, jobIDL)


    @classmethod
    def __writeToFile(cls, batchIDI:str, jobIDL:str):

        batchesDir: str = ".." + os.sep + "batches" + os.sep + batchIDI
        if not os.path.exists(batchesDir):
            os.mkdir(batchesDir)

        job: str = str(cls.__name__) + jobIDL
        text: str = str(cls.__name__) + "().run(" \
                    + "'" + str(batchIDI) + "', " \
                    + "'" + str(jobIDL) + "'" + ")"

        jobFile:str = batchesDir + os.sep + job + ".txt"

        f = open(jobFile, "w")
        f.write(text)
        f.close()
