#!/usr/bin/python3

import os

from typing import List

from abc import ABC, abstractmethod

from batchDefinition.inputABatchDefinition import InputABatchDefinition

#from batch.batch import Batch #class
import batch.batch

class ABatchDefinition(ABC):

    @abstractmethod
    def getBatchName(self):
        pass

    @abstractmethod
    def getParameters(self):
        pass

    @abstractmethod
    def run(self, batchID:str, jobID:str):
        pass

    def generateAllBatches(self):

        batches:List[batch.batch.Batch] = self.getAllBatches()

        for batchI in batches:
            self.__writeToFile(batchI.batchDefinitionClass, batchI.batchID, batchI.jobID)

    def getAllBatches(self):
        batchIDs:List[str] = InputABatchDefinition.getBatchParameters(self.datasetID).keys()

        jobIDs:List[str] = list(self.getParameters().keys())
        if hasattr(self, 'jobIDs'):
            jobIDs = self.jobIDs

        batches:List[batch.batch.Batch] = []

        for batchIDI in batchIDs:
            for jobIDJ in jobIDs:
                batches.append(batch.batch.Batch(self.__class__.__name__, batchIDI, jobIDJ))

        return batches


    def __writeToFile(self, batchDefinitionClass, batchIDI:str, jobIDL:str):

        batchesDir:str = ".." + os.sep + "batches" + os.sep + batchIDI
        if not os.path.exists(batchesDir):
            os.mkdir(batchesDir)

        job:str = str(batchDefinitionClass) + jobIDL
        text:str = str(batchDefinitionClass) + "().run(" \
                    + "'" + str(batchIDI) + "', " \
                    + "'" + str(jobIDL) + "'" + ")"

        jobFile:str = batchesDir + os.sep + job + ".txt"

        f = open(jobFile, "w")
        f.write(text)
        f.close()
