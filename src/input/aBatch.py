#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from abc import ABC, abstractmethod



class ABatch(ABC):

    @abstractmethod
    def getParameters(self):
        pass

    @abstractmethod
    def run(self, batchID:str, jobID:str):
        pass


    def generateBatches(self):

        from execute.generateBatches import BatchParameters #class

        batchIDs:List[str] = BatchParameters.getBatchParameters().keys()

        jobIDs:List[str] = list(self.getParameters().keys())

        for batchIDI in batchIDs:
            for jobIDL in jobIDs:

                batchesDir:str = ".." + os.sep + "batches" + os.sep + batchIDI
                if not os.path.exists(batchesDir):
                    os.mkdir(batchesDir)

                job:str = str(self.__class__.__name__) + jobIDL
                text:str = str(self.__class__.__name__) + "().run(" \
                           + "'" + str(batchIDI) + "', " \
                           + "'" + str(jobIDL) + "'" + ")"

                jobFile:str = batchesDir + os.sep + job + ".txt"
                self.__writeToFile(jobFile, text)



    def __writeToFile(self, fileName:str, text:str):
        f = open(fileName, "w")
        f.write(text)
        f.close()
