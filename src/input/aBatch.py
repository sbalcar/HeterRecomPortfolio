#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from abc import ABC, abstractmethod

from datasets.ml.behavioursML import BehavioursML #class

class BatchParameters:

    @abstractmethod
    def getBatchParameters(datasetID:str):

        divisionsDatasetPercentualSize:List[int] = [90]
        uBehaviours:List[str] = [BehavioursML.BHVR_LINEAR0109,
                                 BehavioursML.BHVR_STATIC08,
                                 #BehavioursML.BHVR_STATIC06,
                                 #BehavioursML.BHVR_STATIC04,
                                 #BehavioursML.BHVR_STATIC02,
                                 BehavioursML.BHVR_POWERLAW054MIN048]
        repetitions:List[int] = [1, 2, 3]

        aDict:dict = {}

        for divisionDatasetPercentualSizeI in divisionsDatasetPercentualSize:
            for uBehaviourJ in uBehaviours:
                for repetitionK in repetitions:
                    batchID:str = datasetID + "Div" + str(divisionDatasetPercentualSizeI) + "U" + uBehaviourJ + "R" + str(repetitionK)

                    aDict[batchID] = (divisionDatasetPercentualSizeI, uBehaviourJ, repetitionK)

        return aDict


class ABatch(ABC):

    @staticmethod
    def getParameters():
        pass

    @abstractmethod
    def run(self, batchID:str, jobID:str):
        pass

    @classmethod
    def generateBatches(cls):

        batchIDs:List[str] = BatchParameters.getBatchParameters(cls.datasetID).keys()

        jobIDs:List[str] = list(cls.getParameters().keys())

        for batchIDI in batchIDs:
            for jobIDL in jobIDs:

                batchesDir:str = ".." + os.sep + "batches" + os.sep + batchIDI
                if not os.path.exists(batchesDir):
                    os.mkdir(batchesDir)

                job:str = str(cls.__name__) + jobIDL
                text:str = str(cls.__name__) + "().run(" \
                           + "'" + str(batchIDI) + "', " \
                           + "'" + str(jobIDL) + "'" + ")"

                jobFile:str = batchesDir + os.sep + job + ".txt"
                cls.__writeToFile(jobFile, text)



    def __writeToFile(fileName:str, text:str):
        f = open(fileName, "w")
        f.write(text)
        f.close()
