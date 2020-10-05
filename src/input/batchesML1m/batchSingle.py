#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from input.InputRecomDefinition import InputRecomDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from input.batchesML1m.aML1MConfig import AML1MConf #function

from datasets.behaviours import Behaviours #class

from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

import pandas as pd


class BatchSingle:

    @staticmethod
    def getParameters():

        aDict:dict = {}
        aDict["CosCBmax"] = "CosCBmax"
        aDict["CosCBwindow3"] = "CosCBwindow3"
        aDict["TMPopular"] = "TMPopular"
        aDict["W2vPosnegMean"] = "W2vPosnegMean"
        aDict["W2vPosnegWindow3"] = "W2vPosnegWindow3"

        return aDict


    @staticmethod
    def run(batchID:str, divisionDatasetPercentualSize:int, uBehaviour:str, repetition:int, jobID:str):

        aConf:AML1MConf = AML1MConf(batchID, divisionDatasetPercentualSize, uBehaviour, repetition)

        pDescr:APortfolioDescription = None
        if jobID == "CosCBmax":
            pDescr = Portfolio1MethDescription("CosCBmax", "cosCBmax",
                    InputRecomDefinition.exportRDescCBmean(aConf.datasetID))
        elif jobID == "CosCBwindow3":
            pDescr = Portfolio1MethDescription("CosCBwindow3", "cosCBwindow3",
                    InputRecomDefinition.exportRDescCBwindow3(aConf.datasetID))
        elif jobID == "TMPopular":
            pDescr = Portfolio1MethDescription("TMPopular", "theMostPopular",
                    InputRecomDefinition.exportRDescTheMostPopular(aConf.datasetID))
        elif jobID == "TMPopular":
            pDescr = Portfolio1MethDescription("W2vPosnegMean", "w2vPosnegMean",
                    InputRecomDefinition.exportRDescW2vPosnegMean(aConf.datasetID))
        elif jobID == "W2vPosnegWindow3":
            pDescr = Portfolio1MethDescription("W2vPosnegWindow3", "w2vPosnegWindow3",
                    InputRecomDefinition.exportRDescW2vPosnegWindow3(aConf.datasetID))

        model:DataFrame = pd.DataFrame()
        eTool:List = EToolSingleMethod({})

        aConf.run(pDescr, model, eTool)



    @staticmethod
    def generateBatches():

        divisionsDatasetPercentualSize:List[int] = [90]
        uBehaviours:List[str] = [Behaviours.BHVR_LINEAR0109, Behaviours.BHVR_STATIC08,
                                 Behaviours.BHVR_STATIC06, Behaviours.BHVR_STATIC04,
                                 Behaviours.BHVR_STATIC02]
        repetitions:List[int] = [1, 2, 3, 5]

        jobIDs:List[str] = list(BatchSingle.getParameters().keys())

        for divisionDatasetPercentualSizeI in divisionsDatasetPercentualSize:
            for uBehaviourJ in uBehaviours:
                for repetitionK in repetitions:
                    for jobIDL in jobIDs:

                        batchID:str = "ml1mDiv" + str(divisionDatasetPercentualSizeI) + "U" + uBehaviourJ + "R" + str(repetitionK)
                        batchesDir:str = ".." + os.sep + "batches" + os.sep + batchID
                        if not os.path.exists(batchesDir):
                            os.mkdir(batchesDir)

                        job:str = str(BatchSingle.__name__) + jobIDL
                        text:str = str(BatchSingle.__name__) + ".run('"\
                                   + str(batchID) + "', "\
                                   + str(divisionDatasetPercentualSizeI) + ", '"\
                                   + str(uBehaviourJ) + "', "\
                                   + str(repetitionK) + ", "\
                                   + "'" + str(jobIDL) + "'" + ")"

                        jobFile:str = batchesDir + os.sep + job + ".txt"
                        BatchSingle.__writeToFile(jobFile, text)

    @staticmethod
    def __writeToFile(fileName:str, text:str):
        f = open(fileName, "w")
        f.write(text)
        f.close()



if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   print(os.getcwd())
   BatchSingle.generateBatches()