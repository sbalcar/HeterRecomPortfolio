#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from input.InputRecomDefinition import InputRecomDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from input.batchesML1m.aML1MConfig import AML1MConf #function

from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

import pandas as pd
from input.aBatch import ABatch #class


class BatchSingle(ABatch):

    def getParameters(self):

        aDict:dict = {}
        aDict["CosCBmax"] = "CosCBmax"
        aDict["CosCBwindow3"] = "CosCBwindow3"
        aDict["TMPopular"] = "TMPopular"
        aDict["W2vPosnegMean"] = "W2vPosnegMean"
        aDict["W2vPosnegWindow3"] = "W2vPosnegWindow3"

        return aDict


    def run(self, batchID:str, jobID:str):

        from execute.generateBatches import BatchParameters #class
        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters()[batchID]

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



if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   print(os.getcwd())
   BatchSingle.generateBatches()