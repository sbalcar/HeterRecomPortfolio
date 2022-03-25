#!/usr/bin/python3

from typing import List
from typing import Dict #class

from evaluationTool.aEvalTool import AEvalTool  # class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class

from pandas.core.frame import DataFrame  # class
from pandas.core.series import Series #class

from aggregation.tools.aggrResponsibilityDHondt import normalizationOfDHondtResponsibility #function
from portfolioModel.pModelDHondtPersonalisedStat import PModelDHondtPersonalisedStat #class

import numpy as np


class EvalToolDHondtPersonal(AEvalTool):

    ARG_LEARNING_RATE_CLICKS:str = "learningRateClicks"
    ARG_LEARNING_RATE_VIEWS:str = "learningRateViews"

    ARG_NORMALIZATION_OF_RESPONSIBILITY = "normalizationOfResponsibility"

    def __init__(self, argumentsDict:Dict[str,object]):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        self.et:AEvalTool = EvalToolDHondt(argumentsDict)

        self.normalizationOfResponsibility:bool = argumentsDict.get(self.ARG_NORMALIZATION_OF_RESPONSIBILITY, False)



    def click(self, userID:int, rItemIDsWithResponsibility:List, clickedItemID:int, portfolioModel:DataFrame, argumentsDict:Dict[str,object]):
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(rItemIDsWithResponsibility) is not Series and \
                type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type Series / list.")
        #if type(clickedItemID) is not int and type(clickedItemID) is not np.int64:
        #    raise ValueError("Argument clickedItemID isn't type int.")
        if not isinstance(portfolioModel, DataFrame):
            raise ValueError("Argument portfolioModel isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        if self.normalizationOfResponsibility:
            rItemIDsWithResponsibility = normalizationOfDHondtResponsibility(rItemIDsWithResponsibility)

        portfolioModelPer:DataFrame = portfolioModel.getModel(userID)
        if isinstance(portfolioModel, PModelDHondtPersonalisedStat):
            portfolioModel.incrementClick(userID)
        #print(portfolioModelPer)

        self.et.click(userID, rItemIDsWithResponsibility, clickedItemID, portfolioModelPer, argumentsDict)
        #print(portfolioModelPer)

        print("HOP")
        print("userID: " + str(userID))
        print("clickedItemID: " + str(clickedItemID))


    def displayed(self, userID:int, rItemIDsWithResponsibility:List, portfolioModel:DataFrame, argumentsDict:Dict[str,object]):
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(rItemIDsWithResponsibility) is not Series and \
                type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type Series / list.")
        if not isinstance(portfolioModel, DataFrame):
            raise ValueError("Argument portfolioModel isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        #print(rItemIDsWithResponsibility)
        rItemIDsWithResponsibilityNorm = normalizationOfDHondtResponsibility(rItemIDsWithResponsibility)

        portfolioModelPer:DataFrame = portfolioModel.getModel(userID)

        self.et.displayed(userID, rItemIDsWithResponsibilityNorm, portfolioModelPer, argumentsDict)
