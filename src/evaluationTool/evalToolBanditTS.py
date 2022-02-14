#!/usr/bin/python3

from typing import List
from typing import Tuple
from typing import Dict #class

from evaluationTool.aEvalTool import AEvalTool #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

import numpy as np

class EvalToolBanditTS(AEvalTool):

    def __init__(self, argumentsDict:Dict[str,object]):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")


    def click(self, userID:int, rItemIDsWithResponsibility:List, clickedItemID:int, portfolioModel:DataFrame, argumentsDict:Dict[str,object]):
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type list.")
        if type(clickedItemID) is not int and type(clickedItemID) is not np.int64:
            raise ValueError("Argument clickedItemID isn't type int.")
        if not isinstance(portfolioModel, DataFrame):
            raise ValueError("Argument portfolioModel isn't type DataFrame.")
        if list(portfolioModel.columns) != ['r', 'n', 'alpha0', 'beta0']:
            raise ValueError("Argument pModelDF doesn't contain rights columns.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        for itemI, methodI in rItemIDsWithResponsibility:
            if itemI == clickedItemID:
                #evaluationDict[AEvalTool.CLICKS] = evaluationDict.get(AEvalTool.CLICKS, 0) + 1

                rowI:Series = portfolioModel.loc[methodI]
                rowI['r'] += 1

        print("HOP")
        print("clickedItemID: " + str(clickedItemID))
        print(portfolioModel)


    def displayed(self, userID:int, rItemIDsWithResponsibility:List, portfolioModel:DataFrame, argumentsDict:Dict[str,object]):
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type list.")
        if not isinstance(portfolioModel, DataFrame):
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(portfolioModel.columns) != ['r', 'n', 'alpha0', 'beta0']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        print(rItemIDsWithResponsibility)
        print(portfolioModel)

        # increment by the number of objects
        for itemIdI,methodIdI in rItemIDsWithResponsibility:
            portfolioModel.loc[methodIdI]['n'] += 1
