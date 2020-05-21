#!/usr/bin/python3

from typing import List
from typing import Tuple

from evaluationTool.aEvalTool import AEvalTool #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

import numpy as np

class EToolBanditTSHit1(AEvalTool):

    @staticmethod
    def click(rItemIDsWithResponsibility:List, clickedItemID:int, probability:float, portfolioModel:DataFrame, evaluationDict:dict):
        if type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type list.")
        if type(clickedItemID) is not int and type(clickedItemID) is not np.int64:
            raise ValueError("Argument clickedItemID isn't type int.")
        if type(probability) is not float:
            raise ValueError("Argument probability isn't type float.")
        if type(portfolioModel) is not DataFrame:
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(portfolioModel.columns) != ['r', 'n', 'alpha0', 'beta0']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")
        if type(evaluationDict) is not dict:
            raise ValueError("Argument evaluationDict isn't type dict.")

        #EToolBanditTSHit1.ignore(rItemIDsWithResponsibility, portfolioModel, evaluationDict)

        for itemI, methodI in rItemIDsWithResponsibility:
            if itemI == clickedItemID:
                print("HOP")
                print("clickedItemID: " + str(clickedItemID))

                evaluationDict[AEvalTool.CLICKS] = evaluationDict.get(AEvalTool.CLICKS, 0) + 1

                rowI:Series = portfolioModel.loc[methodI]
                rowI['r'] += 1
                print(portfolioModel)


    @staticmethod
    def displayed(rItemIDsWithResponsibility:List, portfolioModel:DataFrame, evaluationDict:dict):
        if type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type list.")
        if type(portfolioModel) is not DataFrame:
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(portfolioModel.columns) != ['r', 'n', 'alpha0', 'beta0']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")
        if type(evaluationDict) is not dict:
            raise ValueError("Argument evaluationDict isn't type dict.")

        # increment by the number of objects
        for itemIdI,methodIdI in rItemIDsWithResponsibility:
            portfolioModel.loc[methodIdI]['n'] += 1