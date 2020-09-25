#!/usr/bin/python3

from typing import List

from evaluationTool.aEvalTool import AEvalTool  # class

from pandas.core.frame import DataFrame  # class
from pandas.core.series import Series #class

import numpy as np


class EToolSingleMethod(AEvalTool):

    def __init__(self, argsDict:dict):
        if type(argsDict) is not dict:
            raise ValueError("Argument argsDict isn't type dict.")

    def click(self, rItemIDsWithResponsibility:List, clickedItemID:int, portfolioModel:DataFrame, evaluationDict:dict):
        if type(rItemIDsWithResponsibility) is not Series:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type Series.")
        if type(clickedItemID) is not int and type(clickedItemID) is not np.int64:
            raise ValueError("Argument clickedItemID isn't type int.")
        if type(portfolioModel) is not DataFrame:
            raise ValueError("Argument portfolioModel isn't type DataFrame.")
        if type(evaluationDict) is not dict:
            raise ValueError("Argument evaluationDict isn't type dict.")


        print("HOP")
        print("clickedItemID: " + str(clickedItemID))

        evaluationDict[AEvalTool.CLICKS] = evaluationDict.get(AEvalTool.CLICKS, 0) + 1

    def displayed(self, rItemIDsWithResponsibility:List, portfolioModel:DataFrame, evaluationDict:dict):
        if type(rItemIDsWithResponsibility) is not Series:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type Series.")
        if type(portfolioModel) is not DataFrame:
            raise ValueError("Argument portfolioModel isn't type DataFrame.")
        if type(evaluationDict) is not dict:
            raise ValueError("Argument evaluationDict isn't type dict.")

        pass