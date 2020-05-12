#!/usr/bin/python3

from typing import List

from evaluationTool.aEvalTool import AEvalTool  # class

from pandas.core.frame import DataFrame  # class
from pandas.core.series import Series #class

import numpy as np


class EToolSingleMethod(AEvalTool):

    @staticmethod
    def evaluate(rItemIDs:List[int], rItemIDsWithResponsibility:Series, nextItemID:int, pModelDF:DataFrame, evaluationDict:dict):
        if type(rItemIDs) is not list:
            raise ValueError("Argument rItemIDs isn't type list.")
        for itemIDI in rItemIDs:
            if type(itemIDI) is not int:
                raise ValueError("Argument itemIDI don't contain int.")
        if type(rItemIDsWithResponsibility) is not Series:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type Series.")
        if type(nextItemID) is not int and type(nextItemID) is not np.int64:
            raise ValueError("Argument nextItem isn't type int.")
        if type(pModelDF) is not DataFrame:
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if type(evaluationDict) is not dict:
            raise ValueError("Argument evaluationDict isn't type dict.")


        if nextItemID in rItemIDs:
            print("HOP")
            print("nextItemID: " + str(nextItemID))

            evaluationDict[AEvalTool.CLICKS] = evaluationDict.get(AEvalTool.CLICKS, 0) + 1

