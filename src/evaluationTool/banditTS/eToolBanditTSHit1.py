#!/usr/bin/python3

from typing import List
from typing import Tuple

from evaluationTool.aEvalTool import AEvalTool #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

import numpy as np

class EToolBanditTSHit1(AEvalTool):

    @staticmethod
    def evaluate(rItemIDs:List[int], aggregatedItemIDsWithResponsibility:List[Tuple[int, str]], nextItemID:int,
                 pModelDF:DataFrame, evaluationDict:dict):
        if type(rItemIDs) is not list:
            raise ValueError("Argument rItemIDs isn't type list.")
        for itemIDI in rItemIDs:
            if type(itemIDI) is not int and type(itemIDI) is not np.int64:
                raise ValueError("Argument itemIDI don't contain int.")
        if type(aggregatedItemIDsWithResponsibility) is not list:
            raise ValueError("Argument aggregatedItemIDsWithResponsibility isn't type list.")

        if type(nextItemID) is not int and type(nextItemID) is not np.int64:
            raise ValueError("Argument nextItem isn't type int.")

        if type(pModelDF) is not DataFrame:
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(pModelDF.columns) != ['r', 'n', 'alpha0', 'beta0']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")

        if type(evaluationDict) is not dict:
            raise ValueError("Argument evaluationDict isn't type dict.")

        # increment by the number of objects
        for itemIdI,methodIdI in aggregatedItemIDsWithResponsibility:
            pModelDF.loc[methodIdI]['n'] += 1


        for itemI, methodI in aggregatedItemIDsWithResponsibility:
            if itemI == nextItemID:
                print("HOP")
                print("nextItemID: " + str(nextItemID))

                evaluationDict[AEvalTool.CLICKS] = evaluationDict.get(AEvalTool.CLICKS, 0) + 1

                rowI:Series = pModelDF.loc[methodI]
                rowI['r'] += 1
                print(pModelDF)

