#!/usr/bin/python3

from typing import List
from typing import Tuple

from evaluationTool.aEvalTool import AEvalTool #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

import numpy as np

class EToolBanditTSHit1(AEvalTool):

    @staticmethod
    def evaluate(aggregatedItemIDsWithResponsibility:List[Tuple[int, str]], nextItem:int, pModelDF:DataFrame, evaluationDict:dict):
        if type(aggregatedItemIDsWithResponsibility) is not list:
            raise ValueError("Argument aggregatedItemIDsWithResponsibility isn't type list.")

        if type(nextItem) is not int and type(nextItem) is not np.int64:
            raise ValueError("Argument nextItem isn't type int.")

        if type(pModelDF) is not DataFrame:
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(pModelDF.columns) != ['r', 'n', 'alpha0', 'beta0']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")

        if type(evaluationDict) is not dict:
            raise ValueError("Argument evaluationDict isn't type dict.")

        pModelDF['n'] += 1

        for itemI, methodI in aggregatedItemIDsWithResponsibility:
            if itemI == nextItem:
                print("HOP")
                print("nextItem: " + str(nextItem))

                evaluationDict[AEvalTool.CTR] = evaluationDict.get(AEvalTool.CTR, 0) + 1

                rowI:Series = pModelDF.loc[methodI]
                rowI['r'] += 1
