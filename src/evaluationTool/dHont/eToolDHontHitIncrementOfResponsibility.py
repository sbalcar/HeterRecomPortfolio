#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class

from evaluationTool.aEvalTool import AEvalTool #class

import numpy as np


class EToolDHontHitIncrementOfResponsibility(AEvalTool):

    @staticmethod
    def evaluate(aggregatedItemIDsWithResponsibility:List, nextItem:int, pModelDF:DataFrame, evaluationDict:dict):
        if type(aggregatedItemIDsWithResponsibility) is not list:
            raise ValueError("Argument aggregatedItemIDsWithResponsibility isn't type list.")

        if type(nextItem) is not int and type(nextItem) is not np.int64:
            raise ValueError("Argument nextItem isn't type int.")

        if type(pModelDF) is not DataFrame:
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(pModelDF.columns) != ['votes']:
            print(pModelDF.columns)
            raise ValueError("Argument pModelDF doen't contain rights columns.")

        if type(evaluationDict) is not dict:
            raise ValueError("Argument evaluationDict isn't type dict.")

        aggrItemIDsWithRespDF:DataFrame = DataFrame(aggregatedItemIDsWithResponsibility, columns=["itemId", "responsibility"])
        aggrItemIDsWithRespDF.set_index("itemId", inplace=True)

        if nextItem in aggrItemIDsWithRespDF.index:
            print("HOP")
            print("nextItem: " + str(nextItem))

            evaluationDict[AEvalTool.CTR] = evaluationDict.get(AEvalTool.CTR, 0) + 1
            #print(evaluationDict)

            #responsibility:dict[methodID:str, votes:int]
            responsibility:dict[str, int] = aggrItemIDsWithRespDF.loc[nextItem]["responsibility"]

            # increment user definition
            for methodIdI in responsibility.keys():
                pModelDF.loc[methodIdI] += responsibility[methodIdI]
            print(pModelDF)

