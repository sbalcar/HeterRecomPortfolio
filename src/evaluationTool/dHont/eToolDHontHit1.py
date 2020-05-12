#!/usr/bin/python3

from typing import List

from evaluationTool.aEvalTool import AEvalTool #class

from pandas.core.frame import DataFrame #class

import numpy as np


class EToolDHontHit1(AEvalTool):

    @staticmethod
    def evaluate(rItemIDs:List[int], aggregatedItemIDsWithResponsibility:List, nextItemID:int,
                 pModelDF:DataFrame, evaluationDict:dict):
        if type(rItemIDs) is not list:
            raise ValueError("Argument rItemIDs isn't type list.")
        for itemIDI in rItemIDs:
            if type(itemIDI) is not int:
                raise ValueError("Argument itemIDI don't contain int.")
        if type(aggregatedItemIDsWithResponsibility) is not list:
            raise ValueError("Argument aggregatedItemIDsWithResponsibility isn't type list.")

        if type(nextItemID) is not int and type(nextItemID) is not np.int64:
            raise ValueError("Argument nextItem isn't type int.")

        if type(pModelDF) is not DataFrame:
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(pModelDF.columns) != ['votes']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")

        if type(evaluationDict) is not dict:
            raise ValueError("Argument evaluationDict isn't type dict.")


        aggrItemIDsWithRespDF:DataFrame = DataFrame(aggregatedItemIDsWithResponsibility, columns=["itemId", "responsibility"])
        aggrItemIDsWithRespDF.set_index("itemId", inplace=True)


        if nextItemID in aggrItemIDsWithRespDF.index:
            print("HOP")
            print("nextItemID: " + str(nextItemID))

            evaluationDict[AEvalTool.CLICKS] = evaluationDict.get(AEvalTool.CLICKS, 0) + 1

            #responsibilityDict:dict[methodID:str, votes:int]
            responsibilityDict:dict[str,int] = aggrItemIDsWithRespDF.loc[nextItemID]["responsibility"]

            # increment portfolio model
            for methodIdI in responsibilityDict.keys():
                if responsibilityDict[methodIdI] > 0:
                    pModelDF.loc[methodIdI] += 1
            print(pModelDF)
