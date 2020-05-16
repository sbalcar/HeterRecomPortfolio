#!/usr/bin/python3

from typing import List

from evaluationTool.aEvalTool import AEvalTool #class

from pandas.core.frame import DataFrame #class

import numpy as np


class EToolDHontHit1(AEvalTool):

    @staticmethod
    def click(rItemIDsWithResponsibility:List, clickedItemID:int, probability:float, portfolioModel:DataFrame,
              evaluationDict:dict):
        if type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type list.")
        if type(clickedItemID) is not int and type(clickedItemID) is not np.int64:
            raise ValueError("Argument clickedItemID isn't type int.")
        if type(probability) is not float:
            raise ValueError("Argument probability isn't type float.")
        if type(portfolioModel) is not DataFrame:
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(portfolioModel.columns) != ['votes']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")
        if type(evaluationDict) is not dict:
            raise ValueError("Argument evaluationDict isn't type dict.")


        aggrItemIDsWithRespDF:DataFrame = DataFrame(rItemIDsWithResponsibility, columns=["itemId", "responsibility"])
        aggrItemIDsWithRespDF.set_index("itemId", inplace=True)

        print("HOP")
        print("clickedItemID: " + str(clickedItemID))

        evaluationDict[AEvalTool.CLICKS] = evaluationDict.get(AEvalTool.CLICKS, 0) + 1

        #responsibilityDict:dict[methodID:str, votes:int]
        responsibilityDict:dict[str,int] = aggrItemIDsWithRespDF.loc[clickedItemID]["responsibility"]

        # increment portfolio model
        for methodIdI in responsibilityDict.keys():
            portfolioModel.loc[methodIdI] += responsibilityDict[methodIdI]
        print(portfolioModel)


    @staticmethod
    def ignore(rItemIDsWithResponsibility:List, portfolioModel:DataFrame, evaluationDict:dict):
        if type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type list.")
        if type(portfolioModel) is not DataFrame:
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(portfolioModel.columns) != ['votes']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")
        if type(evaluationDict) is not dict:
            raise ValueError("Argument evaluationDict isn't type dict.")

        pass