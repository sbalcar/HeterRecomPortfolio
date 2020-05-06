#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class


class AEvalTool:

    CTR = "ctr"

    @staticmethod
    def evaluate(aggregatedItemIDsWithResponsibility:List, nextItem:int, methodsParamsDF:DataFrame, evaluationDict:dict):
        pass