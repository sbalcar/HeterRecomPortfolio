#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class


class AEvalTool:

    CLICKS = "clicks"

    @staticmethod
    def evaluate(rItemIDs:List[int], aggregatedItemIDsWithResponsibility:List, nextItem:int, methodsParamsDF:DataFrame, evaluationDict:dict):
        assert False, "this needs to be overridden"
