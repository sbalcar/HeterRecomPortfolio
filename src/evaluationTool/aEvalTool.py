#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class
from abc import ABC, abstractmethod

class AEvalTool(ABC):

    CLICKS = "clicks"

    @abstractmethod
    def __init__(self):
       raise Exception("AAgregation is abstract class, can't be instanced")

    @staticmethod
    def evaluate(rItemIDs:List[int], aggregatedItemIDsWithResponsibility:List, nextItem:int, methodsParamsDF:DataFrame, evaluationDict:dict):
        assert False, "this needs to be overridden"
