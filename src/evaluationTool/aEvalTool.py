#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class
from abc import ABC, abstractmethod

class AEvalTool(ABC):

    CLICKS = "clicks"

    @abstractmethod
    def __init__(self):
       raise Exception("AAgregation is abstract class, can't be instanced")

    @abstractmethod
    def click(rItemIDsWithResponsibility:List, clickedItemID:int, probability:float, portfolioModel:DataFrame, evaluationDict:dict):
        assert False, "this needs to be overridden"

    @abstractmethod
    def ignore(rItemIDsWithResponsibility:List, portfolioModel:DataFrame, evaluationDict:dict):
        assert False, "this needs to be overridden"