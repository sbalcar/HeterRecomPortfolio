#!/usr/bin/python3

from typing import List #class
from typing import Dict #class

from pandas.core.frame import DataFrame #class
from abc import ABC, abstractmethod

class AEvalTool(ABC):

    CLICKS = "clicks"

    @abstractmethod
    def __init__(self, args:dict):
       raise Exception("AAgregation is abstract class, can't be instanced")

    @abstractmethod
    def click(self, userID:int, rItemIDsWithResponsibility:List, clickedItemID:int, portfolioModel:DataFrame, argumentsDict:Dict[str,object]):
        assert False, "this needs to be overridden"

    @abstractmethod
    def displayed(self, userID:int, rItemIDsWithResponsibility:List, portfolioModel:DataFrame, argumentsDict:Dict[str,object]):
        assert False, "this needs to be overridden"