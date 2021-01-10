#!/usr/bin/python3

from typing import Dict #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from history.aHistory import AHistory #class
from abc import ABC, abstractmethod

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class

class AAgregation(ABC):

    ARG_CONTEXT_TYPE:str = "contextType"

    @abstractmethod
    def __init__(self, aHistory:AHistory, argumentsDict:Dict[str,object]):
       raise Exception("AAgregation is abstract class, can't be instanced")

    @abstractmethod
    def update(self, ratingsUpdateDF:DataFrame, argumentsDict:Dict[str,object]):
        assert False, "this needs to be overridden"

    @abstractmethod
    def run(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems:int, argumentsDict:Dict[str,object]):
        assert False, "this needs to be overridden"

    # userDef:DataFrame<(methodID:str, votes:int)>
    @abstractmethod
    def runWithResponsibility(self, methodsResultDict, userDef:DataFrame, userID:int, numberOfItems:int, argumentsDict:Dict[str,object]):
        assert False, "this needs to be overridden"
    # return list<(itemID:int, Series<(rating:int, methodID:str)>)>

