#!/usr/bin/python3

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from history.aHistory import AHistory #class
from abc import ABC, abstractmethod

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class

class AAgregation(ABC):

    @abstractmethod
    def __init__(self, aHistory:AHistory, argumentsDict:dict):
       raise Exception("AAgregation is abstract class, can't be instanced")

    # userDef:DataFrame<(methodID:str, votes:int)>
    @abstractmethod
    def runWithResponsibility(self, methodsResultDict, userDef:DataFrame, numberOfItems:float=20):
        assert False, "this needs to be overridden"
    # return list<(itemID:int, Series<(rating:int, methodID:str)>)>

