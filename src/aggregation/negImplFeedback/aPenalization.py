#!/usr/bin/python3

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from history.aHistory import AHistory #class
from abc import ABC, abstractmethod

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class

class APenalization(ABC):

    @abstractmethod
    def __init__(self, penalFnc, argumentsDict:dict):
       raise Exception("AAgregation is abstract class, can't be instanced")

    # userDef:DataFrame<(methodID:str, votes:int)>
    @abstractmethod
    def runPenalization(self, methodsResultDict:dict, userID:int, aHistory:AHistory):
        assert False, "this needs to be overridden"
    # return list<(itemID:int, Series<(rating:int, methodID:str)>)>

    @abstractmethod
    def runOneMethodPenalization(self, userID:int, methodsResultSrs:Series, history:AHistory):
        assert False, "this needs to be overridden"
    # return Series<(rating:int, itemID:int)>