#!/usr/bin/python3

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from history.aHistory import AHistory #class
from abc import ABC, abstractmethod


class ADHondtSelector(ABC):

    @abstractmethod
    def __init__(self, argumentsDict:dict):
       raise Exception("AAgregation is abstract class, can't be instanced")


    @abstractmethod
    def select(self, resultOfMethod:Series):
        assert False, "this needs to be overridden"
    # return itemID:int

