#!/usr/bin/python3

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class


from history.aHistory import AHistory #class
from abc import ABC, abstractmethod

class ARecommender(ABC):

    @abstractmethod
    def __init__(self, argumentsDict:dict):
        raise Exception("ARecommender is abstract class, can't be instanced")

    @abstractmethod
    def train(self, ratingsTrainDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
        assert False, "this needs to be overridden"

    @abstractmethod
    def update(self, ratingsUpdateDF:DataFrame):
        assert False, "this needs to be overridden"

    @abstractmethod
    def recommend(self, userID:int, numberOfItems:int=20):
        assert False, "this needs to be overridden"

