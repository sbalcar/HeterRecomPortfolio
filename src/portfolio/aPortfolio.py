#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

from typing import List

from aggregation.aAggregation import AAgregation #class

from recommender.aRecommender import ARecommender #class

from history.aHistory import AHistory #class
from abc import ABC, abstractmethod


class APortfolio(ABC):

    @abstractmethod
    def __init__(self, recommIDs: List[str], recommenders: List[ARecommender], agregation:AAgregation):
        raise Exception("APortfolio is abstract class, can't be instanced")

    @abstractmethod
    def getRecommIDs(self):
        assert False, "this needs to be overridden"

    @abstractmethod
    def train(self, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
        assert False, "this needs to be overridden"

    @abstractmethod
    def update(self, ratingsUpdateDF:DataFrame):
        assert False, "this needs to be overridden"

    @abstractmethod
    # portFolioModel:DataFrame<(methodID, votes)>
    def recommendToItem(self, portFolioModel:DataFrame, itemID:int, testRatingsDF:DataFrame, history:AHistory, numberOfItems:int):
        assert False, "this needs to be overridden"
