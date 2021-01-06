#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

from typing import List

from aggregation.aAggregation import AAgregation #class

from recommender.aRecommender import ARecommender #class

from history.aHistory import AHistory #class
from abc import ABC, abstractmethod


class APortfolio(ABC):

    ARG_NUMBER_OF_RECOMM_ITEMS:str = "numberOfRecomItems"
    ARG_NUMBER_OF_AGGR_ITEMS:str = "numberOfAggrItems"

    UPDT_CLICK:str = "click"
    UPDT_VIEW:str = "view"

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
    def recommend(self, userID:int, portFolioModel:DataFrame, argumentsDict:dict):
        assert False, "this needs to be overridden"
