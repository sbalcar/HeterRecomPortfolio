#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

from typing import List

from aggregation.aAggregation import AAgregation #class

from recommender.aRecommender import ARecommender #class

from history.aHistory import AHistory #class

class APortfolio:

    def __init__(self, recommIDs: List[str], recommenders: List[ARecommender], agregation: AAgregation):
        raise Exception("APortfolio is abstract class, can't be instanced")

    def getRecommIDs(self):
        pass

    def train(self, ratingsDF: DataFrame, usersDF: DataFrame, itemsDF: DataFrame):
        pass

    def update(self, ratingsUpdateDF: DataFrame):
        pass

    # portFolioModel:DataFrame<(methodID, votes)>
    def recommendToItem(self, portFolioModel:DataFrame, itemID:int, testRatingsDF:DataFrame, history:AHistory, numberOfItems:int):
        pass