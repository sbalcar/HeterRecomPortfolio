#!/usr/bin/python3

import random
from pandas.core.frame import DataFrame #class

from typing import List
from recommender.aRecommender import ARecommender #class

from recommendation.recommendation import Recommendation #class
from recommendation.resultOfRecommendation import ResultOfRecommendation #class
from history.aHistory import AHistory #class

class RecommenderDummyRandom(ARecommender):

    def __init__(self, argumentsDict:dict):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        self._argumentsDict:dict = argumentsDict

   def train(self, ratingsTrainDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
        pass

   def recommendToItem(self, itemID:int, ratingsTestDF:DataFrame, history:AHistory, numberOfItems:int=20):
        items = list(range(numberOfItems))
        random.shuffle(items)

        recomm:Recommendation = Recommendation(items)

        # ResultOfRecommendation
        return recomm.exportAsResultOfRecommendation()
