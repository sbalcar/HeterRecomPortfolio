#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

from typing import List

import random

from recommender.aRecommender import ARecommender #class

from recommendation.recommendation import Recommendation #class
from recommendation.resultOfRecommendation import ResultOfRecommendation #class
from history.aHistory import AHistory #class


class RecommenderDummyRedirector(ARecommender):

   ARG_RESULT:str = "RESULT"

   def __init__(self, argumentsDict: dict):
       if type(argumentsDict) is not dict:
           raise ValueError("Argument argumentsDict is not type dict.")

       self._argumentsDict: dict = argumentsDict

   def train(self, ratingTrainsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
       pass

   def recommendToItem(self, itemID:int, ratingsTestDF:DataFrame, history:AHistory, numberOfItems:int=20):

       result:ResultOfRecommendation = self._argumentsDict[self.ARG_RESULT]

       # ResultOfRecommendation
       return result
