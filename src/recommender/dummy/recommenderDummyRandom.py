#!/usr/bin/python3

import random
from pandas.core.frame import DataFrame #class

from typing import List
from recommender.aRecommender import ARecommender #class

from recommendation.recommendation import Recommendation #class
from recommendation.resultOfRecommendation import ResultOfRecommendation #class


class RecommenderDummyRandom(ARecommender):

   def train(self, historyDF:DataFrame, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
      pass

   def recommendToItem(self, itemID:int, numberOfItems:int=20):
      items = list(range(numberOfItems))
      random.shuffle(items)

      recomm:Recommendation = Recommendation(items)

      # ResultOfRecommendation
      return recomm.exportAsResultOfRecommendation()
