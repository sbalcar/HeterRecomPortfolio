#!/usr/bin/python3

import random

from recommender.aRecommender import ARecommender #class

from recommendation.recommendation import Recommendation #class
from recommendation.resultOfRecommendation import ResultOfRecommendation #class


class RecommenderDummyRandom(ARecommender):

   def recommend(self, numberOfItems:int=20):
      items = list(range(numberOfItems))
      random.shuffle(items)

      recomm:Recommendation = Recommendation(items)

      # ResultOfRecommendation
      return recomm.exportAsResultOfRecommendation()
