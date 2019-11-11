#!/usr/bin/python3

import random

from recommender.arecommender import ARecommender #class

from recommendation.recommendation import Recommendation #class
from recommendation.resultOfRecommendation import ResultOfRecommendation #class


class RecommenderDummyRandom(ARecommender):

   # topK:int
   def recommend(self, numberOfItems=20):
      items = list(range(numberOfItems))
      random.shuffle(items)

      # recomm:Recommendation
      recomm = Recommendation(items)

      # ResultOfRecommendation
      return recomm.exportAsResultOfRecommendation()
