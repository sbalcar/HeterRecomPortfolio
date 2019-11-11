#!/usr/bin/python3

import random

from configuration.arguments import Arguments #class
from configuration.argument import Argument #class

from recommender.arecommender import ARecommender #class

from recommendation.recommendation import Recommendation #class
from recommendation.resultOfRecommendation import ResultOfRecommendation #class


class RecommenderDummyRedirector(ARecommender):

   # arguments:Argument[]
   def __init__(self, arguments):

       if type(arguments) is not Arguments:
          raise ValueError("Argument arguments is not type Arguments.")

       self._arguments = arguments;

   # topK:int
   def recommend(self, numberOfItems=20):

      # result:ResultOfRecommendation
      result = self._arguments.exportArgumentValue("RESULT")

      # ResultOfRecommendation
      return result;
