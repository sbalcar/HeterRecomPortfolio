#!/usr/bin/python3

from typing import List

import random

from configuration.arguments import Arguments #class
from configuration.argument import Argument #class

from recommender.aRecommender import ARecommender #class

from recommendation.recommendation import Recommendation #class
from recommendation.resultOfRecommendation import ResultOfRecommendation #class


class RecommenderDummyRedirector(ARecommender):

   ARG_RESULT:str = "RESULT"

   def __init__(self, arguments:List[Argument]):

       if type(arguments) is not Arguments:
          raise ValueError("Argument arguments is not type Arguments.")

       self._arguments:List[Argument] = arguments;


   def recommend(self, numberOfItems:int=20):

       result:ResultOfRecommendation = self._arguments.exportArgumentValue(self.ARG_RESULT)

       # ResultOfRecommendation
       return result;
