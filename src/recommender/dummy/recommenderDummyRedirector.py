#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

from typing import List

import random

from recommender.aRecommender import ARecommender #class

from history.aHistory import AHistory #class

import pandas as pd

class RecommenderDummyRedirector(ARecommender):

   ARG_RESULT:str = "RESULT"

   def __init__(self, argumentsDict: dict):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        self._argumentsDict: dict = argumentsDict

   def train(self, ratingTrainsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
        pass

   def update(self, ratingsUpdateDF:DataFrame):
        pass

   def recommend(self, userID: int, numberOfItems: int = 20):
        result:pd.Series = self._argumentsDict[self.ARG_RESULT]

        # pd.Series
        return result.copy()
