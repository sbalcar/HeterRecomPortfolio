#!/usr/bin/python3

import random
from pandas.core.frame import DataFrame #class

from typing import List
from recommender.aRecommender import ARecommender #class

from history.aHistory import AHistory #class

import pandas as pd #class


class RecommenderDummyRandom(ARecommender):

    def __init__(self, argumentsDict:dict):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        self._argumentsDict:dict = argumentsDict

    def train(self, historyDF:DataFrame, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
        pass

    def update(self, ratingsUpdateDF:DataFrame):
        pass

    def recommend(self, userID:int, numberOfItems:int=20):
        items:List[int] = list(range(numberOfItems))
        #random.shuffle(items)

        ratings:List[float] = [1.0/len(items) for itemI in items]

        return pd.Series(ratings,index=items)
