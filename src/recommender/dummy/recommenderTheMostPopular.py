#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

from typing import List

import random

from recommender.aRecommender import ARecommender #class

from datasets.ratings import Ratings #class
from history.aHistory import AHistory #class

import pandas as pd

class RecommenderTheMostPopular(ARecommender):

    def __init__(self, argumentsDict:dict):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        self._argumentsDict:dict = argumentsDict

    # ratingsSum:Dataframe<(userId:int, movieId:int, ratings:int, timestamp:int)>
    def train(self, ratingsTrainDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
        if type(ratingsTrainDF) is not DataFrame:
            raise ValueError("Argument ratingsTrainDF is not type DataFrame.")

        # ratingsSum:Dataframe<(userId:int, movieId:int, ratings:int, timestamp:int)>
        ratings5DF:DataFrame = ratingsTrainDF.loc[ratingsTrainDF[Ratings.COL_RATING] >= 4]

        # ratingsSum:Dataframe<(movieId:int, ratings:int)>
        ratings5SumDF:DataFrame = DataFrame(ratings5DF.groupby(Ratings.COL_MOVIEID)[Ratings.COL_RATING].count())

        # sortedAscRatings5CountDF:Dataframe<(movieId:int, ratings:int)>
        sortedAscRatings5CountDF:DataFrame = ratings5SumDF.sort_values(by=Ratings.COL_RATING, ascending=False)
        #print(sortedAscRatings5CountDF)

        self._sortedAscRatings5CountDF:DataFrame = sortedAscRatings5CountDF


    def update(self, ratingsUpdateDF:DataFrame):
        pass


    def recommend(self, userID:int, numberOfItems:int=20):

        # ratings:Dataframe<(movieId:int, ratings:int)>
        ratingsDF:DataFrame = self._sortedAscRatings5CountDF.head(numberOfItems)

        items:List[int] = list(ratingsDF.index)
        ratings:List[float] = [1.0/len(items) for itemI in items]

        return pd.Series(ratings,index=items)
