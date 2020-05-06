#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

from typing import List

import random

from configuration.arguments import Arguments #class
from configuration.argument import Argument #class

from recommender.aRecommender import ARecommender #class

from recommendation.recommendation import Recommendation #class
from recommendation.resultOfRecommendation import ResultOfRecommendation #class

from datasets.ratings import Ratings #class
from history.aHistory import AHistory #class

class RecommenderTheMostPopular(ARecommender):

    def __init__(self, arguments: List[Argument]):
        if type(arguments) is not Arguments:
            raise ValueError("Argument arguments is not type Arguments.")

        self._arguments:List[Argument] = arguments

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


    def recommendToItem(self, itemID:int, ratingsTestDF:DataFrame, history:AHistory, numberOfItems:int=20):

        # ratings:Dataframe<(movieId:int, ratings:int)>
        ratings:DataFrame = self._sortedAscRatings5CountDF.head(numberOfItems)

        movieIDs:List[int] = list(ratings.index)
        numberOfEvaluators:List[float] = [float(r) for r in ratings[Ratings.COL_RATING]]
        numberOfEvaluators:List[float] = [float(0.05) for r in ratings[Ratings.COL_RATING]]

        return ResultOfRecommendation(movieIDs, numberOfEvaluators)
