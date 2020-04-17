#!/usr/bin/python3

from typing import List

from pandas.core.series import Series #class

from recommender.description.recommenderDescription import RecommenderDescription #class

from recommender.aRecommender import ARecommender #class

from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.dummy.recommenderDummyRedirector import RecommenderDummyRedirector #class

from datasets.ratings import Ratings #class
from datasets.rating import Rating #class

from datasets.users import Users #class

from portfolio.portfolioDescription import PortfolioDescription #class
from portfolio.portfolio import Portfolio #class

from aggregation.aggregationDescription import AggregationDescription #class
from aggregation.aggrElections import AggrElections #class

from recommendation.resultOfRecommendation import ResultOfRecommendation #class

import numpy as np
import pandas as pd

from pandas.core.frame import DataFrame #class

import os


class SimplePositiveFeedback:

    @staticmethod
    def evaluate(aggregatedItemIDsWithResponsibility:List, nextItem:int, methodsParamsDF:DataFrame):
        #print("recommendation " + str(aggregatedItemIDsWithResponsibility))
        #print("nextItem " + str(nextItem))

        aggrItemIDsWithRespDF:DataFrame = DataFrame(aggregatedItemIDsWithResponsibility, columns=["itemId", "responsibility"])
        aggrItemIDsWithRespDF.set_index("itemId", inplace=True)
        #print(aggrItemIDsWithRespDF)

        if nextItem in aggrItemIDsWithRespDF.index:

            #responsibility:dict[methodID:str, votes:int]
            responsibility:dict[str, int] = aggrItemIDsWithRespDF.loc[nextItem]["responsibility"]
            print(responsibility)

            # increment user definition
            for methodIdI in responsibility.keys():
                methodsParamsDF.loc[methodIdI] += responsibility[methodIdI]
