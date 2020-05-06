#!/usr/bin/python3

from typing import List

from simulation.evaluationTool.aEvalTool import AEvalTool #class

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
from aggregation.aggrDHont import AggrDHont #class

from recommendation.resultOfRecommendation import ResultOfRecommendation #class

import numpy as np
import pandas as pd

from pandas.core.frame import DataFrame #class

import os


class EvalToolHit1(AEvalTool):

    @staticmethod
    def evaluate(aggregatedItemIDsWithResponsibility:List, nextItem:int, methodsParamsDF:DataFrame, evaluationDict:dict):
        #print("aggregatedItemIDsWithResponsibility")
        #print(aggregatedItemIDsWithResponsibility)

        #print("methodsParamsDF")
        #print(methodsParamsDF)

        aggrItemIDsWithRespDF:DataFrame = DataFrame(aggregatedItemIDsWithResponsibility, columns=["itemId", "responsibility"])
        aggrItemIDsWithRespDF.set_index("itemId", inplace=True)
        #print("aggrItemIDsWithRespDF")
        #print(aggrItemIDsWithRespDF)

        if nextItem in aggrItemIDsWithRespDF.index:
            print("HOP")
            print("nextItem: " + str(nextItem))

            evaluationDict[AEvalTool.CTR] = evaluationDict.get(AEvalTool.CTR, 0) + 1

            #responsibilityDict:dict[methodID:str, votes:int]
            responsibilityDict:dict[str,int] = aggrItemIDsWithRespDF.loc[nextItem]["responsibility"]

            # increment user definition
            for methodIdI in responsibilityDict.keys():
                if responsibilityDict[methodIdI] > 0:
                   methodsParamsDF.loc[methodIdI] += 1
            print(methodsParamsDF)
