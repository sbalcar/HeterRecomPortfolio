#!/usr/bin/python3

import os

from configuration.configuration import Configuration #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.ratings import Ratings #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class

from aggregation.negImplFeedback.penalUsingFiltering import PenalUsingFiltering #class
from aggregation.negImplFeedback.aPenalization import APenalization #class

import pandas as pd
from history.aHistory import AHistory #class
from history.historySQLite import HistorySQLite #class
from history.historyDF import HistoryDF #class
from history.historyHierDF import HistoryHierDF #class


import pandas as pd


def test01():
    print("Test 01")
    os.chdir("..")

    borderNegFeedback:float = 1.0
    lengthOfHistory:int = 3

    userID:int = 1

    history:AHistory = HistoryHierDF("databse1")

    # userID, itemID, position, observation, clicked
    history.insertRecommendation(userID, 100, 1, 0.5, False)
    history.insertRecommendation(userID, 100, 1, 0.5, True)
    history.insertRecommendation(userID, 100, 1, 0.5, True)
    history.insertRecommendation(userID, 100, 1, 0.5, True)


    recommendationDict:dict = {100:0.35, 125:0.25, 95:0.15, 45:0.1, 78:0.05, 68:0.05, 32:0.02, 6:0.01, 18:0.01, 47:0.01}
    recommendationSrs:Series = Series(recommendationDict)

    penalty:APenalization = PenalUsingFiltering(borderNegFeedback, lengthOfHistory)
    pRecommendationSrs:Series = penalty.runOneMethodPenalization(userID, recommendationSrs, history)

    print(pRecommendationSrs)

test01()