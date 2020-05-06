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
from aggregation.aggrDHont import AggrDHont #class

from recommendation.resultOfRecommendation import ResultOfRecommendation #class

from simulation.evaluationTool.evalToolHitIncrementOfResponsibility import EvalToolHitIncrementOfResponsibility #class

import numpy as np
import pandas as pd

from pandas.core.frame import DataFrame #class

from history.aHistory import AHistory #class

import os


class HistoryDF(AHistory):

    ITEM_ID = "itemID"
    POSITION_IN_RECOMMENDATION = "positionInRecommendation"

    def __init__(self):

        historyData:pd.DataFrame = []
        self._historyDF:pd.DataFrame = pd.DataFrame(historyData, columns=[self.ITEM_ID, self.POSITION_IN_RECOMMENDATION])


    def addRecommendation(self, itemID:int, recommendedItemIDs:List[int]):

        new_row:pd.Series = pd.Series({self.ITEM_ID: itemID, self.POSITION_IN_RECOMMENDATION: recommendedItemIDs})
        self._historyDF.append(new_row, ignore_index=True)