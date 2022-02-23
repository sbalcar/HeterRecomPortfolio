#!/usr/bin/python3

from pandas.core.frame import DataFrame  # class

import os
import json
from typing import List
from typing import Dict  # class

import pandas as pd
import numpy as np

from datasets.aDataset import ADataset  # class

from recommenderDescription.recommenderDescription import RecommenderDescription  # class

from aggregation.aAggregation import AAgregation  # class

from recommender.aRecommender import ARecommender  # class

from history.aHistory import AHistory  # class
from portfolio.aPortfolio import APortfolio  # class

from configuration.configuration import Configuration  # class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class
from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from portfolio.portfolioDynamic import PortfolioDynamic #class

from aggregation.operators.aDHondtSelector import ADHondtSelector  # class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector  # class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector  # class

from aggregation.negImplFeedback.aPenalization import APenalization #class
from batchDefinition.inputAggrDefinition import PenalizationToolDefinition #class

from aggregationDescription.aggregationDescription import AggregationDescription #class
from recommenderDescription.recommenderDescription import RecommenderDescription #class

from aggregation.negImplFeedback.aPenalization import APenalization #class
from batchDefinition.inputAggrDefinition import PenalizationToolDefinition #class



class PortfolioDynamicDescription(APortfolioDescription):

    def __init__(self, portfolioID:str, recommID:str, recommDescr:RecommenderDescription,
                 p1AggrID:str, p1AggrDescr:Portfolio1AggrDescription):
        if type(portfolioID) is not str:
            raise ValueError("Type of portfolioID is not str.")
        if type(recommID) is not str:
            raise ValueError("Type of argument recommID isn't str.")
        if type(recommDescr) is not RecommenderDescription:
            raise ValueError("Type of argument recommDescr isn't RecommenderDescription.")
        if type(p1AggrID) is not str:
            raise ValueError("Type of argument p1AggrID isn't str.")
        if type(p1AggrDescr) is not Portfolio1AggrDescription:
            raise ValueError("Type of argument p1AggrDescr isn't Portfolio1AggrDescription.")

        self._portfolioID:str = portfolioID
        self._recommID:str = recommID
        self._recommDescr:RecommenderDescription = recommDescr
        self._p1AggrID:str = p1AggrID
        self._p1AggrDescr:Portfolio1AggrDescription = p1AggrDescr


    def getPortfolioID(self):
        return self._portfolioID


    def exportPortfolio(self, batchID:str, history:AHistory):
        if type(batchID) is not str:
            raise ValueError("Type of argument batchID isn't str.")
        if type(batchID) is not str:
            raise ValueError("Type of argument jobID isn't str.")
        if not isinstance(history, AHistory):
            raise ValueError("Type of argument history isn't AHistory.")

        recommender:ARecommender = self._recommDescr.exportRecommender(batchID)

        p1Aggr = self._p1AggrDescr.exportPortfolio(batchID, history)

        return PortfolioDynamic(batchID, self.getPortfolioID(), recommender, self._recommID,
                             p1Aggr, self._p1AggrID)
