#!/usr/bin/python3

from typing import List
from typing import ClassVar

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

import os
from history.aHistory import AHistory #class

class SimulationOfNonPersonalisedPortfolio:

    def __init__(self, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame, evaluatonTool:ClassVar,
                 repetitionOfRecommendation:int=1, numberOfItems:int=20):

        if type(ratingsDF) is not DataFrame:
            raise ValueError("Argument ratingsDF isn't type DataFrame.")
        if type(usersDF) is not DataFrame:
            raise ValueError("Argument usersDF isn't type DataFrame.")
        if type(itemsDF) is not DataFrame:
            raise ValueError("Argument itemsDF isn't type DataFrame.")
        if evaluatonTool is None:
            raise ValueError("Argument evaluatonTool is None.")

        if type(repetitionOfRecommendation) is not int:
            raise ValueError("Argument repetitionOfRecommendation isn't type int.")
        if type(numberOfItems) is not int:
            raise ValueError("Argument numberOfItems isn't type int.")

        self._ratingsDF:DataFrame = ratingsDF
        self._usersDF:DataFrame = usersDF
        self._itemsDF:DataFrame = itemsDF
        self._evaluatonTool = evaluatonTool

        self._repetitionOfRecommendation:int = repetitionOfRecommendation
        self._numberOfItems:int = numberOfItems


    def run(self, portfolioDescs:List[PortfolioDescription], portFolioModels:List[pd.DataFrame], histories:List[AHistory]):
        if type(portfolioDescs) is not list:
            raise ValueError("Argument portfolioDescs isn't type list.")
        for portfolioDescI in portfolioDescs:
            if type(portfolioDescI) is not PortfolioDescription:
               raise ValueError("Argument portfolioDescs don't contain PortfolioDescription.")

        if type(portFolioModels) is not list:
            raise ValueError("Argument portFolioModels isn't type list.")
        for portFolioModelI in portFolioModels:
            if type(portFolioModelI) is not pd.DataFrame:
               raise ValueError("Argument portFolioModels don't contain pd.DataFrame.")

        if type(histories) is not list:
            raise ValueError("Argument histories isn't type list.")
        for historyI in histories:
            if not isinstance(historyI, AHistory):
               raise ValueError("Argument histories don't contain AHistory.")

        ratingsSortedDF:DataFrame = self._ratingsDF.sort_values(by=Ratings.COL_TIMESTAMP)
        numberOfRatings:int = ratingsSortedDF.shape[0]

        # dataset division setting
        #divisionDatasetPercentualSizes:List[int] = [50, 60, 70, 80, 90]
        divisionDatasetPercentualSizes:List[int] = [50]
        testDatasetPercentualSize:int = 10

        evaluations:List[int] = []

        # dataset division
        percentualSizeI:int
        for percentualSizeI in divisionDatasetPercentualSizes:
            trainSize:int = (int)(numberOfRatings * percentualSizeI / 100)
            trainDFI:DataFrame = ratingsSortedDF[0:trainSize]

            testSize:int = (int)(numberOfRatings * testDatasetPercentualSize / 100)
            testDFI:DataFrame = ratingsSortedDF[trainSize:(trainSize + testSize)]

            evaluationI = self.__runPortfolioDesc(portfolioDescs, portFolioModels, histories, trainDFI, testDFI)
            evaluations.append(evaluationI)

        return evaluations


    def __runPortfolioDesc(self, portfolioDescs:List[PortfolioDescription], portFolioModels:List[pd.DataFrame],
                           histories:List[AHistory], trainRatingsDF:DataFrame, testRatingsDF:DataFrame):

        portfolios:List[Portfolio] = []

        portfolioDescI:PortfolioDescription
        for portfolioDescI in portfolioDescs:

            print("Training mode: " + str(portfolioDescI.getPortfolioID()))

            # train portfolio model
            portfolioI: Portfolio = portfolioDescI.exportPortfolio()
            portfolioI.train(trainRatingsDF, self._usersDF, self._itemsDF)
            portfolios.append(portfolioI)


        return self.__iterateOverDataset(portfolios, portFolioModels, histories, testRatingsDF)


    def __iterateOverDataset(self, portfolios:List[Portfolio], portFolioModels:List[pd.DataFrame],
                             histories:List[AHistory], testRatingsDF:DataFrame):

        evaluations:List[dict] = [{} for i in range(len(portfolios))]

        counterI:int = 0

        rIndexI:int
        for (currentIndexI, nextIndexI) in zip(list(testRatingsDF.index[1:]), list(testRatingsDF.index)[:-1]):

            counterI += 1
            if counterI  % 100 == 0:
                print("RatingI: " + str(counterI) + " / " + str(testRatingsDF.shape[0]))
## TODO ######
            if counterI == 1000:
                return evaluations
## TODO ######
            currentItemIdI:int = testRatingsDF.loc[currentIndexI][Ratings.COL_MOVIEID]
            nextItemIdI:int = testRatingsDF.loc[nextIndexI][Ratings.COL_MOVIEID]
            nextItemRatingI:int = testRatingsDF.loc[nextIndexI][Ratings.COL_RATING]

            if nextItemRatingI < 4:
              continue

            repetitionI:int
            for repetitionI in range(self._repetitionOfRecommendation):
                self.__simulateRecommendations(portfolios, portFolioModels, testRatingsDF,
                                               histories, evaluations, currentItemIdI, nextItemIdI)
        return evaluations

    def __simulateRecommendations(self, portfolios:List[Portfolio], portFolioModels:List[pd.DataFrame], testRatingsDF:DataFrame,
                                  histories:List[AHistory], evaluations:List[dict], currentItem:int, nextItem:int):

        portfolioI:Portfolio
        portFolioModelI:pd.DataFrame
        historyI:pd.DataFrame
        for portfolioI, portFolioModelI, historyI, evaluationI in zip(portfolios, portFolioModels, histories, evaluations):
            self.__simulateRecommendation(portfolioI, portFolioModelI, testRatingsDF, historyI, evaluationI, currentItem, nextItem)


    def __simulateRecommendation(self, portfolio:Portfolio, portFolioModel:pd.DataFrame,
                                 testRatingsDF: DataFrame, history:AHistory, evaluation:dict, currentItemID:int, nextItem:int):

        # aggregatedItemIDsWithResponsibility:list<(itemID:int, Series<(rating:int, methodID:str)>)>
        aggregatedItemIDsWithResponsibility: List[tuple[int, Series[int, str]]] = portfolio.test(
            portFolioModel, currentItemID, testRatingsDF, history, numberOfItems=self._numberOfItems)

        history.addRecommendation(currentItemID, aggregatedItemIDsWithResponsibility)


        self._evaluatonTool.evaluate(aggregatedItemIDsWithResponsibility, nextItem, portFolioModel, evaluation)
