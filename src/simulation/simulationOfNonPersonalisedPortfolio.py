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

from simulation.evaluationTool.simplePositiveFeedback import SimplePositiveFeedback #class

import numpy as np
import pandas as pd

from pandas.core.frame import DataFrame #class

import os


class SimulationOfNonPersonalisedPortfolio:

    def __init__(self, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame, repetitionOfRecommendation:int=1, numberOfItems:int=20):
        if type(ratingsDF) is not DataFrame:
            raise ValueError("Argument ratingsDF is not type DataFrame.")
        if type(usersDF) is not DataFrame:
            raise ValueError("Argument usersDF is not type DataFrame.")
        if type(itemsDF) is not DataFrame:
            raise ValueError("Argument itemsDF is not type DataFrame.")

        if type(repetitionOfRecommendation) is not int:
            raise ValueError("Argument repetitionOfRecommendation is not type int.")
        if type(numberOfItems) is not int:
            raise ValueError("Argument numberOfItems is not type int.")

        self._ratingsDF:DataFrame = ratingsDF
        self._usersDF:DataFrame = usersDF
        self._itemsDF:DataFrame = itemsDF

        self._repetitionOfRecommendation:int = repetitionOfRecommendation
        self._numberOfItems:int = numberOfItems


    def run(self, portfolioDescs:List[PortfolioDescription], historyModels:List[pd.DataFrame]):
        if type(portfolioDescs) is not list:
            raise ValueError("Argument portfolioDescs isn't type list.")
        for portfolioDescI in portfolioDescs:
            if type(portfolioDescI) is not PortfolioDescription:
               raise ValueError("Argument portfolioDescs don't contain PortfolioDescription.")

        ratingsSortedDF:DataFrame = self._ratingsDF.sort_values(by=Ratings.COL_TIMESTAMP)
        numberOfRatings:int = ratingsSortedDF.shape[0]

        # dataset division setting
        #divisionDatasetPercentualSizes:List[int] = [50, 60, 70, 80, 90]
        divisionDatasetPercentualSizes:List[int] = [50]
        testDatasetPercentualSize:int = 10

        # dataset division
        percentualSizeI:int
        for percentualSizeI in divisionDatasetPercentualSizes:
            trainSize:int = (int)(numberOfRatings * percentualSizeI / 100)
            trainDFI:DataFrame = ratingsSortedDF[0:trainSize]

            testSize:int = (int)(numberOfRatings * testDatasetPercentualSize / 100)
            testDFI:DataFrame = ratingsSortedDF[trainSize:(trainSize + testSize)]

            self.__runPortfolioDesc(portfolioDescs, historyModels, trainDFI, testDFI)


    def __runPortfolioDesc(self, portfolioDescs:List[PortfolioDescription], historyModels:List[pd.DataFrame],
                           trainRatingsDF:DataFrame, testRatingsDF:DataFrame):

        historyDF:DataFrame = None

        portfolios:List[Portfolio] = []

        portfolioDescI:PortfolioDescription
        for portfolioDescI in portfolioDescs:

            print("Training mode: " + str(portfolioDescI.getPortfolioID()))

            # train portfolio model
            portfolioI: Portfolio = portfolioDescI.exportPortfolio()
            portfolioI.train(historyDF, trainRatingsDF, self._usersDF, self._itemsDF)
            portfolios.append(portfolioI)

        self.__iterateOverDataset(portfolios, historyModels, testRatingsDF)


    def __iterateOverDataset(self, portfolios:List[Portfolio], historyModels:List[pd.DataFrame], testRatingsDF:DataFrame):

        counterI:int = 0

        rIndexI:int
        for (currentIndexI, nextIndexI) in zip(list(testRatingsDF.index[1:]), list(testRatingsDF.index)[:-1]):

            counterI += 1
            if counterI  % 100 == 0:
                print("RatingI: " + str(counterI) + " / " + str(testRatingsDF.shape[0]))

            currentItemI:int = testRatingsDF.loc[currentIndexI][Ratings.COL_MOVIEID]
            nextItemI:int = testRatingsDF.loc[nextIndexI][Ratings.COL_MOVIEID]

            repetitionI:int
            for repetitionI in range(self._repetitionOfRecommendation):
                self.__simulateRecommendations(portfolios, historyModels, currentItemI, nextItemI)


    def __simulateRecommendations(self, portfolios:List[Portfolio], historyModels:List[pd.DataFrame], currentItem:int, nextItem:int):

        portfolioI:Portfolio
        historyModelI:pd.DataFrame
        for portfolioI, historyModelI in zip(portfolios, historyModels):
           self.__simulateRecommendation(portfolioI, historyModelI, currentItem, nextItem)


    def __simulateRecommendation(self, portfolio:Portfolio, historyModel:pd.DataFrame, currentItem:int, nextItem:int):

        # aggregatedItemIDsWithResponsibility:list<(itemID:int, Series<(rating:int, methodID:str)>)>
        aggregatedItemIDsWithResponsibility: List[tuple[int, Series[int, str]]] = portfolio.test(
            historyModel, currentItem, numberOfItems=self._numberOfItems)

        SimplePositiveFeedback.evaluate(aggregatedItemIDsWithResponsibility, nextItem, historyModel)


