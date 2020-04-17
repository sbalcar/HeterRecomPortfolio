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


    def run(self, portfolioDesc:PortfolioDescription):

        if type(portfolioDesc) is not PortfolioDescription:
            raise ValueError("Argument portfolioDesc is not type PortfolioDescription.")


        ratingsSortedDF:DataFrame = self._ratingsDF.sort_values(by=Ratings.COL_TIMESTAMP)
        numberOfRatings:int = ratingsSortedDF.shape[0]

        userIDs:List[int] = self._usersDF[Users.COL_USERID].tolist()
        #userIDs:List[int] = [1]


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

            self.__simulateDataset(portfolioDesc, trainDFI, testDFI)


    # portfolioDesc:PortfolioDescription, trainDF:DataFrame, testDF:DataFrame, repetition:int
    def __simulateDataset(self, portfolioDesc:PortfolioDescription, trainRatingsDF:DataFrame, testRatingsDF:DataFrame):

        historyDF:DataFrame = None

        # train model
        portfolio:Portfolio = portfolioDesc.exportPortfolio()
        portfolio.train(historyDF, trainRatingsDF, self._usersDF, self._itemsDF)

        # methods parametes
        methodsParamsData = [[rIdI, 0] for rIdI in portfolio.getRecommIDs()]
        methodsParamsDF = pd.DataFrame(methodsParamsData, columns=["methodID", "votes"])
        methodsParamsDF.set_index("methodID", inplace=True)

        counterI:int = 0

        rIndexI:int
        for (currentIndexI, nextIndexI) in zip(list(testRatingsDF.index[1:]), list(testRatingsDF.index)[:-1]):

            counterI += 1
            if counterI  % 100 == 0:
                print("RatingI: " + str(counterI) + " / " + str(testRatingsDF.shape[0]))

            currentItemI:int = testRatingsDF.loc[currentIndexI][Ratings.COL_MOVIEID]
            nextItemI:int = testRatingsDF.loc[nextIndexI][Ratings.COL_MOVIEID]

            for repetitionI in range(self._repetitionOfRecommendation):

                self.__simulateRecommendation(portfolio, currentItemI, nextItemI, methodsParamsDF)


    def __simulateRecommendation(self, portfolio:Portfolio, currentItem:int, nextItem:int, methodsParamsDF:DataFrame):

        # aggregatedItemIDsWithResponsibility:list<(itemID:int, Series<(rating:int, methodID:str)>)>
        aggregatedItemIDsWithResponsibility: List[tuple[int, Series[int, str]]] = portfolio.test(
            methodsParamsDF, currentItem, numberOfItems=self._numberOfItems)

        SimplePositiveFeedback.evaluate(aggregatedItemIDsWithResponsibility, nextItem, methodsParamsDF)


