#!/usr/bin/python3

from typing import List

from pandas.core.series import Series #class

from datasets.ratings import Ratings #class

from datasets.users import Users #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class
from portfolio.portfolio1Aggr import Portfolio1Aggr #class

import pandas as pd

from pandas.core.frame import DataFrame #class


class SimulationOfPersonalisedPortfolio:

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


    def run(self, portfolioDesc:Portfolio1AggrDescription):

        if type(portfolioDesc) is not Portfolio1AggrDescription:
            raise ValueError("Argument portfolioDesc is not type PortfolioDescription.")


        ratingsSortedDF:DataFrame = self._ratingsDF.sort_values(by=Ratings.COL_TIMESTAMP)
        numberOfRatings:int = ratingsSortedDF.shape[0]

        userIDs:List[int] = self._usersDF[Users.COL_USERID].tolist()
        #userIDs:List[int] = [1]


        # dataset division setting
        divisionDatasetPercentualSizes:List[int] = [50, 60, 70, 80, 90]
        testDatasetPercentualSize:int = 10

        # dataset division
        percentualSizeI:int
        for percentualSizeI in divisionDatasetPercentualSizes:
            trainSize:int = (int)(numberOfRatings * percentualSizeI / 100)
            trainDFI:DataFrame = ratingsSortedDF[0:trainSize]

            testSize:int = (int)(numberOfRatings * testDatasetPercentualSize / 100)
            testDFI:DataFrame = ratingsSortedDF[trainSize:(trainSize + testSize)]

            self.__simulateUsers(userIDs, portfolioDesc, trainDFI, testDFI, repetition=self._repetitionOfRecommendation)


    # userIDs:int, portfolioDesc:PortfolioDescription, trainDF:DataFrame, testDF:DataFrame, repetition:int
    def __simulateUsers(self, userIDs:int, portfolioDesc:Portfolio1AggrDescription, trainRatingsDF:DataFrame,
                        testRatingsDF:DataFrame, repetition:int=1):

        historyDF:DataFrame = None

        # train model
        portfolio:Portfolio1Aggr = portfolioDesc.exportPortfolio()
        portfolio.train(historyDF, trainRatingsDF, self._usersDF, self._itemsDF)

        userIdI:int
        for userIdI in userIDs:
            #print("UserID " + str(userIdI))

            # select ratings of userIDI
            testRatingsUserIDF:DataFrame = testRatingsDF.loc[testRatingsDF[Ratings.COL_USERID == userIdI]]

            # test model
            self.__simulateUser(userIdI, portfolio, testRatingsUserIDF)



    # userID:int, portfolio, testDF, repetition:int
    def __simulateUser(self, userID:int, portfolio:Portfolio1Aggr, testDF:DataFrame):
        #print("UserID " + str(userID))

        recommIDs:List[str] = portfolio.getRecommIDs()

        # methods parametes
        methodsParamsData = [[rIdI, 0] for rIdI in portfolio.getRecommIDs()]
        methodsParamsDF = pd.DataFrame(methodsParamsData, columns=["methodID", "votes"])
        methodsParamsDF.set_index("methodID", inplace=True)


        counterI:int = 0

        rIndexI:int
        for (currentIndexI, nextIndexI) in zip(list(testDF.index[1:]), list(testDF.index)[:-1]):

            counterI += 1
            if counterI  % 100 == 0:
                print("RatingI: " + str(counterI) + " / " + str(testDF.shape[0]))

            currentItemI:int = testDF.loc[currentIndexI][Ratings.COL_MOVIEID]
            nextItemI:int = testDF.loc[nextIndexI][Ratings.COL_MOVIEID]

            for repetitionI in range(self._repetitionOfRecommendation):
                self.__simulateRecommendation(portfolio, currentItemI, nextItemI, methodsParamsDF)


    def __simulateRecommendation(self, portfolio:Portfolio1Aggr, currentItem:int, nextItem:int, methodsParamsDF:DataFrame):

        # aggregatedItemIDsWithResponsibility:list<(itemID:int, Series<(rating:int, methodID:str)>)>
        aggregatedItemIDsWithResponsibility: List[tuple[int, Series[int, str]]] = portfolio.test(
            methodsParamsDF, currentItem, numberOfItems=self._numberOfItems)

        EToolDHontHitIncrementOfResponsibility.evaluate(aggregatedItemIDsWithResponsibility, nextItem, methodsParamsDF)

