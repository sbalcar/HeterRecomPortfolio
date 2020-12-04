#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.ratings import Ratings #class

from datasets.users import Users #class

from portfolio.aPortfolio import APortfolio #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class
from portfolio.portfolio1Aggr import Portfolio1Aggr #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class

from simulation.tools.userBehaviourSimulator import UserBehaviourSimulator #class

from evaluationTool.aEvalTool import AEvalTool #class

from history.aHistory import AHistory #class

from simulation.aSequentialSimulation import ASequentialSimulation #class

import pandas as pd
import numpy as np


class SimulationPortfoliosRecomToItemSeparatedUsers(ASequentialSimulation):

    def __init__(self, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame,
                 uBehaviourDesc:UserBehaviourDescription, repetitionOfRecommendation:int=1, numberOfItems:int=20):

        if type(ratingsDF) is not DataFrame:
            raise ValueError("Argument ratingsDF isn't type DataFrame.")
        if type(usersDF) is not DataFrame:
            raise ValueError("Argument usersDF isn't type DataFrame.")
        if type(itemsDF) is not DataFrame:
            raise ValueError("Argument itemsDF isn't type DataFrame.")
        if type(uBehaviourDesc) is not UserBehaviourDescription:
            raise ValueError("Argument uBehaviourDesc isn't type UserBehaviourDescription.")

        if type(repetitionOfRecommendation) is not int:
            raise ValueError("Argument repetitionOfRecommendation isn't type int.")
        if type(numberOfItems) is not int:
            raise ValueError("Argument numberOfItems isn't type int.")

        self._ratingsDF:DataFrame = ratingsDF
        self._usersDF:DataFrame = usersDF
        self._itemsDF:DataFrame = itemsDF

        self._uBehaviourDesc = uBehaviourDesc

        self._repetitionOfRecommendation:int = repetitionOfRecommendation
        self._numberOfItems:int = numberOfItems


    def run(self, portfolioDescs:List[APortfolioDescription], portFolioModels:List[pd.DataFrame],
            evaluatonTools:List, histories:List[AHistory]):
        if type(portfolioDescs) is not list:
            raise ValueError("Argument portfolioDescs isn't type list.")
        for portfolioDescI in portfolioDescs:
            if not isinstance(portfolioDescI, APortfolioDescription):
                print(type(portfolioDescI))
                raise ValueError("Argument portfolioDescs don't contain PortfolioDescription.")

        if type(portFolioModels) is not list:
            raise ValueError("Argument portFolioModels isn't type list.")
        for portFolioModelI in portFolioModels:
            if type(portFolioModelI) is not pd.DataFrame:
               raise ValueError("Argument portFolioModels don't contain pd.DataFrame.")

        if type(evaluatonTools) is not list:
            raise ValueError("Argument evaluatonTools isn't type list.")

        if type(histories) is not list:
            raise ValueError("Argument histories isn't type list.")
        for historyI in histories:
            if not isinstance(historyI, AHistory):
               raise ValueError("Argument histories don't contain AHistory.")


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

            self.__simulateUsers(userIDs, portfolioDescs, portFolioModels, evaluatonTools, histories,
                                 trainDFI, testDFI, repetition=self._repetitionOfRecommendation)


    # userIDs:int, portfolioDesc:PortfolioDescription, trainDF:DataFrame, testDF:DataFrame, repetition:int
    def __simulateUsers(self, userIDs:int, portfolioDescs:List[APortfolioDescription], portFolioModels:List[pd.DataFrame],
            evaluatonTools:List, histories:List[AHistory], trainRatingsDF:DataFrame, testRatingsDF:DataFrame, repetition:int=1):

        portfolios:List[APortfolio] = []
        for portfolioDescI, historyI in zip(portfolioDescs, histories):

            portfolioI:Portfolio1Aggr = portfolioDescI.exportPortfolio(self._uBehaviourDesc, historyI)
            portfolioI.train(historyI, trainRatingsDF, self._usersDF, self._itemsDF)
            portfolios.append(portfolioI)

        userIdI:int
        for userIdI in userIDs:
            #print("UserID " + str(userIdI))

            # select ratings of userIDI
            testRatingsUserIDF:DataFrame = testRatingsDF.loc[testRatingsDF[Ratings.COL_USERID] == userIdI]

            # test model
            self.__simulateUser(userIdI, portfolios, portFolioModels, evaluatonTools, histories, testRatingsUserIDF)



    # userID:int, portfolio, testDF, repetition:int
    def __simulateUser(self, userID:int, portfolios:List[APortfolio], portFolioModels:List[pd.DataFrame],
                             evaluatonTools:[AEvalTool], histories:List[AHistory], testRatingsDF:DataFrame):

        print("UserID " + str(userID))

        evaluations:List[dict] = [{} for i in range(len(portfolios))]

        counterI:int = 0

        currentIndexI:int
        nextIndexI:int
        for (currentIndexI, nextIndexI) in zip(list(testRatingsDF.index[1:]), list(testRatingsDF.index)[:-1]):

            counterI += 1
            if counterI  % 100 == 0:
                print("RatingI: " + str(counterI) + " / " + str(testRatingsDF.shape[0]))
## TODO ######
#            if counterI == 1000:
#                return evaluations
## TODO ######

            currentItemIdI:int = testRatingsDF.loc[currentIndexI][Ratings.COL_MOVIEID]
            nextItemIdI:int = testRatingsDF.loc[nextIndexI][Ratings.COL_MOVIEID]

            nextItemRatingI:int = testRatingsDF.loc[nextIndexI][Ratings.COL_RATING]
            if nextItemRatingI < 4:
              continue

            repetitionI:int
            for repetitionI in range(self._repetitionOfRecommendation):
                self.__simulateRecommendations(portfolios, portFolioModels, evaluatonTools, testRatingsDF,
                                               histories, evaluations, userID, currentItemIdI, nextItemIdI)
            portfolioI:APortfolio
            for portfolioI in portfolios:
                portfolioI.update(pd.DataFrame(testRatingsDF.loc[nextIndexI]))

        return evaluations

    def __simulateRecommendations(self, portfolios:List[Portfolio1Aggr], portFolioModels:List[pd.DataFrame],
                                  evaluatonTools:[AEvalTool], testRatingsDF:DataFrame, histories:List[AHistory],
                                  evaluations:List[dict], userID:int, currentItem:int, nextItem:int):

        uProbOfObservGenerated:List[float] = UserBehaviourSimulator().simulateStaticProb(self._uBehaviourDesc,
                                                                                         self._numberOfItems)
        #print("uProbOfObservGenerated: " + str(uProbOfObservGenerated))

        uObservation:List[bool] = list(map(lambda x, y: x > y, uProbOfObservGenerated, np.random.uniform(low=0.0, high=1.0, size=self._numberOfItems)))
        #print("uObservation: " + str(uObservation))

        portfolioI:Portfolio1Aggr
        portFolioModelI:pd.DataFrame
        historyI:pd.DataFrame
        for portfolioI, portFolioModelI, evaluatonToolI, historyI, evaluationI in zip(portfolios, portFolioModels, evaluatonTools, histories, evaluations):
            self.__simulateRecommendation(portfolioI, portFolioModelI, evaluatonToolI, testRatingsDF, historyI,
                                          evaluationI, uProbOfObservGenerated, uObservation, userID, currentItem, nextItem)


    def __simulateRecommendation(self, portfolio:Portfolio1Aggr, portfolioModel:pd.DataFrame, evaluatonTool:AEvalTool, testRatingsDF:DataFrame,
                                 history:AHistory, evaluation:dict, uProbOfObserv:List[float], uObservation:List[bool], userID:int,
                                 currentItemID:int, nextItemID:int):

        rItemIDs:List[int]
        rItemIDsWithResponsibility:List[tuple[int, Series[int, str]]]
        rItemIDs, rItemIDsWithResponsibility = portfolio.recommend(
            userID, portfolioModel, testRatingsDF, history, numberOfItems=self._numberOfItems)

        print(testRatingsDF)

        if not nextItemID in rItemIDs:
            return

        index:int = rItemIDs.index(nextItemID)
        probOfObserv:float = uProbOfObserv[index]
        wasObserved:bool = uObservation[index]


        if wasObserved:
            clickedItemID: int = nextItemID

            evaluatonTool.click(rItemIDsWithResponsibility, clickedItemID, probOfObserv, portfolioModel, evaluation)

            # save log of history
            history.insertRecommendations(userID, rItemIDs, uProbOfObserv, clickedItemID)

        else:
            evaluatonTool.ignore(rItemIDsWithResponsibility, portfolioModel, evaluation)

            # save log of history
            history.insertRecommendations(userID, rItemIDs, uProbOfObserv, None)
