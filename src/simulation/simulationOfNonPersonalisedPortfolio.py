#!/usr/bin/python3

from typing import List

from pandas.core.series import Series #class

from datasets.ratings import Ratings #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class
from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class
from portfolio.portfolio1Aggr import Portfolio1Aggr #class

import pandas as pd

from pandas.core.frame import DataFrame #class

from history.aHistory import AHistory #class
from evaluationTool.aEvalTool import AEvalTool #class

from simulation.userBehaviourSimulator.userBehaviourSimulator import UserBehaviourSimulator #class
from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class


class SimulationOfNonPersonalisedPortfolio:

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

            evaluationI = self.__runPortfolioDesc(portfolioDescs, portFolioModels, evaluatonTools, histories, trainDFI, testDFI)
            evaluations.append(evaluationI)

        return evaluations


    def __runPortfolioDesc(self, portfolioDescs:List[Portfolio1AggrDescription], portFolioModels:List[pd.DataFrame],
                           evaluatonTools:[AEvalTool], histories:List[AHistory], trainRatingsDF:DataFrame, testRatingsDF:DataFrame):

        portfolios:List[Portfolio1Aggr] = []

        portfolioDescI:Portfolio1AggrDescription
        historyI:AHistory
        for portfolioDescI, historyI in zip(portfolioDescs, histories):

            print("Training mode: " + str(portfolioDescI.getPortfolioID()))

            # train portfolio model
            portfolioI: Portfolio1Aggr = portfolioDescI.exportPortfolio(self._uBehaviourDesc, historyI)
            portfolioI.train(trainRatingsDF, self._usersDF, self._itemsDF)
            portfolios.append(portfolioI)


        return self.__iterateOverDataset(portfolios, portFolioModels, evaluatonTools, histories, testRatingsDF)


    def __iterateOverDataset(self, portfolios:List[Portfolio1Aggr], portFolioModels:List[pd.DataFrame],
                             evaluatonTools:[AEvalTool], histories:List[AHistory], testRatingsDF:DataFrame):

        evaluations:List[dict] = [{} for i in range(len(portfolios))]

        counterI:int = 0

        currentIndexI:int
        nextIndexI:int
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
                self.__simulateRecommendations(portfolios, portFolioModels, evaluatonTools, testRatingsDF,
                                               histories, evaluations, currentItemIdI, nextItemIdI)
            # TODO
            for portfolioI in portfolios:
                portfolioI.update(pd.DataFrame(testRatingsDF.loc[nextIndexI]))

        return evaluations

    def __simulateRecommendations(self, portfolios:List[Portfolio1Aggr], portFolioModels:List[pd.DataFrame],
                                  evaluatonTools:[AEvalTool], testRatingsDF:DataFrame, histories:List[AHistory],
                                  evaluations:List[dict], currentItem:int, nextItem:int):

        uObservation:List[bool] = UserBehaviourSimulator().simulate(self._uBehaviourDesc, self._numberOfItems)

        portfolioI:Portfolio1Aggr
        portFolioModelI:pd.DataFrame
        historyI:pd.DataFrame
        for portfolioI, portFolioModelI, evaluatonToolI, historyI, evaluationI in zip(portfolios, portFolioModels, evaluatonTools, histories, evaluations):
            self.__simulateRecommendation(portfolioI, portFolioModelI, evaluatonToolI, testRatingsDF, historyI,
                                          evaluationI, uObservation, currentItem, nextItem)


    def __simulateRecommendation(self, portfolio:Portfolio1Aggr, portFolioModel:pd.DataFrame, evaluatonTool:AEvalTool,
                                 testRatingsDF:DataFrame, history:AHistory, evaluation:dict, uObservation:List[bool],
                                 currentItemID:int, nextItem:int):

        rItemIDs:List[int]
        rItemIDsWithResponsibility:List[tuple[int, Series[int, str]]]
        rItemIDs, rItemIDsWithResponsibility = portfolio.recommendToItem(
            portFolioModel, currentItemID, testRatingsDF, history, numberOfItems=self._numberOfItems)

        # save log of history
        history.addRecommendation(currentItemID, rItemIDs, uObservation)

        # evaluation
        evaluatonTool.evaluate(rItemIDs, rItemIDsWithResponsibility, nextItem, portFolioModel, evaluation)
