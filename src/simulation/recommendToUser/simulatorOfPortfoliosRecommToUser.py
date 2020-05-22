#!/usr/bin/python3

from typing import List

from pandas.core.series import Series #class

from datasets.ratings import Ratings #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class
from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class
from portfolio.portfolio1Aggr import Portfolio1Aggr #class

from portfolio.aPortfolio import APortfolio #class

import pandas as pd
import numpy as np

from pandas.core.frame import DataFrame #class

from history.aHistory import AHistory #class
from evaluationTool.aEvalTool import AEvalTool #class

from simulation.tools.userBehaviourSimulator import UserBehaviourSimulator #class
from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class


class SimulationPortfolioToUser:

    ARG_WINDOW_SIZE:str = "windowSize"
    ARG_REPETITION_OF_RECOMMENDATION:str = "repetitionOfRecommendation"
    ARG_NUMBER_OF_ITEMS:str = "numberOfItems"

    def __init__(self, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame,
                 uBehaviourDesc:UserBehaviourDescription, argumentsDict:dict):

        if type(ratingsDF) is not DataFrame:
            raise ValueError("Argument ratingsDF isn't type DataFrame.")
        if type(usersDF) is not DataFrame:
            raise ValueError("Argument usersDF isn't type DataFrame.")
        if type(itemsDF) is not DataFrame:
            raise ValueError("Argument itemsDF isn't type DataFrame.")
        if type(uBehaviourDesc) is not UserBehaviourDescription:
            raise ValueError("Argument uBehaviourDesc isn't type UserBehaviourDescription.")

        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        self._ratingsDF:DataFrame = ratingsDF
        self._usersDF:DataFrame = usersDF
        self._itemsDF:DataFrame = itemsDF

        self._uBehaviourDesc = uBehaviourDesc

        self._windowSize:int = argumentsDict[self.ARG_WINDOW_SIZE]
        self._repetitionOfRecommendation:int = argumentsDict[self.ARG_REPETITION_OF_RECOMMENDATION]
        self._numberOfItems:int = argumentsDict[self.ARG_NUMBER_OF_ITEMS]



    def run(self, portfolioDescs:List[APortfolioDescription], portFolioModels:List[pd.DataFrame],
            evaluatonTools:List, histories:List[AHistory]):
        if type(portfolioDescs) is not list:
            raise ValueError("Argument portfolioDescs isn't type list.")
        for portfolioDescI in portfolioDescs:
            if not isinstance(portfolioDescI, APortfolioDescription):
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
            trainDFI.reset_index(drop=True, inplace=True)

            testSize:int = (int)(numberOfRatings * testDatasetPercentualSize / 100)
            testDFI:DataFrame = ratingsSortedDF[trainSize:(trainSize + testSize)]
            testDFI.reset_index(drop=True, inplace=True)

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
            portfolioI.train(historyI, trainRatingsDF.copy(), self._usersDF.copy(), self._itemsDF.copy())
            portfolios.append(portfolioI)

        portIds:List[str] = [portDescI.getPortfolioID() for portDescI in portfolioDescs]

        return self.__iterateOverDataset(portfolios, portIds, portFolioModels, evaluatonTools, histories, testRatingsDF)


    def __iterateOverDataset(self, portfolios:List[APortfolio], portIds:List[str], portFolioModels:List[pd.DataFrame],
                             evaluatonTools:[AEvalTool], histories:List[AHistory], testRatingsDF:DataFrame):

        model:ModelOfIndexes = ModelOfIndexes(testRatingsDF)

        evaluations:List[dict] = [{} for i in range(len(portfolios))]

        counterI:int = 0

        currentIndexDFI:int
        nextIndexDFI:int
        for currentIndexDFI in list(testRatingsDF.index):

            counterI += 1
            if counterI  % 100 == 0:
                print("RatingI: " + str(counterI) + " / " + str(testRatingsDF.shape[0]))

                print("ids: " + str(portIds))
                print("Evaluations: " + str(evaluations))

## TODO ######
#            if counterI == 3000:
#                return evaluations
## TODO ######

            currentItemIdI:int = testRatingsDF.loc[currentIndexDFI][Ratings.COL_MOVIEID]
            currentRatingI:int = testRatingsDF.loc[currentIndexDFI][Ratings.COL_RATING]
            currentUserIdI:int = testRatingsDF.loc[currentIndexDFI][Ratings.COL_USERID]

            if currentRatingI < 4:
                continue

            nextItemIDsI:int = self.__getNextItemIDs(model, currentUserIdI, currentItemIdI, testRatingsDF, self._windowSize)
            if nextItemIDsI == []:
                continue

            portfolioI:APortfolio
            for portfolioI in portfolios:

                dfI:DataFrame = DataFrame([testRatingsDF.loc[currentIndexDFI]],
                    columns=[Ratings.COL_USERID, Ratings.COL_MOVIEID, Ratings.COL_RATING, Ratings.COL_TIMESTAMP])

                portfolioI.update(dfI)

            repetitionI: int
            for repetitionI in range(self._repetitionOfRecommendation):
                self.__simulateRecommendations(portfolios, portFolioModels, evaluatonTools, testRatingsDF,
                                               histories, evaluations, currentItemIdI, nextItemIDsI, currentUserIdI)

        return evaluations

    # model:ModelOfIndexes
    def __getNextItemIDs(self, model, userId:int, itemId:int, ratingsDF:DataFrame, windowSize:int):

        selectedItems:List[int] = []

        itemIdI:int = itemId
        for i in range(windowSize):
            nextIndexIdI:int = model.getNextIndex(userId, itemIdI)
            if nextIndexIdI is None:
                break

            nextItemIdI:int = ratingsDF.loc[nextIndexIdI][Ratings.COL_MOVIEID]
            nextUserIdI:int = ratingsDF.loc[nextIndexIdI][Ratings.COL_USERID]
            if nextUserIdI != userId:
                raise ValueError("Error")

            selectedItems.append(nextItemIdI)

            itemIdI = nextItemIdI

        return selectedItems

    def __simulateRecommendations(self, portfolios:List[APortfolio], portFolioModels:List[pd.DataFrame],
                                  evaluatonTools:[AEvalTool], testRatingsDF:DataFrame, histories:List[AHistory],
                                  evaluations:List[dict], currentItemID:int, nextItemIDs:List[int], userID:int):

        if type(portfolios) is not list:
            raise ValueError("Argument portfolios isn't type list.")
        for portfolioI in portfolios:
            if not isinstance(portfolioI, APortfolio):
               raise ValueError("Argument portfolioI don't contain APortfolio.")

        if type(portFolioModels) is not list:
            raise ValueError("Argument portFolioModels isn't type list.")
        for portfolioModelI in portFolioModels:
            if type(portfolioModelI) is not DataFrame:
               raise ValueError("Argument portfolioModelI don't contain DataFrame.")

        if type(testRatingsDF) is not DataFrame:
            raise ValueError("Argument testRatingsDF isn't type DataFrame.")

        if type(histories) is not list:
            raise ValueError("Argument histories isn't type list.")
        for historyI in histories:
            if not isinstance(historyI, AHistory):
               raise ValueError("Argument histories don't contain AHistory.")

        if type(evaluations) is not list:
            raise ValueError("Argument evaluations isn't type list.")
        for evaluationI in evaluations:
            if not isinstance(evaluationI, dict):
               raise ValueError("Argument evaluations don't contain dict.")

        if type(currentItemID) is not int and type(currentItemID) is not np.int64:
            raise ValueError("Argument currentIndex isn't type int.")

        uProbOfObservGenerated:List[float] = UserBehaviourSimulator().simulateStaticProb(self._uBehaviourDesc, self._numberOfItems)
        #print("uProbOfObservGenerated: " + str(uProbOfObservGenerated))

        uObservation:List[bool] = list(map(lambda x, y: x > y, uProbOfObservGenerated, np.random.uniform(low=0.0, high=1.0, size=self._numberOfItems)))
        #print("uObservation: " + str(uObservation))

        portfolioI:Portfolio1Aggr
        portFolioModelI:pd.DataFrame
        historyI:pd.DataFrame
        for portfolioI, portFolioModelI, evaluatonToolI, historyI, evaluationI in zip(portfolios, portFolioModels, evaluatonTools, histories, evaluations):
            self.__simulateRecommendation(portfolioI, portFolioModelI, evaluatonToolI, testRatingsDF, historyI,
                                          evaluationI, uProbOfObservGenerated, uObservation, currentItemID, nextItemIDs, userID)

    def __simulateRecommendation(self, portfolio:Portfolio1Aggr, portfolioModel:pd.DataFrame, evaluatonTool:AEvalTool,
                                 testRatingsDF:DataFrame, history:AHistory, evaluation:dict, uProbOfObserv:List[float],
                                 uObservation:List[bool], currentItemID:int, nextItemIDs:List[int], userID:int):

        rItemIDs:List[int]
        rItemIDsWithResponsibility:List[tuple[int, Series[int, str]]]
        rItemIDs, rItemIDsWithResponsibility = portfolio.recommend(
            userID, portfolioModel, numberOfItems=self._numberOfItems)

        nextNoClickedItemIDs:List[int] = [nItemIdI for nItemIdI in nextItemIDs if not history.isObjectClicked(userID, nItemIdI)]

        candidatesToClick:List[int] = list(set(nextNoClickedItemIDs) & set(rItemIDs))
        recommendedRelevantItem:bool = len(candidatesToClick) != 0

        selectedCandidateItemID:int = None
        wasCandidateObserved:bool = False
        probOfObserv:float = -1
        if recommendedRelevantItem:
            selectedCandidateItemID = candidatesToClick[0]  # take the first candidate
            index:int = rItemIDs.index(selectedCandidateItemID)
            wasCandidateObserved = uObservation[index]
            probOfObserv = uProbOfObserv[index]

        evaluatonTool.displayed(rItemIDsWithResponsibility, portfolioModel, evaluation)

        if recommendedRelevantItem and wasCandidateObserved:
            evaluatonTool.click(rItemIDsWithResponsibility, selectedCandidateItemID, probOfObserv, portfolioModel, evaluation)

        # save log of history
        history.insertRecommendations(userID, rItemIDs, uProbOfObserv, selectedCandidateItemID)







class Item:
    # next:Intem
    def __init__(self, itemID: int, indexDF: int, next):
        self._itemID: int = itemID
        self._indexDF: int = indexDF
        self._next: Item = next

    def getIndexDF(self):
        return self._indexDF

    # next:Item
    def getNext(self):
        return self._next

    # next:Item
    def setNext(self, next):
        self._next = next


class ModelOfIndexes:
    def __init__(self, ratingsDF:DataFrame):
        if type(ratingsDF) is not DataFrame:
            raise ValueError("Argument ratingsDF isn't type DataFrame.")

        ratingsCopyDF:DataFrame = ratingsDF.copy()
        ratingsCopyDF['index1'] = ratingsCopyDF.index

        #print(ratingsDF)
        #print(ratingsCopyDF)

        userIds:List[int] = list(set([rowI[Ratings.COL_USERID] for indexDFI, rowI in ratingsCopyDF.iterrows()]))

        # dictionary (index = userID, value = list[tuple(int, int)])
        # each list contains pair(int,int) or (itemID, indefOfDataFrame)
        self._dictionaryOfUserIDs:dict[List[tuple(int, int)]] = {}

        userIdI:int
        for userIdI in userIds:
            # select ratings of userIdI
            ratingsUserIDF:DataFrame = ratingsCopyDF.loc[ratingsCopyDF[Ratings.COL_USERID] == userIdI]

            userDictI:dict = {}
            lastItemI:Item = None

            indexDFI:int
            rowI:Series
            for i, rowI in ratingsUserIDF.iterrows():

                indexDFI:int = rowI['index1']
                userIdI:int = rowI[Ratings.COL_USERID]
                itemIdI:int = rowI[Ratings.COL_MOVIEID]

                itemI:Item = Item(userIdI, indexDFI, None)
                if not lastItemI is None:
                    lastItemI.setNext(itemI)

                lastItemI = itemI
                userDictI[itemIdI] = itemI

            self._dictionaryOfUserIDs[userIdI] = userDictI


    def getNextIndex(self, userId:int, itemId:int):
        u:dict[Item] = self._dictionaryOfUserIDs[userId]
        item:Item = u[itemId]

        itemNext:Item = item.getNext()
        if itemNext is None:
            return None

        return itemNext.getIndexDF()
