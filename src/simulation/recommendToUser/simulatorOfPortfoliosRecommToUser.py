#!/usr/bin/python3

from typing import List

from pandas.core.series import Series #class

from configuration.configuration import Configuration #class

from datasets.ratings import Ratings #class
from datasets.users import Users #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class
from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class
from portfolio.portfolio1Aggr import Portfolio1Aggr #class

from portfolio.aPortfolio import APortfolio #class

import os
import pandas as pd
import numpy as np

from pandas.core.frame import DataFrame #class

from history.aHistory import AHistory #class
from evaluationTool.aEvalTool import AEvalTool #class

from datasets.behaviours import Behaviours #class

from simulation.tools.userBehaviourSimulator import UserBehaviourSimulator #class
from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class


class SimulationPortfolioToUser:

    ARG_WINDOW_SIZE:str = "windowSize"
    ARG_REPETITION_OF_RECOMMENDATION:str = "repetitionOfRecommendation"
    ARG_NUMBER_OF_RECOMM_ITEMS:str = "numberOfRecomItems"
    ARG_NUMBER_OF_AGGR_ITEMS:str = "numberOfAggrItems"

    ARG_DIV_DATASET_PERC_SIZE = "divisionDatasetPercentualSizes"

    def __init__(self, jobID:str, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame,
                 behaviourDF:DataFrame, argumentsDict:dict):

        if type(jobID) is not str:
            raise ValueError("Argument jobID isn't type str.")
        if type(ratingsDF) is not DataFrame:
            raise ValueError("Argument ratingsDF isn't type DataFrame.")
        if type(usersDF) is not DataFrame:
            raise ValueError("Argument usersDF isn't type DataFrame.")
        if type(itemsDF) is not DataFrame:
            raise ValueError("Argument itemsDF isn't type DataFrame.")
        if type(behaviourDF) is not DataFrame:
            raise ValueError("Argument behaviourDF isn't type DataFrame.")

        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        self._jobID:str = jobID

        self._ratingsDF:DataFrame = ratingsDF
        self._usersDF:DataFrame = usersDF
        self._itemsDF:DataFrame = itemsDF
        self._behaviourDF:DataFrame = behaviourDF

        self._windowSize:int = argumentsDict[self.ARG_WINDOW_SIZE]
        self._repetitionOfRecommendation:int = argumentsDict[self.ARG_REPETITION_OF_RECOMMENDATION]
        self._numberOfRecommItems:int = argumentsDict[self.ARG_NUMBER_OF_RECOMM_ITEMS]
        self._numberOfAggrItems:int = argumentsDict[self.ARG_NUMBER_OF_AGGR_ITEMS]

        self._divisionDatasetPercentualSize:int = argumentsDict[self.ARG_DIV_DATASET_PERC_SIZE]

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

        # create directory for results
        dir:str = Configuration.resultsDirectory + os.sep + self._jobID
        if os.path.isdir(dir):
            raise ValueError("Directory results contains old results \'" + str(self._jobID) +"\'")
        os.mkdir(dir)

        computationFileName:str = dir + os.sep + "computation.txt"
        self.computationFile = open(computationFileName, "w+")

        # opening files for portfolio model time evolution
        self.portModelTimeEvolutionFiles = {}
        for portfolioDescI in portfolioDescs:
            fileName:str = dir + os.sep + "portfModelTimeEvolution-" + portfolioDescI.getPortfolioID() + ".txt"
            self.portModelTimeEvolutionFiles[portfolioDescI.getPortfolioID()] = open(fileName, "a")

        # opening files for portfolio model time evolution
        self.historyOfRecommendationFiles:dict = {}
        for portfolioDescI in portfolioDescs:
            fileName:str = dir + os.sep + "historyOfRecommendation-" + portfolioDescI.getPortfolioID() + ".txt"
            self.historyOfRecommendationFiles[portfolioDescI.getPortfolioID()] = open(fileName, "a")

        # results of portfolios evaluations
        evaluations:List[int] = []

        # model with clicked ItemIDs for users
        self._clickedItems:dict[List[int]] = {}
        for indexI, rowI in self._usersDF.iterrows():
            self._clickedItems[rowI[Ratings.COL_USERID]] = []

        # dataset division setting
        #divisionDatasetPercentualSizes:List[int] = [50, 60, 70, 80, 90]
        divisionDatasetPercentualSizes:List[int] = [self._divisionDatasetPercentualSize]
        testDatasetPercentualSize:int = 10

        ratingsSortedDF:DataFrame = self._ratingsDF.sort_values(by=Ratings.COL_TIMESTAMP)
        numberOfRatings:int = ratingsSortedDF.shape[0]


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

        # closing files
        self.computationFile.close()

        #hOfModelDictI:File
        for hOfModelDictI in self.portModelTimeEvolutionFiles.values():
            hOfModelDictI.close()

        #hOfRecommDictI:File
        for hOfRecommDictI in self.historyOfRecommendationFiles.values():
            hOfRecommDictI.close()

        #evalFileName:File
        evalFileName:str = Configuration.resultsDirectory + os.sep + self._id + os.sep + "evaluation.txt"
        evalFile = open(evalFileName, "a")
        evalFile.write("ids: " + str([portDescI.getPortfolioID() for portDescI in portfolioDescs]) + "\n")
        evalFile.write(str(evaluations))
        evalFile.close()

        return evaluations


    def __runPortfolioDesc(self, portfolioDescs:List[Portfolio1AggrDescription], portFolioModels:List[DataFrame],
                           evaluatonTools:[AEvalTool], histories:List[AHistory], trainRatingsDF:DataFrame, testRatingsDF:DataFrame):

        #print("Deletion the first 20 ratings of each user from tesing part of ratings")
        ## deletion the first 20 ratings of each user from ratings
        #ratingsDel:List[DataFrame] = []
        #
        #userIDs:List[int] = list(self._usersDF[Users.COL_USERID])
        #for userIdI in userIDs:
        #    ratingsOfUserIDF:DataFrame = testRatingsDF.loc[testRatingsDF[Ratings.COL_USERID] == userIdI]
        #    ratingsOfUserSortedIDF:DataFrame = ratingsOfUserIDF.sort_values(by=Ratings.COL_TIMESTAMP)
        #    ratingsDel.append(ratingsOfUserSortedIDF.iloc[20:])
        #
        #testRatingsDF = pd.concat(ratingsDel)
        #testRatingsDF:DataFrame = testRatingsDF.sort_values(by=Ratings.COL_TIMESTAMP)



        portfolios:List[Portfolio1Aggr] = []

        portfolioDescI:Portfolio1AggrDescription
        historyI:AHistory
        for portfolioDescI, historyI in zip(portfolioDescs, histories):

            print("Training mode: " + str(portfolioDescI.getPortfolioID()))

            # train portfolio model
            portfolioI:Portfolio1Aggr = portfolioDescI.exportPortfolio(self._jobID, historyI)
            portfolioI.train(historyI, trainRatingsDF.copy(), self._usersDF.copy(), self._itemsDF.copy())
            portfolios.append(portfolioI)

        return self.__iterateOverDataset(portfolios, portfolioDescs, portFolioModels, evaluatonTools, histories, testRatingsDF)


    def __iterateOverDataset(self, portfolios:List[APortfolio], portfolioDescs:List[Portfolio1AggrDescription],
                             portFolioModels:List[pd.DataFrame], evaluatonTools:[AEvalTool], histories:List[AHistory],
                             testRatingsDF:DataFrame):

        model:ModelOfIndexes = ModelOfIndexes(testRatingsDF)

        portIds:List[str] = [portDescI.getPortfolioID() for portDescI in portfolioDescs]

        evaluations:List[dict] = [{} for i in range(len(portfolios))]

        counterI:int = 0

        currentIndexDFI:int
        nextIndexDFI:int
        for currentIndexDFI in list(testRatingsDF.index):

            counterI += 1
            if counterI  % 100 == 0:
                print("RatingI: " + str(counterI) + " / " + str(testRatingsDF.shape[0]))
                print("PortfolioIds: " + str(portIds))
                print("Evaluations: " + str(evaluations))

                self.computationFile.write("RatingI: " + str(counterI) + " / " + str(testRatingsDF.shape[0]) + "\n")
                self.computationFile.write("PortfolioIds: " + str(portIds) + "\n")
                self.computationFile.write("Evaluations: " + str(evaluations) + "\n")
                self.computationFile.flush()

#            if counterI % 1000 == 0:
#                for historyI in histories:
#                    historyI.delete(self._repetitionOfRecommendation * self._numberOfItems)

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
                self.__simulateRecommendations(portfolios, portfolioDescs, portFolioModels, evaluatonTools,
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

    def __simulateRecommendations(self, portfolios:List[APortfolio], portfolioDescs:List[Portfolio1AggrDescription],
                                  portFolioModels:List[DataFrame], evaluatonTools:[AEvalTool], histories:List[AHistory],
                                  evaluations:List[dict], currentItemID:int, nextItemIDs:List[int], userID:int):

        #print("userID: " + str(userID))
        #print("currentItemID: " + str(currentItemID))

        isUser:List[bool] = self._behaviourDF[Behaviours.COL_USERID] == userID
        isItem:List[bool] = self._behaviourDF[Behaviours.COL_MOVIEID] == currentItemID
        uObservation:List[bool] = self._behaviourDF[(isUser) & (isItem)][Behaviours.COL_LINEAR0109].iloc[0]
        #print(uObservation)

        portfolioI:Portfolio1Aggr
        portFolioModelI:pd.DataFrame
        historyI:pd.DataFrame
        for portfolioI, portfolioDescI, portFolioModelI, evaluatonToolI, historyI, evaluationI in zip(portfolios, portfolioDescs, portFolioModels, evaluatonTools, histories, evaluations):
            self.__simulateRecommendation(portfolioI, portfolioDescI, portFolioModelI, evaluatonToolI, historyI,
                                          evaluationI, uObservation, currentItemID, nextItemIDs, userID)

    def __simulateRecommendation(self, portfolio:APortfolio, portfolioDesc:Portfolio1AggrDescription, portfolioModel:pd.DataFrame,
                                 evaluatonTool:AEvalTool, history:AHistory, evaluation:dict, uObservation:List[bool],
                                 currentItemID:int, nextItemIDs:List[int], userID:int):

        #print("userID: " + str(userID))
        portId:str = portfolioDesc.getPortfolioID()

        args:dict = {APortfolio.ARG_NUMBER_OF_RECOMM_ITEMS:self._numberOfRecommItems, Portfolio1Aggr.ARG_NUMBER_OF_AGGR_ITEMS:self._numberOfAggrItems}

        rItemIDs:List[int]
        rItemIDsWithResponsibility:List[tuple[int, Series[int, str]]]
        rItemIDs, rItemIDsWithResponsibility = portfolio.recommend(userID, portfolioModel, args)

        evaluatonTool.displayed(rItemIDsWithResponsibility, portfolioModel, evaluation)


        #nextNoClickedItemIDs:List[int] = [nItemIdI for nItemIdI in nextItemIDs if not history.isObjectClicked(userID, nItemIdI)]
        nextNoClickedItemIDs:List[int] = list(set(nextItemIDs) -set(self._clickedItems[userID]))

        candidatesToClick:List[int] = list(set(rItemIDs) & set(nextNoClickedItemIDs))
        clickedItemIDs:List[int] = []
        for candidateToClickI in candidatesToClick:
            indexI:int = rItemIDs.index(candidateToClickI)
            wasCandidateObservedI:bool = uObservation[indexI]
            if wasCandidateObservedI:
                clickedItemIDs.append(candidateToClickI)


        for clickedItemIdI in clickedItemIDs:
            evaluatonTool.click(rItemIDsWithResponsibility, candidateToClickI, portfolioModel, evaluation)

            if not clickedItemIdI in self._clickedItems[userID]:
                self._clickedItems[userID].append(clickedItemIdI)

            print("clickedItems: " + str(self._clickedItems[userID]))

            self.portModelTimeEvolutionFiles[portId].write("currentItemID: " + str(currentItemID) + "\n")
            self.portModelTimeEvolutionFiles[portId].write(str(portfolioModel) + "\n\n")

        # store history of recommendations to file
        self.historyOfRecommendationFiles[portfolioDesc.getPortfolioID()].write(str(userID) + "\n")
        self.historyOfRecommendationFiles[portfolioDesc.getPortfolioID()].write(str(rItemIDs) + "\n")
        self.historyOfRecommendationFiles[portfolioDesc.getPortfolioID()].write(str(clickedItemIDs) + "\n\n")

        # save log of history
        history.insertRecomAndClickedItemIDs(userID, rItemIDs, clickedItemIDs)

        # delete log of history
        history.deletePreviousRecomOfUser(userID, self._repetitionOfRecommendation * self._numberOfRecommItems)






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
