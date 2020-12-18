#!/usr/bin/python3

from typing import List

import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pandas.core.series import Series #class

from configuration.configuration import Configuration #class

from portfolio.aPortfolio import APortfolio #class
from portfolio.portfolio1Aggr import Portfolio1Aggr #class

from history.aHistory import AHistory #class
from evaluationTool.aEvalTool import AEvalTool #class

from datasets.aDataset import ADataset #class
from datasets.ml.behaviours import Behaviours #class

from pandas.core.frame import DataFrame #class

from history.aHistory import AHistory #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class


class ASequentialSimulation(ABC):


    ARG_WINDOW_SIZE:str = "windowSize"
    ARG_REPETITION_OF_RECOMMENDATION:str = "repetitionOfRecommendation"
    ARG_NUMBER_OF_RECOMM_ITEMS:str = "numberOfRecomItems"
    ARG_NUMBER_OF_AGGR_ITEMS:str = "numberOfAggrItems"

    ARG_DIV_DATASET_PERC_SIZE:str = "divisionDatasetPercentualSizes"
    AGR_USER_BEHAVIOUR_DFINDEX:str = "userBehaviourDFIndex"

    def __init__(self, batchID:str, dataset:ADataset, behaviourDF:DataFrame, argumentsDict:dict):

        if type(batchID) is not str:
            raise ValueError("Argument batchID isn't type str.")
        if not isinstance(dataset, ADataset):
            raise ValueError("Argument dataset isn't type ADataset.")
        if type(behaviourDF) is not DataFrame:
            raise ValueError("Argument behaviourDF isn't type DataFrame.")

        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        self._batchID:str = batchID

        self._dataset:ADataset = dataset
        self._behaviourDF:DataFrame = behaviourDF
        #self._behaviourClass = None

        self._windowSize:int = argumentsDict[self.ARG_WINDOW_SIZE]
        self._repetitionOfRecommendation:int = argumentsDict[self.ARG_REPETITION_OF_RECOMMENDATION]
        self._numberOfRecommItems:int = argumentsDict[self.ARG_NUMBER_OF_RECOMM_ITEMS]
        self._numberOfAggrItems:int = argumentsDict[self.ARG_NUMBER_OF_AGGR_ITEMS]

        self._divisionDatasetPercentualSize:int = argumentsDict[self.ARG_DIV_DATASET_PERC_SIZE]
        #self._uBehaviourDFIndex:str = argumentsDict[self.AGR_USER_BEHAVIOUR_DFINDEX]


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
        dir:str = Configuration.resultsDirectory + os.sep + self._batchID
        if not os.path.isdir(dir):
            os.mkdir(dir)

        computationFileName:str = dir + os.sep + "computation-" + portfolioDescI.getPortfolioID() +".txt"
        if os.path.isfile(computationFileName):
            raise ValueError("Results directory contains old results.")

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
        self._clickedItems:dict[List[int]] = self.initEvaluationModel(self._dataset)

        # dataset division setting
        #divisionDatasetPercentualSizes:List[int] = [50, 60, 70, 80, 90]
        divisionDatasetPercentualSizes:List[int] = [self._divisionDatasetPercentualSize]
        testDatasetPercentualSize:int = 10

        # dataset division
        percentualSizeI:int
        for percentualSizeI in divisionDatasetPercentualSizes:

            trainDataset:ADataset
            testRatingsDF:DataFrame
            trainDataset, testRatingsDF = self.divideDataset(
                self._dataset, percentualSizeI, testDatasetPercentualSize)

            evaluationI = self.runPortfolioDesc(portfolioDescs, portFolioModels, evaluatonTools, histories, trainDataset, testRatingsDF)
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
        evalFileName:str = Configuration.resultsDirectory + os.sep + self._batchID + os.sep + "evaluation.txt"
        evalFile = open(evalFileName, "a")
        evalFile.write("ids: " + str([portDescI.getPortfolioID() for portDescI in portfolioDescs]) + "\n")
        evalFile.write(str(evaluations) + "\n\n")
        evalFile.close()

        return evaluations


    def runPortfolioDesc(self, portfolioDescs:List[APortfolioDescription], portFolioModels:List[DataFrame],
                           evaluatonTools:List[AEvalTool], histories:List[AHistory], trainDataset:ADataset, testRatingsDF:DataFrame):

        portfolios:List[APortfolio] = []

        portfolioDescI:APortfolioDescription
        historyI:AHistory
        for portfolioDescI, historyI in zip(portfolioDescs, histories):

            print("Training mode: " + str(portfolioDescI.getPortfolioID()))

            # train portfolio model
            portfolioI:APortfolio = portfolioDescI.exportPortfolio(self._batchID, historyI)
            portfolioI.train(historyI, trainDataset)
            portfolios.append(portfolioI)

        return self.iterateOverDataset(portfolios, portfolioDescs, portFolioModels, evaluatonTools, histories, testRatingsDF)



    def simulateRecommendations(self, portfolios:List[APortfolio], portfolioDescs:List[APortfolioDescription],
                                  portFolioModels:List[DataFrame], evaluatonTools:List[AEvalTool], histories:List[AHistory],
                                  evaluations:List[dict], currentItemID:int, nextItemIDs:List[int], userID:int, repetition:int):

        print("userID: " + str(userID))
        print("currentItemID: " + str(currentItemID))
        print("repetition: " + str(repetition))

        COL_USERID:str = self._behaviourClass.getColNameUserID()
        COL_ITEMID:str = self._behaviourClass.getColNameItemID()
        COL_REPETITION:str = self._behaviourClass.getColNameRepetition()
        COL_BEHAVIOUR:str = self._behaviourClass.getColNameBehaviour()

        isUser:List[bool] = self._behaviourDF[COL_USERID] == userID
        isItem:List[bool] = self._behaviourDF[COL_ITEMID] == currentItemID
        isRepetition:List[bool] = self._behaviourDF[COL_REPETITION] == repetition

        #print(self._behaviourDF.head(10))
        uObservationUserItem:str = self._behaviourDF[(isUser) & (isItem) & (isRepetition)][COL_BEHAVIOUR]

        if uObservationUserItem.shape[0] != 1:
            print(uObservationUserItem)
            raise ValueError("Error")
        uObservationUserItemStr:str = uObservationUserItem.iloc[0]
        #print(uObservationUserItemStr)
        uObservation:List[bool] = Behaviours.convertToListOfBoolean(uObservationUserItemStr)

        print("uObservation: " + str(uObservation))

        portfolioI:Portfolio1Aggr
        portFolioModelI:pd.DataFrame
        historyI:pd.DataFrame
        for portfolioI, portfolioDescI, portFolioModelI, evaluatonToolI, historyI, evaluationI in zip(
                portfolios, portfolioDescs, portFolioModels, evaluatonTools, histories, evaluations):

            self.simulateRecommendation(portfolioI, portfolioDescI, portFolioModelI, evaluatonToolI, historyI,
                                          evaluationI, uObservation, currentItemID, nextItemIDs, userID)


    def simulateRecommendation(self, portfolio:APortfolio, portfolioDesc:APortfolioDescription, portfolioModel:pd.DataFrame,
                                 evaluatonTool:AEvalTool, history:AHistory, evaluation:dict, uObservation:List[bool],
                                 currentItemID:int, nextItemIDs:List[int], userID:int):

        #print("userID: " + str(userID))
        portId:str = portfolioDesc.getPortfolioID()

        args:dict = {APortfolio.ARG_NUMBER_OF_RECOMM_ITEMS:self._numberOfRecommItems, Portfolio1Aggr.ARG_NUMBER_OF_AGGR_ITEMS:self._numberOfAggrItems}

        rItemIDs:List[int]
        rItemIDsWithResponsibility:List[tuple[int, Series[int, str]]]
        rItemIDs, rItemIDsWithResponsibility = portfolio.recommend(userID, portfolioModel, args)

        evaluatonTool.displayed(rItemIDsWithResponsibility, portfolioModel, evaluation)

        nextNoClickedItemIDs:List[int] = list(set(nextItemIDs) -set(self._clickedItems[userID]))

        candidatesToClick:List[int] = list(set(rItemIDs) & set(nextNoClickedItemIDs))
        clickedItemIDs:List[int] = []
        for candidateToClickI in candidatesToClick:
            indexI:int = rItemIDs.index(candidateToClickI)
            wasCandidateObservedI:bool = uObservation[indexI]
            if wasCandidateObservedI:
                clickedItemIDs.append(candidateToClickI)

        print("nextItemIDs: " + str(nextItemIDs))
        print("rItemIDs: " + str(rItemIDs))
        print("uObservation: " + str(uObservation))
        print("candidatesToClick: " + str(candidatesToClick))
        print("clickedItemIDs: " + str(clickedItemIDs))


        for clickedItemIdI in clickedItemIDs:
            evaluatonTool.click(rItemIDsWithResponsibility, clickedItemIdI, portfolioModel, evaluation)

            if not clickedItemIdI in self._clickedItems[userID]:
                self._clickedItems[userID].append(clickedItemIdI)

            print("clickedItems: " + str(self._clickedItems[userID]))

        # store port model time evolution to file
        self.portModelTimeEvolutionFiles[portId].write("currentItemID: " + str(currentItemID) + "\n")
        self.portModelTimeEvolutionFiles[portId].write("userID: " + str(userID) + "\n")
        self.portModelTimeEvolutionFiles[portId].write(str(portfolioModel) + "\n\n")

        # store history of recommendations to file
        self.historyOfRecommendationFiles[portfolioDesc.getPortfolioID()].write("userID: " + str(userID) + "\n")
        self.historyOfRecommendationFiles[portfolioDesc.getPortfolioID()].write("currentItemID: " + str(currentItemID) + "\n")
        self.historyOfRecommendationFiles[portfolioDesc.getPortfolioID()].write("rItemIDs: " + str(rItemIDs) + "\n")
        self.historyOfRecommendationFiles[portfolioDesc.getPortfolioID()].write("uObservation: " + str(uObservation) + "\n")
        self.historyOfRecommendationFiles[portfolioDesc.getPortfolioID()].write("clickedItemIDs: " + str(clickedItemIDs) + "\n\n")

        # save log of history
        history.insertRecomAndClickedItemIDs(userID, rItemIDs, clickedItemIDs)

        # delete log of history
        history.deletePreviousRecomOfUser(userID, self._repetitionOfRecommendation * self._numberOfRecommItems)



    @abstractmethod
    def divideDataset(dataset:ADataset, divisionDatasetPercentualSize:int, testDatasetPercentualSize:int):
        assert False, "this needs to be overridden"

    @abstractmethod
    def initEvaluationModel(self, dataset:ADataset):
        assert False, "this needs to be overridden"

    @abstractmethod
    def iterateOverDataset(self, portfolios:List[APortfolio], portfolioDescs:List[APortfolioDescription],
                             portFolioModels:List[pd.DataFrame], evaluatonTools:List[AEvalTool], histories:List[AHistory],
                             testRatingsDF:DataFrame):
        assert False, "this needs to be overridden"

    @abstractmethod
    def getNextItemIDs(self, model, userId:int, itemId:int, ratingsDF:DataFrame, windowSize:int):
        assert False, "this needs to be overridden"

