#!/usr/bin/python3

from typing import List
from typing import Dict

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
from evaluationTool.evalToolContext import EvalToolContext #class

from datasets.aDataset import ADataset #class
from datasets.ml.behavioursML import BehavioursML #class

from pandas.core.frame import DataFrame #class

from history.aHistory import AHistory #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.evalToolContext import EvalToolContext #class

class ASequentialSimulation(ABC):


    ARG_WINDOW_SIZE:str = "windowSize"
    ARG_RECOM_REPETITION_COUNT:str = "recomRepetitionCount"
    ARG_NUMBER_OF_RECOMM_ITEMS:str = "numberOfRecomItems"
    ARG_NUMBER_OF_AGGR_ITEMS:str = "numberOfAggrItems"

    ARG_DIV_DATASET_PERC_SIZE:str = "divisionDatasetPercentualSizes"
    AGR_USER_BEHAVIOUR_DFINDEX:str = "userBehaviourDFIndex"

    ARG_HISTORY_LENGTH:str = "historyLength"

    ARG_MODE_PROTECT_OLD_RESULTS:str = "protectOldResults"
    ARG_MODE_OVERWRITE_OLD_RESULTS:str = "overwriteOldResults"
    ARG_MODE_OVERWRITE_EMPTY_RESULTS:str = "overwriteEmptyResults"


    def __init__(self, batchID:str, dataset:ADataset, behaviourDF:DataFrame, argumentsDict:dict):

        if type(batchID) is not str:
            raise ValueError("Argument batchID isn't type str.")
        if not isinstance(dataset, ADataset):
            raise ValueError("Argument dataset isn't type ADataset.")
        if type(behaviourDF) is not DataFrame:
            raise ValueError("Argument behaviourDF isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")
        if 1 != int(argumentsDict.get(self.ARG_MODE_PROTECT_OLD_RESULTS, False)) +\
            int(argumentsDict.get(self.ARG_MODE_OVERWRITE_OLD_RESULTS, False)) +\
            int(argumentsDict.get(self.ARG_MODE_OVERWRITE_EMPTY_RESULTS, False)):
            raise ValueError("Argument argumentsDict doesn't contain mode.")


        self._batchID:str = batchID

        self._dataset:ADataset = dataset
        self._behaviourDF:DataFrame = behaviourDF
        #self._behaviourClass = None

        self._windowSize:int = argumentsDict[self.ARG_WINDOW_SIZE]
        self._recomRepetitionCount:int = argumentsDict[self.ARG_RECOM_REPETITION_COUNT]
        self._numberOfRecommItems:int = argumentsDict[self.ARG_NUMBER_OF_RECOMM_ITEMS]
        self._numberOfAggrItems:int = argumentsDict[self.ARG_NUMBER_OF_AGGR_ITEMS]

        self._divisionDatasetPercentualSize:int = argumentsDict[self.ARG_DIV_DATASET_PERC_SIZE]

        self._historyLength:int = argumentsDict[self.ARG_HISTORY_LENGTH]


        self._modeProtectOldResults:bool = argumentsDict.get(self.ARG_MODE_PROTECT_OLD_RESULTS, False)
        self._modeOwerwriteOldResults:bool = argumentsDict.get(self.ARG_MODE_OVERWRITE_OLD_RESULTS, False)
        self._modeOverwriteEmptyResults:bool = argumentsDict.get(self.ARG_MODE_OVERWRITE_EMPTY_RESULTS, False)

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

        if self._modeProtectOldResults and os.path.isfile(computationFileName):
            raise ValueError("Results directory contains old results.")

        if self._modeOverwriteEmptyResults and os.path.isfile(computationFileName) and\
                os.path.getsize(computationFileName) != 0:
            raise ValueError("Results directory contains old non-empty results.")

        # remove files
        if os.path.exists(computationFileName):
            os.remove(computationFileName)
        self.computationFile = open(computationFileName, "w+")

        # opening files for portfolio model time evolution
        self.portModelTimeEvolutionFiles = {}
        for portfolioDescI in portfolioDescs:
            fileName:str = dir + os.sep + "portfModelTimeEvolution-" + portfolioDescI.getPortfolioID() + ".txt"
            if os.path.exists(fileName):
                os.remove(fileName)
            self.portModelTimeEvolutionFiles[portfolioDescI.getPortfolioID()] = open(fileName, "a")

        # opening files for portfolio model time evolution
        self.historyOfRecommendationFiles:dict = {}
        for portfolioDescI in portfolioDescs:
            fileName:str = dir + os.sep + "historyOfRecommendation-" + portfolioDescI.getPortfolioID() + ".txt"
            if os.path.exists(fileName):
                os.remove(fileName)
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
            testRelevantRatingsDF:DataFrame
            testBehaviourDict:dict[DataFrame]
            trainDataset, testRatingsDF, testRelevantRatingsDF, testBehaviourDict = self.divideDataset(
                self._dataset, self._behaviourDF, percentualSizeI, testDatasetPercentualSize, self._recomRepetitionCount)

            evaluationI = self.runPortfolioDesc(portfolioDescs, portFolioModels, evaluatonTools,
                                                histories, trainDataset, testRatingsDF, testRelevantRatingsDF, testBehaviourDict)
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
                           evaluatonTools:List[AEvalTool], histories:List[AHistory], trainDataset:ADataset,
                           testRatingsDF:DataFrame, testRelevantRatingsDF:DataFrame, testBehaviourDict:Dict[int, DataFrame]):

        portfolios:List[APortfolio] = []

        portfolioDescI:APortfolioDescription
        historyI:AHistory
        for portfolioDescI, historyI in zip(portfolioDescs, histories):

            print("Training mode: " + str(portfolioDescI.getPortfolioID()))

            # train portfolio model
            portfolioI:APortfolio = portfolioDescI.exportPortfolio(self._batchID, historyI)
            portfolioI.train(historyI, trainDataset)
            portfolios.append(portfolioI)

        return self.iterateOverDataset(portfolios, portfolioDescs, portFolioModels, evaluatonTools,
                            histories, testRatingsDF, testRelevantRatingsDF, testBehaviourDict)



    def simulateRecommendations(self, portfolios:List[APortfolio], portfolioDescs:List[APortfolioDescription],
                                  portFolioModels:List[DataFrame], evaluatonTools:List[AEvalTool], histories:List[AHistory],
                                  evaluations:List[dict], currentDFIndex:int, userID:int, sessionID:int, repetition:int,
                                  testRatingsDF:DataFrame, testBehaviourDict:Dict[int, DataFrame], windowOfItemIDsI:List[int],
                                  currentPageType:object):

        COL_BEHAVIOUR:str = self._behaviourClass.getColNameBehaviour()
        COL_ITEMID:str = self._ratingClass.getColNameItemID()


        currentItemID:int = testRatingsDF.loc[currentDFIndex][COL_ITEMID]

        print("userID: " + str(userID))
        print("currentDFIndex: " + str(currentDFIndex))
        print("currentItemID: " + str(currentItemID))
        print("repetition: " + str(repetition))

        uObservationStrI:str = testBehaviourDict[repetition].loc[currentDFIndex][COL_BEHAVIOUR]
        uObservation:List[bool] = BehavioursML.convertToListOfBoolean(uObservationStrI)
        #uObservation: List[bool] = [True]*20

        print("uObservation: " + str(uObservation))

        portfolioI:Portfolio1Aggr
        portFolioModelI:pd.DataFrame
        historyI:pd.DataFrame
        for portfolioI, portfolioDescI, portFolioModelI, evaluatonToolI, historyI, evaluationI in zip(
                portfolios, portfolioDescs, portFolioModels, evaluatonTools, histories, evaluations):

            self.simulateRecommendation(portfolioI, portfolioDescI, portFolioModelI, evaluatonToolI, historyI,
                                        evaluationI, currentDFIndex, testRatingsDF, uObservation, userID,
                                        sessionID, windowOfItemIDsI, currentPageType)


    def simulateRecommendation(self, portfolio:APortfolio, portfolioDesc:APortfolioDescription, portfolioModel:DataFrame,
                                 evaluatonTool:AEvalTool, history:AHistory, evaluationDict:Dict[str,object],
                                 currentDFIndex:int, testRatingsDF:DataFrame, uObservation:List[bool], userID:int,
                                 sessionID:int, windowOfItemIDsI:List[int], currentPageType:object):

        from datasets.slantour.events import Events #class
        COL_ITEMID:str = self._ratingClass.getColNameItemID()
        currentItemID:int = testRatingsDF.loc[currentDFIndex][COL_ITEMID]

        print("userID: " + str(userID))
        portId:str = portfolioDesc.getPortfolioID()

        args:Dict[str, object] = {APortfolio.ARG_NUMBER_OF_RECOMM_ITEMS:self._numberOfRecommItems,
                                  Portfolio1Aggr.ARG_NUMBER_OF_AGGR_ITEMS:self._numberOfAggrItems,
                                  EvalToolContext.ARG_PAGE_TYPE:currentPageType,
                                  EvalToolContext.ARG_USER_ID: userID,
                                  EvalToolContext.ARG_ITEM_ID:currentItemID,
                                  EvalToolContext.ARG_SENIORITY:sessionID}

        rItemIDs:List[int]
        rItemIDsWithResponsibility:List[tuple[int, Series[int, str]]]
        rItemIDs, rItemIDsWithResponsibility = portfolio.recommend(userID, portfolioModel, args)

        evaluatonTool.displayed(rItemIDsWithResponsibility, portfolioModel, args)

        candidatesToClick:List[int] = [itemIDI for itemIDI, observedI in zip(rItemIDs, uObservation[:len(rItemIDs)]) if observedI]
        clickedItemIDs:List[int] = list(set(candidatesToClick) & set(windowOfItemIDsI))
        clickedNewItemIDs:List[int] = [itemIdI for itemIdI in clickedItemIDs if itemIdI not in self._clickedItems[userID]]

        print("windowOfItemIDsI: " + str(windowOfItemIDsI))
        print("rItemIDs: " + str(rItemIDs))
        print("uObservation: " + str(uObservation))
        print("candidatesToClick: " + str(candidatesToClick))
        print("clickedItemIDs: " + str(clickedItemIDs))
        print("clickedNewItemIDs: " + str(clickedNewItemIDs))
        print("historyOfClickedItems: " + str(self._clickedItems[userID]))

        for clickedNewItemIdI in clickedNewItemIDs:
            evaluationDict[AEvalTool.CLICKS] = evaluationDict.get(AEvalTool.CLICKS, 0) + 1

            evaluatonTool.click(rItemIDsWithResponsibility, clickedNewItemIdI, portfolioModel, args)

            self._clickedItems[userID].append(clickedNewItemIdI)


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
        self.historyOfRecommendationFiles[portfolioDesc.getPortfolioID()].write("clickedNewItemIDs: " + str(clickedNewItemIDs) + "\n\n")


        # save log of history
        history.insertRecomAndClickedItemIDs(userID, rItemIDs, clickedItemIDs)

        # delete log of history
        lengthOfHistory:int = 10 * self._recomRepetitionCount * self._numberOfAggrItems
        #print("lengthOfHistory: " + str(lengthOfHistory))
        history.deletePreviousRecomOfUser(userID, lengthOfHistory)



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

