#!/usr/bin/python3

from pandas.core.frame import DataFrame #class
from pandas.core.frame import Series #class

import os
import json
from typing import List
from typing import Dict #class

from datasets.aDataset import ADataset #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from aggregation.aAggregation import AAgregation #class

from recommender.aRecommender import ARecommender #class

from history.aHistory import AHistory #class
from portfolio.aPortfolio import APortfolio #class

from portfolioModel.pModelDHondtPersonalisedStat import PModelDHondtPersonalisedStat #class
from portfolioModel.pModelDHondt import PModelDHondt #class

from configuration.configuration import Configuration #class

import pandas as pd
import numpy as np


class PortfolioDynamic(APortfolio):

    def __init__(self, batchID:str, portfolioID:str, recommender:ARecommender, recommID:str,
                 portfolioInternal:APortfolio, portfolioInternalID:str):
        if type(batchID) is not str:
            raise ValueError("Argument batchID isn't type str.")
        if type(portfolioID) is not str:
            raise ValueError("Argument portfolioID isn't type str.")
        if not isinstance(recommender, ARecommender):
            raise ValueError("Argument recommenders isn't type ARecommender.")
        if type(recommID) is not str:
            raise ValueError("Argument recommID isn't type str.")
        if not isinstance(portfolioInternal, APortfolio):
            print(type(portfolioInternal))
            raise ValueError("Argument agregation isn't type APortfolio.")
        if type(portfolioInternalID) is not str:
            raise ValueError("Argument portfolioInternalID isn't type str.")

        self._batchID:str = batchID
        self._portfolioID:str = portfolioID
        self._recommender:ARecommender = recommender
        self._recommID:str = recommID
        self._portfolioInternal:APortfolio = portfolioInternal
        self._portfolioInternalID:str = portfolioInternalID

    def getRecommIDs(self):
        return self._portfolioInternal.getRecommIDs()

    def train(self, history:AHistory, dataset:ADataset):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if not isinstance(dataset, ADataset):
            raise ValueError("Argument dataset isn't type ADataset.")

        self._portfolioInternal.train(history, dataset)

        self._recommender.train(history, dataset)

        dir:str = Configuration.resultsDirectory + os.sep + self._batchID
        logFileName:str = dir + os.sep + "log-portfolio1Aggr" + self._portfolioID + ".txt"

        #if self._modeProtectOldResults and os.path.isfile(logFileName):
        #    raise ValueError("Results directory contains old results.")

        if os.path.exists(logFileName):
            os.remove(logFileName)
        self._logFile = open(logFileName, "w+")



    def update(self, ratingsUpdateDF:DataFrame, argumentsDict:Dict[str,object]):
        if type(ratingsUpdateDF) is not DataFrame:
           raise ValueError("Argument ratingsUpdateDF isn't type DataFrame.")
        if type(argumentsDict) is not dict:
           raise ValueError("Argument argumentsDict isn't type dict.")

        self._portfolioInternal.update(ratingsUpdateDF, argumentsDict)

        self._recommender.update(ratingsUpdateDF, argumentsDict)


    # portFolioModel:DataFrame<(methodID, votes)>
    def recommend(self, userID:int, portFolioModel:DataFrame, argumentsDict:Dict[str,object]):
        #print("userID: " + str(userID))
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if not isinstance(portFolioModel, DataFrame):
            raise ValueError("Argument portFolioModel isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        currentItemID:int = argumentsDict[self.ARG_ITEM_ID]
        numberOfRecomItems:int = argumentsDict[self.ARG_NUMBER_OF_RECOMM_ITEMS]
        numberOfAggrItems:int = argumentsDict[self.ARG_NUMBER_OF_AGGR_ITEMS]

        arguments2Dict:Dict[str, object] = {}
        #arguments2Dict.update(self._recomDesc.getArguments())
        arguments2Dict.update(argumentsDict)

        # creates model if doesn't exist
        modelOfUserID:DataFrame = portFolioModel.getModel(userID)

        if portFolioModel.loc[userID, PModelDHondtPersonalisedStat.COL_CLICK_COUNT] < 1:
            print("ahoj franto")
            itemIDsWithResponsibility:Series = self._recommender.recommend(
                   userID, numberOfItems=numberOfRecomItems, argumentsDict=arguments2Dict)
            recomItemIDs:List[int] = list(int(itemIdI) for itemIdI in itemIDsWithResponsibility.index)
            a = PModelDHondt.getEqualResponsibilityForAll(recomItemIDs, self.getRecommIDs())
            print("recomItemIDs: " + str(recomItemIDs))
            print("aaa: " + str(a))
            return (recomItemIDs, a)
        print("cus franto")

        aggItemIDs, aggItemIDsWithResponsibility = self._portfolioInternal.recommend(
            userID, portFolioModel, argumentsDict)


        # Tuple
        return (aggItemIDs, aggItemIDsWithResponsibility)
