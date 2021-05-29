#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

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

from configuration.configuration import Configuration #class

import pandas as pd
import numpy as np


class Portfolio1Aggr(APortfolio):

    def __init__(self, batchID:str, portfolioID:str, recommenders:List[ARecommender], recommIDs:List[str], recomDescs:List[RecommenderDescription],
                agregation:AAgregation):
        if type(batchID) is not str:
            raise ValueError("Argument batchID isn't type str.")
        if type(portfolioID) is not str:
            raise ValueError("Argument portfolioID isn't type str.")
        if type(recommenders) is not list:
            raise ValueError("Argument recommenders isn't type list.")
        for recommenderI in recommenders:
            if not isinstance(recommenderI, ARecommender):
              raise ValueError("Argument recommenders don't contains type ARecommender.")

        if type(recomDescs) is not list:
            raise ValueError("Argument recomDescs isn't type list.")
        for recomDescI in recomDescs:
            if not type(recomDescI) is RecommenderDescription:
                raise ValueError("Argument recomDescs don't contains type RecommenderDescription.")

        if not isinstance(agregation, AAgregation):
            raise ValueError("Argument agregation isn't type AAgregation.")

        self._batchID:str = batchID
        self._portfolioID:str = portfolioID
        self._recommenders:List[ARecommender] = recommenders
        self._recomDescs:List[RecommenderDescription] = recomDescs
        self._recommIDs:List[str] = recommIDs
        self._aggregation:AAgregation = agregation

    def getRecommIDs(self):
        return self._recommIDs

    def train(self, history:AHistory, dataset:ADataset):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")

        self._aggregation.train(history, dataset)

        recommenderI:ARecommender
        for recommenderI in self._recommenders:
            recommenderI.train(history, dataset)

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

        self._aggregation.update(ratingsUpdateDF, argumentsDict)

        recommenderI:ARecommender
        for recommenderI in self._recommenders:
           recommenderI.update(ratingsUpdateDF, argumentsDict)


    # portFolioModel:DataFrame<(methodID, votes)>
    def recommend(self, userID:int, portFolioModel:DataFrame, argumentsDict:Dict[str,object]):
        #print("userID: " + str(userID))
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(portFolioModel) is not DataFrame:
            raise ValueError("Argument portFolioModel isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        currentItemID:int = argumentsDict[self.ARG_ITEM_ID]
        numberOfRecomItems:int = argumentsDict[self.ARG_NUMBER_OF_RECOMM_ITEMS]
        numberOfAggrItems:int = argumentsDict[self.ARG_NUMBER_OF_AGGR_ITEMS]

        resultsOfBaseRecommendersDict:Dict[str,object] = {}

        for recomI, recomIdI, recomDescsI in zip(self._recommenders, self._recommIDs, self._recomDescs):

           arguments2Dict:Dict[str, object] = {}
           arguments2Dict.update(recomDescsI.getArguments())
           arguments2Dict.update(argumentsDict)

           resultOfRecommendationI:List[int] = recomI.recommend(
               userID, numberOfItems=numberOfRecomItems, argumentsDict=arguments2Dict)
           resultOfRecommendationI = resultOfRecommendationI.loc[resultOfRecommendationI.index.dropna()]
           #resultOfRecommendationI.index = resultOfRecommendationI.index.astype(int)
           resultsOfBaseRecommendersDict[recomIdI] = resultOfRecommendationI


        aggItemIDsWithResponsibility:List
        aggItemIDsWithResponsibility = self._aggregation.runWithResponsibility(
           resultsOfBaseRecommendersDict, portFolioModel, userID, numberOfAggrItems, argumentsDict)
        #print(aggregatedItemIDsWithResponsibility)

        aggItemIDs:List[int] = list(map(lambda rs: rs[0], aggItemIDsWithResponsibility))

        #print("aggItemIDs: " + str(aggItemIDs))
        #print("aggItemIDsWithResponsibility: " + str(aggItemIDsWithResponsibility))
        #print("resultsOfRecommendations: " + str(resultsOfBaseRecommendersDict))

        resultsOfBRDict:Dict = {keyI:resultsOfBaseRecommendersDict[keyI].to_json() for keyI in resultsOfBaseRecommendersDict.keys()}

        logDict = {
            'userID': userID,
            'currentitemID': currentItemID,
            'recommended': aggItemIDsWithResponsibility,
            'bases': json.dumps(resultsOfBRDict)
            }

        logJson:str = json.dumps(logDict, default=myconverter).replace("\\", "")

        self._logFile.write(logJson)
        self._logFile.write("\n")
        self._logFile.flush()

        # Tuple
        return (aggItemIDs, aggItemIDsWithResponsibility)

def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    #elif isinstance(obj, datetime.datetime):
    #    return obj.__str__()
    return obj