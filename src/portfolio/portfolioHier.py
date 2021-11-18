#!/usr/bin/python3

import os
import numpy as np
from sklearn.preprocessing import normalize

from typing import List #class
from typing import Dict #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from configuration.configuration import Configuration #class
from datasets.aDataset import ADataset #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class
from recommender.aRecommender import ARecommender #class

from aggregation.aAggregation import AAgregation #class
from aggregation.negImplFeedback.aPenalization import APenalization #class

from history.aHistory import AHistory #class
from portfolio.aPortfolio import APortfolio #class
from portfolio.portfolio1Aggr import Portfolio1Aggr #class

from aggregation.tools.aggrResponsibilityBandits import countAggrBanditsResponsibility #fnc
from aggregation.tools.aggrResponsibilityDHondt import countAggrDHondtResponsibility #fnc

class PortfolioHier(APortfolio):

    def __init__(self, batchID: str, portfolioID:str,
                 recommender:ARecommender, recomID:str, recomDesc:RecommenderDescription,
                 portfolio1Aggr:Portfolio1Aggr, aggrHier:AAgregation, penaltyTool:APenalization):
        if type(batchID) is not str:
            raise ValueError("Argument batchID isn't type str.")
        if type(portfolioID) is not str:
            raise ValueError("Argument portfolioID isn't type str.")
        if not isinstance(recommender, ARecommender):
            raise ValueError("Argument recommender isn't type ARecommender.")
        if type(recomID) is not str:
            raise ValueError("Argument recomID isn't type str.")
        if type(recomDesc) is not RecommenderDescription:
            raise ValueError("Argument recomDesc isn't type RecommenderDescription.")
        if type(portfolio1Aggr) is not Portfolio1Aggr:
            raise ValueError("Argument portfolio1Aggr isn't type Portfolio1Aggr.")
        if not isinstance(aggrHier, AAgregation):
            raise ValueError("Argument aggrHier isn't type AAgregation.")
        if not isinstance(penaltyTool, APenalization):
            raise ValueError("Argument penaltyTool isn't type APenalization.")

        self._batchID:str = batchID
        self._portfolioID:str = portfolioID
        self._recommender:ARecommender = recommender
        self._recomID:str = recomID
        self._recomDesc:RecommenderDescription = recomDesc
        self._portfolio1Aggr:Portfolio1Aggr = portfolio1Aggr
        self._aggrHier:AAgregation = aggrHier
        self._penaltyTool = penaltyTool

    def getRecommIDs(self):
        return [self._recomID]

    def train(self, history:AHistory, dataset:ADataset):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if not isinstance(dataset, ADataset):
            raise ValueError("Argument dataset isn't type ADataset.")

        self._history:AHistory = history

        self._recommender.train(history, dataset)
        self._portfolio1Aggr.train(history, dataset)
        self._aggrHier.train(history, dataset)

        dir:str = Configuration.resultsDirectory + os.sep + self._batchID
        logFileName:str = dir + os.sep + "log-portfolioHier" + self._portfolioID + ".txt"

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

        self._recommender.update(ratingsUpdateDF, argumentsDict)
        self._portfolio1Aggr.update(ratingsUpdateDF, argumentsDict)
        self._aggrHier.update(ratingsUpdateDF, argumentsDict)

    # portFolioModel:DataFrame<(methodID, votes)>
    def recommend(self, userID:int, portFolioModel:DataFrame, argumentsDict:Dict[str,object]):
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(portFolioModel) is not DataFrame:
            raise ValueError("Argument portFolioModel isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        numberOfItems:int = argumentsDict[self.ARG_NUMBER_OF_AGGR_ITEMS]

        recomItemIDsWithRspR1Ser:Series = self._recommender.recommend(userID, numberOfItems=numberOfItems, argumentsDict=argumentsDict)

        recomItemIDsAggr1:List[int]
        recomItemIDsWithRspAggr1:Series
        recomItemIDsAggr1, recomItemIDsWithRspAggr1 = self._portfolio1Aggr.recommend(userID, portFolioModel, argumentsDict=argumentsDict)
        print(recomItemIDsWithRspAggr1)

        aggrBanditsResp = countAggrBanditsResponsibility(recomItemIDsWithRspAggr1, portFolioModel)
        #aggrBanditsResp = countAggrDHondtResponsibility(dict(recomItemIDsWithRspAggr1), portFolioModel)
        aggrBanditsRespSer:Series = Series(dict(aggrBanditsResp))

        recomItemIDsNegativeSer:Series = Series(self._penaltyTool.getPenaltiesOfItemIDs(userID, self._history))
        if len(recomItemIDsNegativeSer) > 0:
            finalNegScores = normalize(np.expand_dims(recomItemIDsNegativeSer.values, axis=0))[0, :]
            recomItemIDsNegativeSer:Series = Series(finalNegScores.tolist(), index=recomItemIDsNegativeSer.index)
        #print(a)

        #recomItemIDsNegative = normalize(np.expand_dims(recomItemIDsNegative, axis=0))[0, :]

        inputItemIDsDict:dict = {"input1":recomItemIDsWithRspR1Ser,
                                 "input2":aggrBanditsRespSer,
                                 "negative":recomItemIDsNegativeSer}

        aggItemIDsWithRelevanceSer:Series = self._aggrHier.runWithResponsibility(inputItemIDsDict, DataFrame(), userID, numberOfItems, argumentsDict)

        aggItemIDs:List[int] = list(aggItemIDsWithRelevanceSer.index)

        aggItemIDsWithRelevance:List = [(itemI, dict(recomItemIDsWithRspAggr1).get(itemI, {})) for itemI in aggItemIDs]

        return (aggItemIDs, aggItemIDsWithRelevance)





if __name__ == "__main__":
    os.chdir("..")

    aggItemIDsWithRelevanceSer:List[tuple] = [(7, {'metoda1': 0, 'metoda2': 24.0, 'metoda3': 18.0}), (1, {'metoda1': 30.0, 'metoda2': 8.0, 'metoda3': 0}), (32, {'metoda1': 20.0, 'metoda2': 16.0, 'metoda3': 0}), (8, {'metoda1': 30.0, 'metoda2': 0, 'metoda3': 0}), (6, {'metoda1': 0, 'metoda2': 24.0, 'metoda3': 0}), (64, {'metoda1': 0, 'metoda2': 0, 'metoda3': 18.0}), (2, {'metoda1': 10.0, 'metoda2': 0, 'metoda3': 6.0}), (77, {'metoda1': 0, 'metoda2': 0, 'metoda3': 12.0}), (4, {'metoda1': 10.0, 'metoda2': 0, 'metoda3': 0}), (5, {'metoda1': 0, 'metoda2': 8.0, 'metoda3': 0}), (12, {'metoda1': 0, 'metoda2': 0, 'metoda3': 6.0})]

    aggItemIDs = [7, 1, 100]
    aggItemIDsWithRelevance:List = [(itemI,dict(aggItemIDsWithRelevanceSer).get(itemI, {})) for itemI in aggItemIDs]
