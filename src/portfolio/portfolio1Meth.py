#!/usr/bin/python3

import numpy as np

from typing import List #class
from typing import Dict #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.aDataset import ADataset #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class
from recommender.aRecommender import ARecommender #class

from history.aHistory import AHistory #class
from portfolio.aPortfolio import APortfolio #class


class Portfolio1Meth(APortfolio):

   def __init__(self, batchID:str, portfolioID:str, recommender:ARecommender, recomID:str, recomDesc:RecommenderDescription):

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

       self._batchID:str = batchID
       self._portfolioID:str = portfolioID
       self._recommender:ARecommender = recommender
       self._recomID:str = recomID
       self._recomDesc:RecommenderDescription = recomDesc

   def getRecommIDs(self):
        return [self._recommID]

   def train(self, history:AHistory, dataset:ADataset):
        if not isinstance(history, AHistory):
           raise ValueError("Argument history isn't type AHistory.")
        if not isinstance(dataset, ADataset):
           raise ValueError("Argument dataset isn't type ADataset.")

        self._history:AHistory = history

        self._recommender.train(history, dataset)

   def update(self, ratingsUpdateDF:DataFrame, argumentsDict:Dict[str,object]):
        if type(ratingsUpdateDF) is not DataFrame:
            raise ValueError("Argument ratingsUpdateDF isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")
        self._recommender.update(ratingsUpdateDF, argumentsDict)


   def recommend(self, userID:int, portFolioModel:DataFrame, argumentsDict:dict):
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(portFolioModel) is not DataFrame:
            raise ValueError("Argument portFolioModel isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        numberOfItems:int = argumentsDict[self.ARG_NUMBER_OF_AGGR_ITEMS]

        arguments2Dict:Dict[str,object] = {}
        arguments2Dict.update(self._recomDesc.getArguments())
        arguments2Dict.update(argumentsDict)

        recomItemIDsWithResponsibility:Series = self._recommender.recommend(userID, numberOfItems=numberOfItems, argumentsDict=arguments2Dict)
        #print(recomItemIDsWithResponsibility)

        recomItemIDs:List[int] = list(recomItemIDsWithResponsibility.index)

        return (recomItemIDs, recomItemIDsWithResponsibility)
