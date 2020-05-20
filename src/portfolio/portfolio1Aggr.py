#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

from typing import List

from aggregation.aAggregation import AAgregation #class

from recommender.aRecommender import ARecommender #class

from history.aHistory import AHistory #class
from portfolio.aPortfolio import APortfolio #class

import pandas as pd
import numpy as np


class Portfolio1Aggr(APortfolio):

   def __init__(self, recommIDs:List[str], recommenders:List[ARecommender], agregation:AAgregation):
      if type(recommIDs) is not list:
         raise ValueError("Argument recommIDs isn't type list.")

      for recommIdI in recommIDs:
          if not type(recommIdI) is str:
              raise ValueError("Argument recommIDs don't contains type str.")

      if type(recommenders) is not list:
         raise ValueError("Argument recommenders isn't type list.")
      for recommenderI in recommenders:
          if not isinstance(recommenderI, ARecommender):
              raise ValueError("Argument recommenders don't contains type ARecommender.")

      if not isinstance(agregation, AAgregation):
         raise ValueError("Argument agregation isn't type AAgregation.")

      self._recommIDs:List[str] = recommIDs
      self._recommenders:List[ARecommender] = recommenders
      self._aggregation:AAgregation = agregation

   def getRecommIDs(self):
       return self._recommIDs

   def train(self, history:AHistory, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if type(ratingsDF) is not DataFrame:
            raise ValueError("Argument ratingsDF isn't type DataFrame.")
        if type(usersDF) is not DataFrame:
            raise ValueError("Argument usersDF isn't type DataFrame.")
        if type(itemsDF) is not DataFrame:
            raise ValueError("Argument ratingsUpdateDF isn't type DataFrame.")

        recommenderI:ARecommender
        for recommenderI in self._recommenders:
            recommenderI.train(history, ratingsDF, usersDF, itemsDF)

   def update(self, ratingsUpdateDF:DataFrame):
       if type(ratingsUpdateDF) is not DataFrame:
           raise ValueError("Argument ratingsUpdateDF isn't type DataFrame.")

       recommenderI:ARecommender
       for recommenderI in self._recommenders:
           recommenderI.update(ratingsUpdateDF)


   # portFolioModel:DataFrame<(methodID, votes)>
   def recommend(self, userID:int, portFolioModel:DataFrame, numberOfItems:int):
       if type(userID) is not int and type(userID) is not np.int64:
           raise ValueError("Argument userID isn't type int.")
       if type(portFolioModel) is not DataFrame:
           raise ValueError("Argument portFolioModel isn't type DataFrame.")
       if type(numberOfItems) is not int:
           raise ValueError("Argument numberOfItems isn't type int.")

       resultsOfRecommendations:dict = {}

       recommIndexI: int
       for recommIndexI in range(len(self._recommenders)):
           recommIDI: str = self._recommIDs[recommIndexI]
           recommI:ARecommender = self._recommenders[recommIndexI]

           resultOfRecommendationI:List[int] = recommI.recommend(userID, numberOfItems=numberOfItems)
           resultsOfRecommendations[recommIDI] = resultOfRecommendationI

       aggItemIDsWithResponsibility:List = self._aggregation.runWithResponsibility(
           resultsOfRecommendations, portFolioModel, userID, numberOfItems=numberOfItems)
       #print(aggregatedItemIDsWithResponsibility)

       aggItemIDs:List[int] = list(map(lambda rs: rs[0], aggItemIDsWithResponsibility))

       # list<int>
       return (aggItemIDs, aggItemIDsWithResponsibility)
