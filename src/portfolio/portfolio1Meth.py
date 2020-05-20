#!/usr/bin/python3

import numpy as np

from typing import List #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from recommender.aRecommender import ARecommender #class

from history.aHistory import AHistory #class
from portfolio.aPortfolio import APortfolio #class


class Portfolio1Meth(APortfolio):

   def __init__(self, recommID:str, recommender:ARecommender):
       if type(recommID) is not str:
          raise ValueError("Argument recommID isn't type str.")
       if not isinstance(recommender, ARecommender):
           raise ValueError("Argument recommender isn't type ARecommender.")

       self._recommID:str = recommID
       self._recommender:ARecommender = recommender

   def getRecommIDs(self):
        return [self._recommID]

   def train(self, history:AHistory, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
        if not isinstance(history, AHistory):
           raise ValueError("Argument history isn't type AHistory.")
        if type(ratingsDF) is not DataFrame:
            raise ValueError("Argument ratingsDF isn't type DataFrame.")
        if type(usersDF) is not DataFrame:
            raise ValueError("Argument usersDF isn't type DataFrame.")
        if type(itemsDF) is not DataFrame:
            raise ValueError("Argument ratingsUpdateDF isn't type DataFrame.")

        self._recommender.train(history, ratingsDF, usersDF, itemsDF)

   def update(self, ratingsUpdateDF:DataFrame):
        if type(ratingsUpdateDF) is not DataFrame:
            raise ValueError("Argument ratingsUpdateDF isn't type DataFrame.")

        self._recommender.update(ratingsUpdateDF)


   def recommend(self, userID:int, portFolioModel:DataFrame, numberOfItems:int):
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(portFolioModel) is not DataFrame:
            raise ValueError("Argument portFolioModel isn't type DataFrame.")
        if type(numberOfItems) is not int:
            raise ValueError("Argument numberOfItems isn't type int.")

        recomItemIDsWithResponsibility:Series = self._recommender.recommend(userID, numberOfItems=numberOfItems)

        recomItemIDs:List[int] = list(recomItemIDsWithResponsibility.index)

        return (recomItemIDs, recomItemIDsWithResponsibility)
