#!/usr/bin/python3

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

   def train(self, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
        if type(ratingsDF) is not DataFrame:
            raise ValueError("Argument ratingsDF isn't type DataFrame.")
        if type(usersDF) is not DataFrame:
            raise ValueError("Argument usersDF isn't type DataFrame.")
        if type(itemsDF) is not DataFrame:
            raise ValueError("Argument ratingsUpdateDF isn't type DataFrame.")

        self._recommender.train(ratingsDF, usersDF, itemsDF)

   def update(self, ratingsUpdateDF:DataFrame):
        if type(ratingsUpdateDF) is not DataFrame:
            raise ValueError("Argument ratingsUpdateDF isn't type DataFrame.")

        self._recommender.update(ratingsUpdateDF)


   def recommendToItem(self, portFolioModel:DataFrame, itemID:int, testRatingsDF:DataFrame, history:AHistory, numberOfItems:int):

        recomItemIDsWithResponsibility:Series = self._recommender.recommend(itemID, testRatingsDF, history, numberOfItems)

        recomItemIDs:List[int] = list(recomItemIDsWithResponsibility.index)

        return (recomItemIDs, recomItemIDsWithResponsibility)
