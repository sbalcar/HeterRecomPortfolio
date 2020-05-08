#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

from typing import List

from aggregation.aAggregation import AAgregation #class

from recommender.aRecommender import ARecommender #class

from recommendation.resultOfRecommendation import ResultOfRecommendation #class
from recommendation.resultsOfRecommendations import ResultsOfRecommendations #class

from history.aHistory import AHistory #class
from portfolio.aPortfolio import APortfolio #class

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

   def train(self, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
       if type(ratingsDF) is not DataFrame:
           raise ValueError("Argument ratingsDF isn't type DataFrame.")
       if type(usersDF) is not DataFrame:
           raise ValueError("Argument usersDF isn't type DataFrame.")
       if type(itemsDF) is not DataFrame:
           raise ValueError("Argument ratingsUpdateDF isn't type DataFrame.")

       recommenderI:ARecommender
       for recommenderI in self._recommenders:
           recommenderI.train(ratingsDF, usersDF, itemsDF)

   def update(self, ratingsUpdateDF:DataFrame):
       if type(ratingsUpdateDF) is not DataFrame:
           raise ValueError("Argument ratingsUpdateDF isn't type DataFrame.")

       recommenderI:ARecommender
       for recommenderI in self._recommenders:
           recommenderI.update(ratingsUpdateDF)


   # portFolioModel:DataFrame<(methodID, votes)>
   def recommendToItem(self, portFolioModel:DataFrame, itemID:int, testRatingsDF:DataFrame, history:AHistory, numberOfItems:int):

       resultsOfRecommendations:ResultsOfRecommendations = ResultsOfRecommendations([], [])

       recommIndexI: int
       for recommIndexI in range(len(self._recommenders)):
           recommIDI: str = self._recommIDs[recommIndexI]
           recommI: ARecommender = self._recommenders[recommIndexI]

           resultOfRecommendationI: ResultOfRecommendation = recommI.recommendToItem(itemID, testRatingsDF, history, numberOfItems)
           resultsOfRecommendations.add(recommIDI, resultOfRecommendationI)

       #aggregatedItemIDs:List[int] = self._aggregation.run(resultsOfRecommendations, userDef, numberOfItems)

       methodsResultDict:dict = resultsOfRecommendations.exportAsDictionaryOfSeries()
       #print(methodsResultDict)

       aggregatedItemIDsWithResponsibility:list = self._aggregation.runWithResponsibility(
           methodsResultDict, portFolioModel, numberOfItems=numberOfItems)
       #print(aggregatedItemIDsWithResponsibility)

       # list<int>
       return aggregatedItemIDsWithResponsibility

