#!/usr/bin/python3

from typing import List

from recommender.aRecommender import ARecommender #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from portfolio.portfolio1Aggr import Portfolio1Aggr #class
from portfolioDescription.aPortfolioDescription import APortfolioDescription #class
from history.aHistory import AHistory #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class

class Portfolio1AggrDescription(APortfolioDescription):

    def __init__(self, portfolioID:str, recommIDs:List[str], recommDescrs:List[RecommenderDescription], aggrDescr:List[AggregationDescription]):

       if type(portfolioID) is not str:
          raise ValueError("Type of portfolioID isn't str.")
       self._portfolioID:str = portfolioID

       if type(recommIDs) is not list:
          raise ValueError("Type of argument recommIDs isn't list.")
       recommIDI:str
       for recommIDI in recommIDs:
          if type(recommIDI) is not str:
             raise ValueError("Argument recommIDs don't contains str.")

       if type(recommDescrs) is not list:
          raise ValueError("Type of argument recommDescrs isn't list.")
       recommDescrI:RecommenderDescription
       for recommDescrI in recommDescrs:
          if type(recommDescrI) is not RecommenderDescription:
             raise ValueError("Argument recommDescrs don't contains RecommenderDescription.")

       if type(aggrDescr) is not AggregationDescription :
          raise ValueError("Type of argument aggrDescr isn't AggregationDescription.")

       self._recommIDs:List[str] = recommIDs
       self._recommDescrs:List[RecommenderDescription] = recommDescrs
       self._aggrDescr:AggregationDescription = aggrDescr

    def getPortfolioID(self):
       return self._portfolioID

    def getRecommendersIDs(self):
       return self._recommIDs

    def getRecommendersDescriptions(self):
       return self._recommDescrs

    def getAggregationDescription(self):
       return self._aggrDescr



    def exportPortfolio(self, batchID:str, history:AHistory):
       if type(batchID) is not str:
          raise ValueError("Type of argument batchID isn't str.")
       if not isinstance(history, AHistory):
          raise ValueError("Type of argument history isn't AHistory.")

       recommenders:List[ARecommender] = []

       recommDescrI:RecommenderDescription
       for recommDescrI in self._recommDescrs:
          recommenderI:ARecommender = recommDescrI.exportRecommender(batchID)
          recommenders.append(recommenderI)

       # aggregation:Aggregation
       aggregation = self._aggrDescr.exportAggregation(history)

       return Portfolio1Aggr(batchID, self.getPortfolioID(), recommenders, self._recommIDs, self._recommDescrs, aggregation)
