#!/usr/bin/python3

from typing import List

from recommender.aRecommender import ARecommender #class

from recommender.description.recommenderDescription import RecommenderDescription #class

from aggregation.aggregationDescription import AggregationDescription #class

from portfolio.portfolio import Portfolio #class


class PortfolioDescription:

    def __init__(self, recommIDs:List[str], recommDescrs:List[RecommenderDescription], aggrDescr:List[AggregationDescription]):

       if type(recommIDs) is not list:
          raise ValueError("Type of argument recommIDs is not list.")
       recommIDI:str
       for recommIDI in recommIDs:
          if type(recommIDI) is not str:
             raise ValueError("Argument recommIDs don't contains str.")

       if type(recommDescrs) is not list:
          raise ValueError("Type of argument recommDescrs is not list.")
       recommDescrI:RecommenderDescription
       for recommDescrI in recommDescrs:
          if type(recommDescrI) is not RecommenderDescription:
             raise ValueError("Argument recommDescrs don't contains RecommenderDescription.")

       if type(aggrDescr) is not AggregationDescription :
          raise ValueError("Type of argument aggrDescr is not AggregationDescription.")

       self._recommIDs:list[str] = recommIDs;
       self._recommDescrs:list[RecommenderDescription] = recommDescrs;
       self._aggrDescr:AggregationDescription = aggrDescr;

   
    def getRecommendersIDs(self):
       return self._recommIDs

    def getRecommendersDescriptions(self):
       return self._recommDescrs

    def getAggregationDescription(self):
       return self._aggrDescr



    def exportPortfolio(self):

       recommenders: List[ARecommender] = []

       recommDescrI: RecommenderDescription
       for recommDescrI in self._recommDescrs:
          recommenderI: ARecommender = recommDescrI.exportRecommender()
          recommenders.append(recommenderI)

       # aggregation:Aggregation
       aggregation = self._aggrDescr.exportAggregation()

       return Portfolio(self._recommIDs, recommenders, aggregation)
