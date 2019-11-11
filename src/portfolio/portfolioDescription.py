#!/usr/bin/python3

from recommender.recommenderDescription import RecommenderDescription #class

from aggregation.aggregationDescription import AggregationDescription #class

class PortfolioDescription:

    # recommIDs:str[], recommDescrs:RecommenderDescription[], aggrDescr:AggregationDescription
    def __init__(self, recommIDs, recommDescrs, aggrDescr):

       if type(recommIDs) is not list:
          raise ValueError("Type of argument recommIDs is not list.")
       #recommIDI:str
       for recommIDI in recommIDs:
          if type(recommIDI) is not str:
             raise ValueError("Argument recommIDs don't contains str.")

       if type(recommDescrs) is not list:
          raise ValueError("Type of argument recommDescrs is not list.")
       #recommDescrI:RecommenderDescription
       for recommDescrI in recommDescrs:
          if type(recommDescrI) is not RecommenderDescription:
             raise ValueError("Argument recommDescrs don't contains RecommenderDescription.")

       if type(aggrDescr) is not AggregationDescription :
          raise ValueError("Type of argument aggrDescr is not AggregationDescription.")

       self._recommIDs = recommIDs;
       self._recommDescrs = recommDescrs;
       self._aggrDescr = aggrDescr;

   
    def getRecommendersIDs(self):
       return self._recommIDs;

    def getRecommendersDescriptions(self):
       return self._recommDescrs;

    def getAggregationDescription(self):
       return self._aggrDescr;

