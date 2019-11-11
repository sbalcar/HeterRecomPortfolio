#!/usr/bin/python3

from configuration.arguments import Arguments #class
from configuration.argument import Argument #class

from recommender.recommenderDescription import RecommenderDescription #class

from portfolio.portfolioDescription import PortfolioDescription #class

from recommendation.resultOfRecommendation import ResultOfRecommendation #class
from recommendation.resultsOfRecommendations import ResultsOfRecommendations #class

from aggregation.aggregationDescription import AggregationDescription #class

from aggregation.aggrElections import AggrElections #class


class Portfolio:

   # portfolioDescr:PortfolioDescription
   def __init__(self, portfolioDescr):

      if type(portfolioDescr) is not PortfolioDescription :
         raise ValueError("Argument portfolioDescr is not type PortfolioDescription.")

      # _portfolioDescr:PortfolioDescription
      self._portfolioDescr = portfolioDescr;


      # recommDescrs:RecommenderDescription[]
      recommDescrs = self._portfolioDescr.getRecommendersDescriptions();

      # _recommenders:Recommender[]
      self._recommenders = []

      # recommDescrI:RecommenderDescription
      for recommDescrI in recommDescrs:
          # recommenderI:Recommender
          recommenderI = recommDescrI.exportRecommender()
          self._recommenders.append(recommenderI);


      # aggrDescr:AggregationDescription
      aggrDescr = self._portfolioDescr.getAggregationDescription();

      # _aggregation:Aggregation
      self._aggregation = aggrDescr.exportAggregation()


   # evaluationOfRecommenders:EvaluationOfRecommenders, numberOfItems:int
   def run(self, evaluationOfRecommenders, numberOfItems):

      # resultsOfRecommendations:ResultsOfRecommendations
      resultsOfRecommendations = ResultsOfRecommendations();

      # recommIndexI:int
      for recommIndexI in range(len(self._recommenders)):

          # recommenderI:Recommender
          recommenderI = self._recommenders[recommIndexI]

          # recommIDI:str
          recommIDI = self._portfolioDescr.getRecommendersIDs()[recommIndexI]

          # resultOfRecommendationI:ResultOfRecommendation
          resultOfRecommendationI = recommenderI.recommend(numberOfItems)

          resultsOfRecommendations.add(recommIDI, resultOfRecommendationI);


      # aggrDescr:AggregationDescription
      #aggrDescr = AggregationDescription(AggrElections, Arguments([]))
      aggrDescr = self._portfolioDescr.getAggregationDescription()
      aggr = aggrDescr.exportAggregation()

      # items:list<int>
      itemIDs = aggr.run(resultsOfRecommendations, evaluationOfRecommenders, numberOfItems)
      
      # list<int> 
      return itemIDs

