#!/usr/bin/python3

import numpy as np

from typing import List #class
from  typing import Dict #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class
from recommender.aRecommender import ARecommender #class

from history.aHistory import AHistory #class
from portfolio.aPortfolio import APortfolio #class
from portfolio.portfolio1Meth import Portfolio1Meth #class


class PortfolioNeg1Meth(Portfolio1Meth):

   def __init__(self, recommender:ARecommender, recomID:str, recomDesc:RecommenderDescription,
                penaltyTool:APenalization):

       if not isinstance(recommender, ARecommender):
           raise ValueError("Argument recommender isn't type ARecommender.")
       if type(recomID) is not str:
          raise ValueError("Argument recomID isn't type str.")
       if type(recomDesc) is not RecommenderDescription:
          raise ValueError("Argument recomDesc isn't type RecommenderDescription.")
       if not isinstance(penaltyTool, APenalization):
          raise ValueError("Argument penaltyTool isn't type APenalization.")

       self._recommender:ARecommender = recommender
       self._recomID:str = recomID
       self._recomDesc:RecommenderDescription = recomDesc
       self._penaltyTool:APenalization = penaltyTool


   def recommend(self, userID:int, portFolioModel:DataFrame, argumentsDict:Dict[str,object]):
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(portFolioModel) is not DataFrame:
            raise ValueError("Argument portFolioModel isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        numberOfRecommItems:int = argumentsDict[self.ARG_NUMBER_OF_RECOMM_ITEMS]
        numberOfAggrItems:int = argumentsDict[self.ARG_NUMBER_OF_AGGR_ITEMS]
        #print("ARG_NUMBER_OF_AGGR_ITEMS " + str(argumentsDict[self.ARG_NUMBER_OF_AGGR_ITEMS]))
        #print("ARG_NUMBER_OF_RECOMM_ITEMS " + str(argumentsDict[self.ARG_NUMBER_OF_RECOMM_ITEMS]))

        arguments2Dict:Dict[str,object] = {}
        arguments2Dict.update(self._recomDesc.getArguments())
        arguments2Dict.update(argumentsDict)

        recomItemIDsWithRating:Series = self._recommender.recommend(
            userID, numberOfItems=numberOfRecommItems, argumentsDict=arguments2Dict)

        penalizedRecomItemIDsWithRating:Series = self._penaltyTool.runOneMethodPenalization(
            userID, recomItemIDsWithRating, self._history)

        cuttedRecomItemIDsWithRating:Series = penalizedRecomItemIDsWithRating[:numberOfAggrItems]

        cuttedRecomItemIDs:List[int] = list(cuttedRecomItemIDsWithRating.index)

        return (cuttedRecomItemIDs, cuttedRecomItemIDsWithRating, None)
