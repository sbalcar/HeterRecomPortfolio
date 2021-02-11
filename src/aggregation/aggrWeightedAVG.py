#!/usr/bin/python3
import random

import numpy as np
import pandas as pd

from numpy.random import beta
from typing import List
from typing import Dict #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from aggregation.aAggregation import AAgregation #class
from aggregation.tools.responsibilityDHont import countDHontResponsibility #function
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class

from history.aHistory import AHistory #class
from abc import ABC, abstractmethod

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class


class AggrWeightedAVG(AAgregation):

    def __init__(self, history:AHistory, argumentsDict:Dict[str,object]):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        self._history = history

    def update(self, ratingsUpdateDF:DataFrame, argumentsDict:Dict[str,object]):
        pass


    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # modelDF:pd.DataFrame[numberOfVotes:int], numberOfItems:int
    def run(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems:int, argumentsDict:Dict[str,object]={}):

      # testing types of parameters
      if type(methodsResultDict) is not dict:
          raise ValueError("Type of methodsResultDict isn't dict.")
      if type(modelDF) is not DataFrame:
          raise ValueError("Type of methodsParamsDF isn't DataFrame.")
      if list(modelDF.columns) != ['votes']:
          raise ValueError("Argument methodsParamsDF doen't contain rights columns.")
      if type(numberOfItems) is not int:
          raise ValueError("Type of numberOfItems isn't int.")

      if sorted([mI for mI in modelDF.index]) != sorted([mI for mI in methodsResultDict.keys()]):
          raise ValueError("Arguments methodsResultDict and methodsParamsDF have to define the same methods.")
      for mI in methodsResultDict.keys():
          if modelDF.loc[mI] is None:
              raise ValueError("Argument modelDF contains in ome method an empty list of items.")
      if numberOfItems < 0:
          raise ValueError("Argument numberOfItems must be positive value.")
      if type(argumentsDict) is not dict:
          raise ValueError("Argument argumentsDict isn't type dict.")

      # votes number of parties
      #print("VotesOfPartiesDictI: ", votesOfPartiesDictI)
      #weighting of individual recommenders
      resSeries = None
      for mI in modelDF.index:
        votes =  modelDF.votes.loc[mI]
        weightedRes = methodsResultDict[mI] * votes
        #print(weightedRes)
        if isinstance(resSeries, pd.Series):
           resSeries = resSeries.append(weightedRes)
        else:
           resSeries = weightedRes 

      #print(resSeries)
      
      groupedResults = resSeries.groupby(level=0).sum()    
      groupedResults.sort_values(ascending = False, inplace = True)
      #print(groupedResults)

      recommendedItemIDs =  groupedResults.index[:numberOfItems].tolist()
      return recommendedItemIDs


    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # modelDF:pd.DataFrame[numberOfVotes:int], numberOfItems:int
    def runWithScore(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems:int, argumentsDict:Dict[str,object]={}):

      # testing types of parameters
      if type(methodsResultDict) is not dict:
          raise ValueError("Type of methodsResultDict isn't dict.")
      if type(modelDF) is not DataFrame:
          raise ValueError("Type of methodsParamsDF isn't DataFrame.")
      if list(modelDF.columns) != ['votes']:
          raise ValueError("Argument methodsParamsDF doen't contain rights columns.")
      if type(numberOfItems) is not int:
          raise ValueError("Type of numberOfItems isn't int.")

      if sorted([mI for mI in modelDF.index]) != sorted([mI for mI in methodsResultDict.keys()]):
          raise ValueError("Arguments methodsResultDict and methodsParamsDF have to define the same methods.")
      for mI in methodsResultDict.keys():
          if modelDF.loc[mI] is None:
              raise ValueError("Argument modelDF contains in ome method an empty list of items.")
      if numberOfItems < 0:
          raise ValueError("Argument numberOfItems must be positive value.")
      if type(argumentsDict) is not dict:
          raise ValueError("Argument argumentsDict isn't type dict.")

      # votes number of parties
      #print("VotesOfPartiesDictI: ", votesOfPartiesDictI)
      #weighting of individual recommenders
      resSeries = None
      for mI in modelDF.index:
        votes =  modelDF.votes.loc[mI]
        weightedRes = methodsResultDict[mI] * votes
        #print(weightedRes)
        if isinstance(resSeries, pd.Series):
           resSeries = resSeries.append(weightedRes)
        else:
           resSeries = weightedRes 

      #print(resSeries)
      
      groupedResults = resSeries.groupby(level=0).sum()    
      groupedResults.sort_values(ascending = False, inplace = True)
      #print(groupedResults)

      return groupedResults.iloc[:numberOfItems]
      


    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # modelDF:DataFrame<(methodID:str, votes:int)>, numberOfItems:int
    def runWithResponsibility(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems:int, argumentsDict:Dict[str,object]={}):

        # testing types of parameters
        if type(methodsResultDict) is not dict:
            raise ValueError("Type of methodsResultDict isn't dict.")
        for methI in methodsResultDict.values():
            if type(methI) is not pd.Series:
                raise ValueError("Type of methodsParamsDF doen't contain Series.")
        if type(modelDF) is not DataFrame:
            raise ValueError("Type of methodsParamsDF isn't DataFrame.")
        if list(modelDF.columns) != ['votes']:
            raise ValueError("Argument methodsParamsDF doesn't contain rights columns.")
        if type(numberOfItems) is not int:
            raise ValueError("Type of numberOfItems isn't int.")

        if sorted([mI for mI in modelDF.index]) != sorted([mI for mI in methodsResultDict.keys()]):
            raise ValueError("Arguments methodsResultDict and methodsParamsDF have to define the same methods.")
        for mI in methodsResultDict.keys():
            if modelDF.loc[mI] is None:
                raise ValueError("Argument modelDF contains in ome method an empty list of items.")
        if numberOfItems < 0:
            raise ValueError("Argument numberOfItems must be positive value.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        aggregatedItemIDs:List[int] = self.run(methodsResultDict, modelDF, userID, numberOfItems)
        itemsWithResposibilityOfRecommenders = []
        for item in  aggregatedItemIDs:
            votesOfItemDictI:dict[str,float] = {mI:methodsResultDict[mI].get(key = item, default = 0) for mI in modelDF.index}
            itemsWithResposibilityOfRecommenders.append((item, votesOfItemDictI))

        # list<(itemID:int, Series<(rating:int, methodID:str)>)>
        return itemsWithResposibilityOfRecommenders

