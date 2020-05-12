#!/usr/bin/python3

from typing import List

import random
import numpy as np
import pandas as pd

from numpy.random import beta
from typing import List

from aggregation.aAggregation import AAgregation

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class


class AggrBanditTS(AAgregation):

    def __init__(self, argumentsDict:dict):
       if type(argumentsDict) is not dict:
          raise ValueError("Argument argumentsDict is not type dict.")

       self._argumentsDict = argumentsDict

    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # modelDF:pd.DataFrame[methodID:String, r:int, n:int, alpha0:int, beta0:int], numberOfItems:int
    def run(self, methodsResultDict:dict, modelDF:DataFrame, numberOfItems=20):

      if sorted([mI for mI in modelDF.index]) != sorted([mI for mI in methodsResultDict.keys()]):
          raise ValueError("Arguments methodsResultDict and methodsParamsDF have to define the same methods.")
      for methIdI in methodsResultDict.keys():
          if modelDF.loc[methIdI] is None:
              raise ValueError("Argument methodsParamsDF contains in ome method an empty list of items.")
      if numberOfItems < 0 :
          raise ValueError("Argument numberOfItems can't contain negative value.")


      methodsResultDictI:dict = methodsResultDict
      methodsParamsDFI:DataFrame = modelDF

      recommendedItemIDs:List[tuple(int,str)] = []

      for iIndex in range(0, numberOfItems):
        #print("iIndex: ", iIndex)
        #print(methodsResultDictI)
        #print(methodsParamsDFI)

        if len([mI for mI in methodsResultDictI]) == 0:
          return recommendedItemIDs[:numberOfItems]

        methodProbabilitiesDicI:dict = {}

        # computing probabilities of methods
        for mIndex in methodsParamsDFI.index:
          #print("mIndexI: ", mIndex)
          methodI = methodsParamsDFI.loc[methodsParamsDFI.index == mIndex]#.iloc[0]
          # alpha + number of successes, beta + number of failures
          pI = beta(methodI.alpha0 + methodI.r, methodI.beta0 + (methodI.n - methodI.r), size=1)[0]
          methodProbabilitiesDicI[mIndex] = pI
        #print(methodProbabilitiesDicI)

        # get max probability of method prpabilities
        maxPorbablJ:float = max(methodProbabilitiesDicI.values())
        #print("MaxPorbablJ: ", maxPorbablJ)

        # selecting method with highest probability
        theBestMethodID:str = random.choice([aI for aI in methodProbabilitiesDicI.keys() if methodProbabilitiesDicI[aI] == maxPorbablJ])
    
        # extractiion results of selected method (method with highest probability)
        resultsOfMethodI:Series = methodsResultDictI.get(theBestMethodID)
        #print(resultsOfMethodI)
    
        # select next item (itemID)
        selectedItemI:int = self.__exportRouletteWheelRatedItem(resultsOfMethodI)
        #selectedItemI = self.__exportRandomItem(resultsOfMethodI)
        #selectedItemI = self.__exportTheMostRatedItem(resultsOfMethodI)
    
        #print("SelectedItemI: ", selectedItemI)
    
        recommendedItemIDs.append((selectedItemI, theBestMethodID))

        # deleting selected element from method results
        for mrI in methodsResultDictI:
            try:
                methodsResultDictI[mrI].drop(selectedItemI, inplace=True, errors="ignore")
            except:
                #TODO some error recordings?
                pass
        #methodsResultDictI = {mrI:methodsResultDictI[mrI].append(pd.Series([None],[selectedItemI])).drop(selectedItemI) for mrI in methodsResultDictI}
        #print(methodsResultDictI)

        # methods with empty list of items
        methodEmptyI = [mI for mI in methodsResultDictI if len(methodsResultDictI.get(mI)) == 0]

        # removing methods with the empty list of items
        methodsParamsDFI = methodsParamsDFI[~methodsParamsDFI.index.isin(methodEmptyI)]

        # removing methods definition with the empty list of items
        for meI in methodEmptyI: methodsResultDictI.pop(meI)
      return recommendedItemIDs[:numberOfItems]


    # resultOfMethod:pd.Series([raitings],[itemIDs])
    def __exportTheMostRatedItem(self, resultOfMethod):
      maxValue = max(resultOfMethod.values)
      return resultOfMethod[resultOfMethod == maxValue].index[0]
      #return method.idxmax()

    # resultOfMethod:pd.Series([raitings],[itemIDs])
    def __exportTheFirstItem(self, resultOfMethod:Series):
      return resultOfMethod.index[0]

    # resultOfMethod:pd.Series([raitings],[itemIDs])
    def __exportRandomItem(self, resultOfMethod:Series):
      return random.choice(resultOfMethod.index)

    # resultOfMethod:pd.Series([raitings],[itemIDs])
    def __exportRouletteWheelRatedItem(self, resultOfMethod:Series):
        # weighted random choice
        pick = random.uniform(0, sum(resultOfMethod.values))
        current = 0
        for itemIDI in resultOfMethod.index:
            current += resultOfMethod[itemIDI]
            if current > pick:
              return itemIDI




    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # modelDF:DataFrame<(methodID:str, votes:int)>, numberOfItems:int
    def runWithResponsibility(self, methodsResultDict:dict, modelDF:DataFrame, numberOfItems:int=20):

        return self.run(methodsResultDict, modelDF, numberOfItems)