#!/usr/bin/python3

from typing import List

import random
import numpy as np
import pandas as pd

from numpy.random import beta
from typing import List

from aggregation.tools.responsibilityDHont import countDHontResponsibility #function

from recommendation.resultOfRecommendation import ResultOfRecommendation #class
from recommendation.resultsOfRecommendations import ResultsOfRecommendations #class

from evaluationOfRecommender.evaluationOfRecommenders import EvaluationOfRecommenders

from aggregation.aAggregation import AAgregation

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class


class AggrBanditTS(AAgregation):

    def __init__(self, argumentsDict:dict):

       if type(argumentsDict) is not dict:
          raise ValueError("Argument argumentsDict is not type dict.")

       self._argumentsDict = argumentsDict

    def run(self, resultsOfRecommendations:ResultsOfRecommendations, evaluationOfRecommenders:EvaluationOfRecommenders, numberOfItems:int=20):

        if type(resultsOfRecommendations) is not ResultsOfRecommendations:
             raise ValueError("Argument resultsOfRecommendations is not type ResultsOfRecommendations.")

        if type(numberOfItems) is not int:
             raise ValueError("Argument numberOfItems is not type int.")

        methodsResultDict = resultsOfRecommendations.exportAsDictionaryOfSeries()
        #print(methodsResultDict)

        methodsParamsDF = evaluationOfRecommenders.exportAsParamsDF()
        #print(methodsParamsDF)

        return self.aggrBanditTSRun(methodsResultDict, methodsParamsDF, topK=numberOfItems)


    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # methodsParamsDF:pd.DataFrame[methodID:String, r:int, n:int, alpha0:int, beta0:int], topK:int
    def aggrBanditTSRun(self, methodsResultDict:dict, methodsParamsDF:DataFrame, topK=20):

      if sorted([mI for mI in methodsParamsDF.index]) != sorted([mI for mI in methodsResultDict.keys()]):
          raise ValueError("Arguments methodsResultDict and methodsParamsDF have to define the same methods.")

      if np.prod([len(methodsResultDict.get(mI)) for mI in methodsResultDict]) == 0:
        raise ValueError("Argument methodsParamsDF contains in ome method an empty list of items.")

      if topK < 0 :
        raise ValueError("Argument topK must be positive value.")


      methodsResultDictI:dict = methodsResultDict
      methodsParamsDFI:DataFrame = methodsParamsDF

      recommendedItemIDs:List[tuple(int,str)] = []

      for iIndex in range(0, topK):
        #print("iIndex: ", iIndex)
        #print(methodsResultDictI)
        #print(methodsParamsDFI)

        if len([mI for mI in methodsResultDictI]) == 0:
          return recommendedItemIDs[:topK]

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
        selectedItemI:int = self.exportRouletteWheelRatedItem(resultsOfMethodI)
        #selectedItemI = self.exportRandomItem(resultsOfMethodI)
        #selectedItemI = self.exportTheMostRatedItem(resultsOfMethodI)
    
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
      return recommendedItemIDs[:topK]


    # resultOfMethod:pd.Series([raitings],[itemIDs])
    def exportTheMostRatedItem(self, resultOfMethod):
      maxValue = max(resultOfMethod.values)
      return resultOfMethod[resultOfMethod == maxValue].index[0]
      #return method.idxmax()

    # resultOfMethod:pd.Series([raitings],[itemIDs])
    def exportTheFirstItem(self, resultOfMethod:Series):
      return resultOfMethod.index[0]

    # resultOfMethod:pd.Series([raitings],[itemIDs])
    def exportRandomItem(self, resultOfMethod:Series):
      return random.choice(resultOfMethod.index)

    # resultOfMethod:pd.Series([raitings],[itemIDs])
    def exportRouletteWheelRatedItem(self, resultOfMethod:Series):
        # weighted random choice
        pick = random.uniform(0, sum(resultOfMethod.values))
        current = 0
        for itemIDI in resultOfMethod.index:
            current += resultOfMethod[itemIDI]
            if current > pick:
              return itemIDI




    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # methodsParamsDF:DataFrame<(methodID:str, votes:int)>, topK:int
    def runWithResponsibility(self, methodsResultDict:dict, methodsParamsDF:DataFrame, numberOfItems:int=20):

        return self.aggrBanditTSRun(methodsResultDict, methodsParamsDF, numberOfItems)
