import random
import numpy as np
import pandas as pd

from numpy.random import beta

from configuration.arguments import Arguments #class
from configuration.argument import Argument #class

from recommendation.resultOfRecommendation import ResultOfRecommendation #class
from recommendation.resultsOfRecommendations import ResultsOfRecommendations #class


class AggrBanditTS:

    # arguments:Arguments
    def __init__(self, arguments):

       if type(arguments) is not Arguments:
          raise ValueError("Argument arguments is not type Arguments.")

       self._arguments = arguments;

    # resultsOfRecommendations:ResultsOfRecommendations, evaluationOfRecommenders:EvaluationOfRecommenders, numberOfItems:int
    def run(self, resultsOfRecommendations, evaluationOfRecommenders, numberOfItems=20):

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
    def aggrBanditTSRun(self, methodsResultDict, methodsParamsDF, topK=20):

      if sorted([mI for mI in methodsParamsDF.index]) != sorted([mI for mI in methodsResultDict.keys()]):
        raise ValueError("Arguments methodsResultDict and methodsParamsDF have to define the same methods.")

      if np.prod([len(methodsResultDict.get(mI)) for mI in methodsResultDict]) == 0:
        raise ValueError("Argument methodsParamsDF contains in ome method an empty list of items.")

      if topK < 0 :
        raise ValueError("Argument topK must be positive value.")


      methodsResultDictI = methodsResultDict;
      methodsParamsDFI = methodsParamsDF;

      recommendedItemIDs = []

      for iIndex in range(0, topK):
        #print("iIndex: ", iIndex)
        #print(methodsResultDictI)
        #print(methodsParamsDFI)

        if len([mI for mI in methodsResultDictI]) == 0:
          return recommendedItemIDs[:topK];

        methodProbabilitiesDicI = {}

        # computing probabilities of methods
        for mIndex in methodsParamsDFI.index:
          #print("mIndexI: ", mIndex)
          methodI = methodsParamsDFI.loc[methodsParamsDFI.index == mIndex]#.iloc[0]
          # alpha + number of successes, beta + number of failures
          pI = beta(methodI.alpha0 + methodI.r, methodI.beta0 + (methodI.n - methodI.r), size=1)[0]
          methodProbabilitiesDicI[mIndex] = pI
        #print(methodProbabilitiesDicI)

        # get max probability of method prpabilities
        maxPorbablJ = max(methodProbabilitiesDicI.values())
        #print("MaxPorbablJ: ", maxPorbablJ)

        # selecting method with highest probability
        theBestMethodID = random.choice([aI for aI in methodProbabilitiesDicI.keys() if methodProbabilitiesDicI[aI] == maxPorbablJ])
    
        # extractiion results of selected method (method with highest probability)
        resultsOfMethodI = methodsResultDictI.get(theBestMethodID)
        #print(resultsOfMethodI)
    
        # select next item (itemID)
        selectedItemI = self.exportRouletteWheelRatedItem(resultsOfMethodI)
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
    def exportTheFirstItem(self, resultOfMethod):
      return resultOfMethod.index[0]

    # resultOfMethod:pd.Series([raitings],[itemIDs])
    def exportRandomItem(self, resultOfMethod):
      return random.choice(resultOfMethod.index)

    # resultOfMethod:pd.Series([raitings],[itemIDs])
    def exportRouletteWheelRatedItem(self, resultOfMethod):
        # weighted random choice
        pick = random.uniform(0, sum(resultOfMethod.values))
        current = 0
        for itemIDI in resultOfMethod.index:
            current += resultOfMethod[itemIDI]
            if current > pick:
              return itemIDI


