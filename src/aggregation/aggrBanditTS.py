#!/usr/bin/python3

from typing import List
from typing import Dict

import random
import numpy as np
import pandas as pd

from numpy.random import beta
from typing import List

from aggregation.aAggregation import AAgregation

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from history.aHistory import AHistory #class
from abc import ABC, abstractmethod

from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class


class AggrBanditTS(AAgregation):

    ARG_SELECTOR:str = "selector"

    def __init__(self, history:AHistory, argumentsDict:Dict[str,object]):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        self._history = history
        self._selector = argumentsDict[self.ARG_SELECTOR]


    def update(self, ratingsUpdateDF:DataFrame, argumentsDict:Dict[str,object]):
        pass


    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # modelDF:pd.DataFrame[methodID:String, r:int, n:int, alpha0:int, beta0:int], numberOfItems:int
    def run(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems=20):
        if type(methodsResultDict) is not dict:
            raise ValueError("Argument methodsResultDict isn't type dict.")
        if type(modelDF) is not DataFrame:
            raise ValueError("Argument modelDF isn't type DataFrame.")
        if type(numberOfItems) is not int:
            raise ValueError("Argument numberOfItems isn't type int.")

        result:List[tuple] = self.runWithResponsibility(methodsResultDict, modelDF, numberOfItems)

        return list(map(lambda itemWithResponsibilityI: itemWithResponsibilityI[0], result))


    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # modelDF:DataFrame<(methodID:str, votes:int)>, numberOfItems:int
    def runWithResponsibility(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems:int=20):
        #print("userID: " + str(userID))

        if type(methodsResultDict) is not dict:
            raise ValueError("Argument methodsResultDict isn't type dict.")
        if type(modelDF) is not DataFrame:
            raise ValueError("Argument modelDF isn't type DataFrame.")
        if type(numberOfItems) is not int:
            raise ValueError("Argument numberOfItems isn't type int.")

        if sorted([mI for mI in modelDF.index]) != sorted([mI for mI in methodsResultDict.keys()]):
            raise ValueError("Arguments methodsResultDict and methodsParamsDF have to define the same methods.")
        for methIdI in methodsResultDict.keys():
            if modelDF.loc[methIdI] is None:
                raise ValueError("Argument methodsParamsDF contains in ome method an empty list of items.")
        if numberOfItems < 0:
            raise ValueError("Argument numberOfItems can't contain negative value.")

        methodsResultDictI:dict = methodsResultDict
        methodsParamsDFI:DataFrame = modelDF

        recommendedItemIDs: List[tuple(int, str)] = []

        for iIndex in range(0, numberOfItems):
            # print("indexI: ", iIndex)
            # print(methodsResultDictI)
            # print(methodsParamsDFI)

            if len([mI for mI in methodsResultDictI]) == 0:
                return recommendedItemIDs[:numberOfItems]

            methodProbabilitiesDicI: dict = {}

            # computing probabilities of methods
            for mIndex in methodsParamsDFI.index:
                # print("mIndexI: ", mIndex)
                methodI = methodsParamsDFI.loc[methodsParamsDFI.index == mIndex]  # .iloc[0]
                # alpha + number of successes, beta + number of failures
                pI = beta(methodI.alpha0 + methodI.r, methodI.beta0 + (methodI.n - methodI.r), size=1)[0]
                methodProbabilitiesDicI[mIndex] = pI
            # print(methodProbabilitiesDicI)

            # get max probability of method prpabilities
            maxPorbablJ: float = max(methodProbabilitiesDicI.values())
            # print("MaxPorbablJ: ", maxPorbablJ)

            # selecting method with highest probability
            theBestMethodID: str = random.choice(
                [aI for aI in methodProbabilitiesDicI.keys() if methodProbabilitiesDicI[aI] == maxPorbablJ])

            # extractiion results of selected method (method with highest probability)
            resultsOfMethodI:Series = methodsResultDictI.get(theBestMethodID)
            #print(resultsOfMethodI)

            resultsOfMethodDictI:dict = dict([(itemIDI, votesI) for itemIDI, votesI in resultsOfMethodI.items()])
            #print(resultsOfMethodDictI)

            # select next item (itemID)
            # selectedItemI:int = AggrBanditTS.selectorOfRouletteWheelRatedItem(resultsOfMethodI)
            selectedItemI:int = self._selector.select(resultsOfMethodDictI)

            # print("SelectedItemI: ", selectedItemI)

            recommendedItemIDs.append((selectedItemI, theBestMethodID))

            # deleting selected element from method results
            for mrI in methodsResultDictI:
                try:
                    methodsResultDictI[mrI].drop(selectedItemI, inplace=True, errors="ignore")
                except:
                    # TODO some error recordings?
                    pass
            # methodsResultDictI = {mrI:methodsResultDictI[mrI].append(pd.Series([None],[selectedItemI])).drop(selectedItemI) for mrI in methodsResultDictI}
            # print(methodsResultDictI)

            # methods with empty list of items
            methodEmptyI = [mI for mI in methodsResultDictI if len(methodsResultDictI.get(mI)) == 0]

            # removing methods with the empty list of items
            methodsParamsDFI = methodsParamsDFI[~methodsParamsDFI.index.isin(methodEmptyI)]

            # removing methods definition with the empty list of items
            for meI in methodEmptyI: methodsResultDictI.pop(meI)
        return recommendedItemIDs[:numberOfItems]





    # selectors definition

    # resultOfMethod:pd.Series([raitings],[itemIDs])
    @staticmethod
    def selectorOfTheMostRatedItem(resultOfMethod:Series):
      maxValue:float = max(resultOfMethod.values)
      return resultOfMethod[resultOfMethod == maxValue].index[0]

    # resultOfMethod:pd.Series([raitings],[itemIDs])
    @staticmethod
    def selectorOfTheFirstItem(resultOfMethod:Series):
      return resultOfMethod.index[0]

    # resultOfMethod:pd.Series([raitings],[itemIDs])
    @staticmethod
    def selectorOfRandomItem(resultOfMethod:Series):
      return random.choice(resultOfMethod.index)

    # resultOfMethod:pd.Series([raitings],[itemIDs])
    @staticmethod
    def selectorOfRouletteWheelRatedItem(resultOfMethod:Series):
        return RouletteWheelSelector.run(resultOfMethod)
