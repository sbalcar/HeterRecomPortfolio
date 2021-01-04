#!/usr/bin/python3

import numpy as np
import pandas as pd

from typing import List

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt #class
from aggregation.aggrFuzzyDHondtINF import AggrFuzzyDHondtINF #class
from aggregation.aggrDHondtThompsonSampling import AggrDHondtThompsonSampling #class

from history.aHistory import AHistory #class


class AggrDHondtThompsonSamplingINF(AggrDHondtThompsonSampling):

    ARG_SELECTOR:str = "selector"
    ARG_PENALTY_TOOL:str = "penaltyTool"

    def __init__(self, history:AHistory, argumentsDict:dict):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        self._history = history
        self.argumentsDict:dict = argumentsDict.copy()

        self._selector = argumentsDict[self.ARG_SELECTOR]
        self._penaltyTool = argumentsDict[self.ARG_PENALTY_TOOL]


    def update(self, updtType:str, ratingsUpdateDF:DataFrame):
        pass


    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # modelDF:pd.DataFrame[numberOfVotes:int], numberOfItems:int
    def run(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems:int=20):

        # testing types of parameters
        if type(methodsResultDict) is not dict:
            raise ValueError("Type of methodsResultDict isn't dict.")
        if type(modelDF) is not DataFrame:
            raise ValueError("Type of methodsParamsDF isn't DataFrame.")
        if list(modelDF.columns) != ['r', 'n', 'alpha0', 'beta0']:
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

        methodsResultNewDict: dict[str, pd.Series] = self._penaltyTool.runPenalization(
                userID, methodsResultDict, self._history)

        itemsWithResposibilityOfRecommenders:List[int,np.Series[int,str]] =\
            super().run(methodsResultNewDict, modelDF, userID, numberOfItems=numberOfItems)

        # list<(itemID:int, Series<(rating:int, methodID:str)>)>
        return itemsWithResposibilityOfRecommenders


    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # modelDF:DataFrame<(methodID:str, votes:int)>, numberOfItems:int
    def runWithResponsibility(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems:int=20):

        # testing types of parameters
        if type(methodsResultDict) is not dict:
            raise ValueError("Type of methodsResultDict isn't dict.")
        for methI in methodsResultDict.values():
            if type(methI) is not pd.Series:
                raise ValueError("Type of methodsParamsDF doen't contain Series.")
        if type(modelDF) is not DataFrame:
            raise ValueError("Type of methodsParamsDF isn't DataFrame.")
        if list(modelDF.columns) != ['r', 'n', 'alpha0', 'beta0']:
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

        methodsResultNewDict: dict[str, pd.Series] = self._penaltyTool.runPenalization(
                userID, methodsResultDict, self._history)

        itemsWithResposibilityOfRecommenders:List[int,Series[int,str]] = super().runWithResponsibility(
            methodsResultNewDict, modelDF, userID, numberOfItems=numberOfItems)

        # list<(itemID:int, Series<(rating:int, methodID:str)>)>
        return itemsWithResposibilityOfRecommenders
