#!/usr/bin/python3
import random

import numpy as np
import pandas as pd

from numpy.random import beta
from typing import List

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from aggregation.aAggregation import AAgregation #class
from aggregation.aggrDHont import AggrDHont #class

from aggregation.tools.penalizationOfResultsByNegImpFeedbackUsingFiltering import PenalizationOfResultsByNegImpFeedbackUsingFiltering #class
from aggregation.tools.penalizationOfResultsByNegImpFeedbackUsingReduceRelevance import PenalizationOfResultsByNegImpFeedbackUsingReduceRelevance #class

from history.aHistory import AHistory #class
from abc import ABC, abstractmethod

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class


class AggrDHontNegativeImplFeedback(AggrDHont):

    ARG_SELECTORFNC:str = "selectorFnc"

    ARG_MAX_PENALTY_VALUE = "maxPenaltyValue"
    ARG_MIN_PENALTY_VALUE = "minPenaltyValue"
    ARG_LENGTH_OF_HISTORY:str = "lengthOfHistory"

    def __init__(self, history:AHistory, argumentsDict:dict):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        self._history = history
        self.argumentsDict:dict = argumentsDict.copy()

        self._selectorFnc = argumentsDict[self.ARG_SELECTORFNC][0]
        self._selectorArg = argumentsDict[self.ARG_SELECTORFNC][1]
        self._maxPenaltyValue:float = argumentsDict[self.ARG_MAX_PENALTY_VALUE]
        self._minPenaltyValue:float = argumentsDict[self.ARG_MIN_PENALTY_VALUE]
        self._lengthOfHistory:int = argumentsDict[self.ARG_LENGTH_OF_HISTORY]

    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # modelDF:pd.DataFrame[numberOfVotes:int], numberOfItems:int
    def run(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems:int=20):

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

        p = PenalizationOfResultsByNegImpFeedbackUsingReduceRelevance(
            self._history, self._maxPenaltyValue, self._minPenaltyValue, self._lengthOfHistory)
        methodsResultNewDict:dict[str, pd.Series] = p.proportionalRelevanceReduction(methodsResultDict, userID)

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

        p = PenalizationOfResultsByNegImpFeedbackUsingReduceRelevance(
            self._history, self._maxPenaltyValue, self._minPenaltyValue, self._lengthOfHistory)
        methodsResultNewDict:dict[str, pd.Series] = p.proportionalRelevanceReduction(methodsResultDict, userID)

        #methodsResultNewDict:List[int,Series[int,str]] = PenalizationOfResultsByNegImpFeedbackUsingReduceRelevance\
        #    .proportionalRelevanceReduction(methodsResultDict, self._history, userID, self._lengthOfHistory)

        itemsWithResposibilityOfRecommenders:List[int,Series[int,str]] = super().runWithResponsibility(
            methodsResultNewDict, modelDF, userID, numberOfItems=numberOfItems)

        # list<(itemID:int, Series<(rating:int, methodID:str)>)>
        return itemsWithResposibilityOfRecommenders
