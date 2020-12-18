#!/usr/bin/python3

import numpy as np
import pandas as pd

from typing import List

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt #class

from history.aHistory import AHistory #class


class AggrContextFuzzyDHondt(AggrFuzzyDHondt):

    ARG_SELECTOR:str = "selector"

    def __init__(self, history:AHistory, argumentsDict:dict):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        self._selector = argumentsDict[self.ARG_SELECTOR]
        self._history = history

    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # modelDF:pd.DataFrame[numberOfVotes:int], numberOfItems:int
    def run(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems:int=20):
        # TODO: CHECK DATA INTEGRITY!

        # methodsResultNewDict: dict[str, pd.Series] = self._penaltyTool.runPenalization(
        #         userID, methodsResultDict, self._history)

        itemsWithResposibilityOfRecommenders:List[int,np.Series[int,str]] =\
            super().run(methodsResultDict, modelDF, userID, numberOfItems=numberOfItems)

        # list<(itemID:int, Series<(rating:int, methodID:str)>)>
        return itemsWithResposibilityOfRecommenders


    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # modelDF:DataFrame<(methodID:str, votes:int)>, numberOfItems:int
    def runWithResponsibility(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems:int=20):
        # TODO: CHECK DATA INTEGRITY!
        # methodsResultNewDict: dict[str, pd.Series] = self._penaltyTool.runPenalization(
        #        userID, methodsResultDict, self._history)

        itemsWithResposibilityOfRecommenders:List[int,Series[int,str]] = super().runWithResponsibility(
            methodsResultDict, modelDF, userID, numberOfItems=numberOfItems)

        return itemsWithResposibilityOfRecommenders
