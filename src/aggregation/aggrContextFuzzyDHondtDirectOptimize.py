#!/usr/bin/python3

from typing import List
from typing import Dict  # class

import numpy as np

from pandas.core.frame import DataFrame  # class
from pandas.core.series import Series  # class

from aggregation.aggrContextFuzzyDHondtINF import AggrContextFuzzyDHondtINF #class
from aggregation.aggrFuzzyDHondtDirectOptimize import AggrFuzzyDHondtDirectOptimize  # class
from aggregation.mixinContextAggregation import MixinContextAggregation  # class
from aggregation.aAggregation import AAgregation  # class

from history.aHistory import AHistory  # class


class AggrContextFuzzyDHondtDirectOptimize(MixinContextAggregation, AggrFuzzyDHondtDirectOptimize, AAgregation):

    ARG_EVAL_TOOL:str = AggrContextFuzzyDHondtINF.ARG_EVAL_TOOL

    def __init__(self, history: AHistory, argumentsDict: Dict[str, object]):
        AggrFuzzyDHondtDirectOptimize.__init__(self, history, argumentsDict)
        MixinContextAggregation.__init__(self, history, argumentsDict)

    def run(self, methodsResultDict: dict, modelDF: DataFrame, userID: int, numberOfItems: int = 20,
            argumentsDict: Dict[str, object] = {}):
        itemsWithResposibilityOfRecommenders: List[int, np.Series[int, str]] = \
            super().run(methodsResultDict, modelDF, userID, numberOfItems, argumentsDict=argumentsDict)
        return itemsWithResposibilityOfRecommenders

    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # modelDF:DataFrame<(methodID:str, votes:int)>, numberOfItems:int
    def runWithResponsibility(self, methodsResultDict: dict, modelDF: DataFrame, userID: int, numberOfItems: int = 20,
                              argumentsDict: Dict[str, object] = {}):
        # print(argumentsDict)

        itemsWithResposibilityOfRecommenders: List[int, Series[int, str]] = super() \
            .runWithResponsibility(methodsResultDict, modelDF, userID, numberOfItems, argumentsDict=argumentsDict)
        return itemsWithResposibilityOfRecommenders