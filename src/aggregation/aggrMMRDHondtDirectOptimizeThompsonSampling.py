#!/usr/bin/python3

import numpy as np
import pandas as pd

from typing import List
from typing import Dict #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from aggregation.aggrWeightedAVG import AggrWeightedAVG #class

from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt #class
from aggregation.aggrFuzzyDHondtINF import AggrFuzzyDHondtINF #class
from aggregation.aggrDHondtDirectOptimizeThompsonSampling import AggrDHondtDirectOptimizeThompsonSampling #class

from history.aHistory import AHistory #class
from recommender.tools.toolMMR import ToolMMR #class

from datasets.aDataset import ADataset  #class
from datasets.datasetML import DatasetML  #class
from datasets.datasetRetailrocket import DatasetRetailRocket  # lass
from datasets.datasetST import DatasetST  #class



class AggrMMRDHondtDirectOptimizeThompsonSampling(AggrDHondtDirectOptimizeThompsonSampling):

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
        self._lastRecommendedPerUser = {}


    def train(self, history:AHistory, dataset:ADataset):
        self.toolMMR = ToolMMR()   
        self.toolMMR.init(dataset)


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
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        items = super().runWithScore(methodsResultDict, modelDF, userID, numberOfItems=numberOfItems*5, argumentsDict=argumentsDict)

        previousRecs = self._lastRecommendedPerUser.get(userID, [])
        results = self.toolMMR.mmr_sorted_with_prefix(0.5, items, previousRecs, numberOfItems)
        selectedItems = results.index.tolist()
        self._lastRecommendedPerUser[userID] = selectedItems
    
        # list<(itemID:int, Series<(rating:int, methodID:str)>)>
        return selectedItems


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
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        aggregatedItemIDs:List[int] = self.run(methodsResultDict, modelDF, userID, numberOfItems)
        itemsWithResposibilityOfRecommenders = []
        for item in  aggregatedItemIDs:
            votesOfItemDictI:dict[str,float] = {mI:methodsResultDict[mI].get(key = item, default = 0) for mI in modelDF.index}
            itemsWithResposibilityOfRecommenders.append((item, votesOfItemDictI))


        # list<(itemID:int, Series<(rating:int, methodID:str)>)>
        return itemsWithResposibilityOfRecommenders
