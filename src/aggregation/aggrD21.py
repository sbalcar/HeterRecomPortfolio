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
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class

from history.aHistory import AHistory #class
from abc import ABC, abstractmethod

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class


class AggrD21(AAgregation):

    ARG_RATING_THRESHOLD_FOR_NEG:str = "ratingThresholdForNeg"

    def __init__(self, history:AHistory, argumentsDict:Dict[str,object]):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")
        np.random.seed(42)

        self._history = history
        self._ratingThresholdForNeg:float = argumentsDict[self.ARG_RATING_THRESHOLD_FOR_NEG]


    def update(self, ratingsUpdateDF:DataFrame, argumentsDict:Dict[str,object]):
        pass


    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # modelDF:pd.DataFrame[methodID:String, r:int, n:int, alpha0:int, beta0:int], numberOfItems:int
    def run(self, methodsResultDict:dict, modelDF: DataFrame, userID: int, numberOfItems,
            argumentsDict: Dict[str, object] = {}):
        if type(methodsResultDict) is not dict:
            raise ValueError("Argument methodsResultDict isn't type dict.")
        if type(modelDF) is not DataFrame:
            raise ValueError("Argument modelDF isn't type DataFrame.")
        if type(numberOfItems) is not int:
            raise ValueError("Argument numberOfItems isn't type int.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        result:List[tuple] = self.runWithResponsibility(methodsResultDict, modelDF, numberOfItems)

        return list(map(lambda itemWithResponsibilityI: itemWithResponsibilityI[0], result))


    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # modelDF:DataFrame<(methodID:str, votes:int)>, numberOfItems:int
    def runWithResponsibility(self, inputItemIDsDict:dict, modelDF:DataFrame, userID:int, numberOfItems:int,
                              argumentsDict:Dict[str, object] = {}):
        print(inputItemIDsDict)
        if type(inputItemIDsDict) is not dict:
            raise ValueError("Argument inputItemIDsDict isn't type dict.")
        if type(numberOfItems) is not int:
            raise ValueError("Argument numberOfItems isn't type int.")
        if numberOfItems < 0:
            raise ValueError("Argument numberOfItems can't contain negative value.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        if float('nan') in inputItemIDsDict["input2"].keys():
            inputItemIDsDict["input2"].pop(float('nan'))

        inputItemIDs1:List[int] = list(inputItemIDsDict["input1"].keys())
        inputItemIDs2:List[int] = list(inputItemIDsDict["input2"].keys())
        negativeItemIDs:List[int] = list(inputItemIDsDict["negative"].keys())

        allItemIDs:List[int] = list(set(inputItemIDs1 + inputItemIDs2 + negativeItemIDs))
        allItemIDs = [int(x) for x in allItemIDs]

        rInput = [0.0]*len(allItemIDs)
        zipped = list(zip(rInput, rInput, rInput, rInput))
        df = pd.DataFrame(zipped, columns=['rInput1', 'rInput2', 'rNegative', 'rSum'], index=allItemIDs)

        for itemId1I, r1 in inputItemIDsDict["input1"].items():
            df.loc[int(itemId1I), 'rInput1'] = r1
        counter = numberOfItems
        for itemId2I, r2 in inputItemIDsDict["input2"].items():
            df.loc[int(itemId2I), 'rInput2'] = r2 + counter* 0.001 /numberOfItems
            counter = counter -1
        for itemId3I, r3 in inputItemIDsDict["negative"].items():
            df.loc[int(itemId3I), 'rNegative'] = r3

        for rowI in df.iterrows():
            itemIdI:int = int(rowI[0])
            valsI = rowI[1]
            df.loc[itemIdI, 'rSum'] = 0.1*valsI['rInput1'] + valsI['rInput2'] -valsI['rNegative']
            #if df.loc[itemIdI, 'rSum'] > self._ratingThresholdForNeg:
            #    df.loc[itemIdI, 'rSum'] -= valsI['rNegative']

        #df = df[df['rSum'] > 0.0]
        df = df.sort_values('rSum', ascending=False)

        print(df)
        df = df['rSum']
        print(df)

        return df.head(numberOfItems)
