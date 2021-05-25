#!/usr/bin/python3
import random

import numpy as np
import pandas as pd

from numpy.random import beta
from typing import List
from typing import Dict  # class

from pandas.core.frame import DataFrame  # class
from pandas.core.series import Series  # class

from aggregation.aAggregation import AAgregation  # class
from aggregation.tools.responsibilityDHont import countDHontResponsibility  # function
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector  # class

from history.aHistory import AHistory  # class
from abc import ABC, abstractmethod

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription  # class


class AggrFAI(AAgregation):
    DEBUG: bool = False

    def __init__(self, history: AHistory, argumentsDict: Dict[str, object]):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")
        np.random.seed(42)
        self._history = history

    def update(self, ratingsUpdateDF: DataFrame, argumentsDict: Dict[str, object]):
        pass

    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # modelDF:pd.DataFrame[methodID:String, r:int, n:int, alpha0:int, beta0:int], numberOfItems:int
    def run(self, methodsResultDict: dict, modelDF: DataFrame, userID: int, numberOfItems,
            argumentsDict: Dict[str, object] = {}):
        if type(methodsResultDict) is not dict:
            raise ValueError("Argument methodsResultDict isn't type dict.")
        if type(modelDF) is not DataFrame:
            raise ValueError("Argument modelDF isn't type DataFrame.")
        if type(numberOfItems) is not int:
            raise ValueError("Argument numberOfItems isn't type int.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        result: List[tuple] = self.runWithResponsibility(methodsResultDict, modelDF, numberOfItems)

        return list(map(lambda itemWithResponsibilityI: itemWithResponsibilityI[0], result))

    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # modelDF:DataFrame<(methodID:str, votes:int)>, numberOfItems:int
    def runWithResponsibility(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems:int,
                              argumentsDict:Dict[str, object] = {}):
        # print("userID: " + str(userID))

        if type(methodsResultDict) is not dict:
            raise ValueError("Argument methodsResultDict isn't type dict.")
        if type(numberOfItems) is not int:
            raise ValueError("Argument numberOfItems isn't type int.")
        if numberOfItems < 0:
            raise ValueError("Argument numberOfItems can't contain negative value.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        methodsResultDictI:dict = methodsResultDict
        methodsParamsDFI:DataFrame = modelDF

        availableRecommenders = [i for i in methodsResultDictI.keys() if len(methodsResultDictI[i]) > 0]
        availableRecommenders = np.random.permutation(availableRecommenders)
        sequenceOfRecommenders = list(availableRecommenders) + list(availableRecommenders[::-1])

        lastIndex = {}
        resList = []
        results = []
        for rec in availableRecommenders:
            lastIndex[rec] = -1
            methodsResultDictI[rec].sort_values(ascending=False, inplace=True)

        for i in range(numberOfItems):
            recommenderID = sequenceOfRecommenders[i % len(sequenceOfRecommenders)]

            (item, itemPos) = self.getNextItem(methodsResultDictI, lastIndex, recommenderID, resList)
            if item != -1:  # check for valid itemID
                lastIndex[recommenderID] = itemPos
                resList.append(item)
                results.append((item, recommenderID))

        return results

    def getNextItem(self, methodsResultDictI, lastIndex, recommenderID, resList):
        # print(methodsResultDictI[recommenderID])
        itemPos = lastIndex[recommenderID] + 1
        item = methodsResultDictI[recommenderID].index[itemPos]

        while (item in resList):
            itemPos = itemPos + 1
            if itemPos >= len(methodsResultDictI[recommenderID]):
                return (-1, -1)  # no more valid items to return

            item = methodsResultDictI[recommenderID].index[itemPos]

        return (item, itemPos)