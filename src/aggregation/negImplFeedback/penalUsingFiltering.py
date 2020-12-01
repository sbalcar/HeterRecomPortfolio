#!/usr/bin/python3
import random
import itertools

import numpy as np
import pandas as pd

from numpy.random import beta
from typing import List

from pandas.core.frame import DataFrame  # class
from pandas.core.series import Series  # class

from history.aHistory import AHistory #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from aggregation.negImplFeedback.aPenalization import APenalization #class

from sklearn.preprocessing import normalize


# transforming Results Of D'Hont Methods
class PenalUsingFiltering(APenalization):

    def __init__(self, borderNegFeedback:float, lengthOfHistory:int):
        if type(borderNegFeedback) is not float:
            raise ValueError("Type of borderNegFeedback isn't float.")
        if type(lengthOfHistory) is not int:
            raise ValueError("Type of lengthOfHistory isn't int.")

        self._borderNegFeedback:float = borderNegFeedback
        self._lengthOfHistory:int = lengthOfHistory


    # methodsResultDict:dict[int,Series[int,str]]
    def runPenalization(self, userID:int, methodsResultDict:dict, history:AHistory):

        def getItemIDs(methIdI:str):
            recommI:Series = methodsResultDict[methIdI]
            itemIDsI:List[int] = recommI.index
            return itemIDsI

        itemsInRecomendationsLL:List[List[int]] = map(getItemIDs, methodsResultDict)
        itemsInRecomendations:List[int] = list(itertools.chain.from_iterable(itemsInRecomendationsLL))

        itemIdsToRemove:List[int] = []
        for itemIdI in itemsInRecomendations:
            valueOfIgnoringI:float = history.getIgnoringValue(userID, itemIdI, limit=self._lengthOfHistory)
            #print("valueOfIgnoringI: " + str(valueOfIgnoringI))
            if valueOfIgnoringI >= self._borderNegFeedback:
                itemIdsToRemove.append(itemIdI)

        #print("itemIdsToRemove: " + str(itemIdsToRemove))

        methodsResultNewDict:dict = {}
        for methIdI in methodsResultDict.keys():
            ratingsI:Series = methodsResultDict[methIdI]
            ratingsNewI:Series = ratingsI.drop(list(set(ratingsI.keys()) & set(itemIdsToRemove)))
            methodsResultNewDict[methIdI] = ratingsNewI

        return methodsResultNewDict



    def runOneMethodPenalization(self, userID:int, methodsResultSrs:Series, history:AHistory):

        itemIdsToRemove:List[int] = []
        for itemIdI in methodsResultSrs.index:
            #print(itemIdI)
            valueOfIgnoringI:float = history.getIgnoringValue(userID, itemIdI, limit=self._lengthOfHistory)
            #print("valueOfIgnoringI: " + str(valueOfIgnoringI))
            if valueOfIgnoringI >= self._borderNegFeedback:
                itemIdsToRemove.append(itemIdI)

        print("borderNegFeedback " + str(self._borderNegFeedback))
        print(itemIdsToRemove)

        newMethodsResultSrs:Series = methodsResultSrs.drop(itemIdsToRemove)
        #print(newMethodsResultSrs)
        #print(normalize(newMethodsResultSrs.values[:,np.newaxis], axis=0).ravel())

        normalizedNewMethodsResultSrs:Series = Series(
            normalize(newMethodsResultSrs.values[:, np.newaxis], axis=0).ravel(),
            index=newMethodsResultSrs.index)

        return normalizedNewMethodsResultSrs
