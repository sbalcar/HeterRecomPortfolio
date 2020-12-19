#!/usr/bin/python3

import numpy as np
import pandas as pd
import math

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
        self._listOfPreviousRecommendations = None
        self._contextDim:int = 10
        self._b:dict = None
        self._A:dict = None
        self._context = None
        self._inverseA:dict = None
        self._INVERSE_CALCULATION_THRESHOLD: int = 100
        self._inverseCounter = 101


    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # modelDF:pd.DataFrame[numberOfVotes:int], numberOfItems:int
    def run(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems:int=20):
        # TODO: CHECK DATA INTEGRITY!

        # initialize A and b if not done already
        if self._b is None or self._A is None or self._context is None:
            self._b: dict = {}
            self._A: dict = {}
            self._inverseA = {}
            self._context = self.__calculateContext()
            for recommender, value in methodsResultDict.items():
                self._b[recommender] = np.zeros(self._contextDim)
                self._A[recommender] = np.identity(self._contextDim)
                self._inverseA[recommender] = np.identity(self._contextDim)

        # else update b's
        else:
            # TODO: How to get ListOfClickedItems from previous reccommendation? (Ask Stepan!)
            ListOfClickedItems = [7]
            if self._listOfPreviousRecommendations is None:
                raise ValueError("self._listOfPreviousRecommendations has to contain previous recommendations!")
            dictOfRewards:dict = {}
            counter = 0
            for recommender, value in methodsResultDict.items():
                # TODO: Is performance OK here? This is just sketch
                succesfulRecomendations = len(list(set(ListOfClickedItems).intersection(self._listOfPreviousRecommendations)))
                dictOfRewards[recommender] = succesfulRecomendations
                counter += 1
                # TODO: I don't take into account previous relevance or previous votes, maybe needs to be improved in future

            # update b's
            for recommender, value in self._b.items():
                reward = dictOfRewards[recommender] / counter
                self._b[recommender] += reward * self._context

        # update recommender's votes
        for recommender, votes in modelDF.iterrows():

            # Calculate change rate
            ridgeRegression = self._inverseA[recommender].dot(self._b[recommender])
            UCB = self._context.T.dot(self._inverseA[recommender]).dot(self._context)
            change_rate = ridgeRegression.T.dot(self._context) + math.sqrt(UCB)

            # update votes
            modelDF.at[recommender, 'votes'] = change_rate * votes['votes']

        itemsWithResposibilityOfRecommenders:List[int,np.Series[int,str]] =\
            super().run(methodsResultDict, modelDF, userID, numberOfItems=numberOfItems)

        # update A's
        for recommender, value in self._A.items():

            # get relevance of items, which were recommended by recommender and are in itemsWithResposibilityOfRecommenders
            relevanceSum = 0
            for recommendedItemID in itemsWithResposibilityOfRecommenders:
                if recommendedItemID in methodsResultDict[recommender].index:
                    relevanceSum += methodsResultDict[recommender][recommendedItemID]
            self._A[recommender] += self._context.dot(self._context.T)

        # recompute inverse A's if threshold is hit
        if self._inverseCounter > self._INVERSE_CALCULATION_THRESHOLD:
            for recommender, value in self._inverseA.items():
                self._inverseA[recommender] = np.linalg.inv(self._A[recommender])
            self._inverseCounter = 0
        self._inverseCounter += 1
        self._listOfPreviousRecommendations = itemsWithResposibilityOfRecommenders
        return itemsWithResposibilityOfRecommenders

    def __calculateContext(self):
        # TODO: get context
        return np.zeros(self._contextDim)

    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # modelDF:DataFrame<(methodID:str, votes:int)>, numberOfItems:int
    def runWithResponsibility(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems:int=20):
        # TODO: CHECK DATA INTEGRITY!
        # methodsResultNewDict: dict[str, pd.Series] = self._penaltyTool.runPenalization(
        #        userID, methodsResultDict, self._history)

        itemsWithResposibilityOfRecommenders:List[int,Series[int,str]] = super().runWithResponsibility(
            methodsResultDict, modelDF, userID, numberOfItems=numberOfItems)

        return itemsWithResposibilityOfRecommenders
