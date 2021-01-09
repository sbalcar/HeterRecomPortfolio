#!/usr/bin/python3

import numpy as np
import math

from typing import List

from pandas.core.frame import DataFrame  # class
from pandas.core.series import Series  # class

from aggregation.aAggregation import AAgregation # class

from history.aHistory import AHistory  # class



class MixinContextAggregation(AAgregation):

    ARG_EVAL_TOOL = 'eTool'
    def __init__(self, history: AHistory, argumentsDict: dict):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")
        self.eTool = argumentsDict[self.ARG_EVAL_TOOL]


    def run(self, methodsResultDict: dict, modelDF: DataFrame, userID: int, numberOfItems: int = 20):

        self._context = self.eTool.calculateContext(userID)

        # update recommender's votes
        updatedVotes = dict()
        totalUpdatedVotes = 0
        for recommender, votes in modelDF.iterrows():
            # Calculate change rate
            ridgeRegression = self.eTool._inverseA[recommender].dot(self.eTool._b[recommender])
            UCB_secondpart = 0.01 * self._context.T.dot(self.eTool._inverseA[recommender]).dot(self._context)
            UCB = (ridgeRegression.T.dot(self.eTool._context) + math.sqrt(UCB_secondpart))
            print("UCB: ", UCB)
            # update votes
            updatedVotes[recommender] = UCB
            totalUpdatedVotes += updatedVotes[recommender]

        for recommender, votes in modelDF.iterrows():
            modelDF.at[recommender, 'votes'] = updatedVotes[recommender] / totalUpdatedVotes

        itemsWithResposibilityOfRecommenders: List[int, np.Series[int, str]] = \
            super().run(methodsResultDict, modelDF, userID, numberOfItems=numberOfItems)
        return itemsWithResposibilityOfRecommenders

    def runWithResponsibility(self, methodsResultDict: dict, modelDF: DataFrame, userID: int, numberOfItems: int = 20):
        # TODO: CHECK DATA INTEGRITY!
        # methodsResultNewDict: dict[str, pd.Series] = self._penaltyTool.runPenalization(
        #        userID, methodsResultDict, self._history)

        itemsWithResposibilityOfRecommenders: List[int, Series[int, str]] = super().runWithResponsibility(
            methodsResultDict, modelDF, userID, numberOfItems=numberOfItems)

        return itemsWithResposibilityOfRecommenders
