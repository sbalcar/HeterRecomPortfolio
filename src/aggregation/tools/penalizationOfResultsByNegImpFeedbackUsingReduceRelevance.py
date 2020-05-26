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

class PenalizationOfResultsByNegImpFeedbackUsingReduceRelevance:

    def __init__(self, history:AHistory, maxPenaltyValue:float, minPenaltyValue:float, lengthOfHistory:int):
        if not isinstance(history, AHistory):
            raise ValueError("Type of history isn't AHistory.")
        if type(maxPenaltyValue) is not float and type(maxPenaltyValue) is not int:
            raise ValueError("Type of maxPenaltyValue isn't float/int.")
        if type(minPenaltyValue) is not float and type(minPenaltyValue) is not int:
            raise ValueError("Type of minPenaltyValue isn't float/int.")
        if type(lengthOfHistory) is not float and type(lengthOfHistory) is not int:
            raise ValueError("Type of lengthOfHistory isn't float/int.")

        self._history = history
        self._maxPenaltyValue:float = maxPenaltyValue
        self._minPenaltyValue:float = minPenaltyValue
        self._lengthOfHistory:int = lengthOfHistory

    # methodsResultDict:dict[int,Series[int,str]]
    def proportionalRelevanceReduction(self, methodsResultDict:dict, userID:int):

        itemIDs:List[int] = list(set(itertools.chain(*[rI.index for mI, rI in methodsResultDict.items()])))
        #print("itemIDs: " + str(itemIDs))

        penalties:dict = {}
        for itemIdI in itemIDs:
            prevRecomendations:List[tuple] = self._history.getPreviousRecomOfUserAndItem(userID, itemIdI, self._lengthOfHistory)
            prevRecomendations.reverse()

            penaltyI:float = 0
            i:int = 0
            for indexJ, userIdJ, itemIdJ, positionJ, observationJ, clickedJ, timestampJ in prevRecomendations:
                penaltyPositionJ:float = PenalizationOfResultsByNegImpFeedbackUsingReduceRelevance.getPenaltyPosition(positionJ, 1)

                penaltyLinear:float = PenalizationOfResultsByNegImpFeedbackUsingReduceRelevance.getPenaltyLinear2(
                    i, self._maxPenaltyValue, self._minPenaltyValue, self._lengthOfHistory)

                penaltyI += penaltyPositionJ * penaltyLinear
                i += 1
            penalties[itemIdI] = penaltyI


        penalizedResultsDict:dict = {}

        methodIdI:str
        resulrsI:Series
        for methodIdI, resultsI in methodsResultDict.items():
            penalizedResultDictI:dict = {}

            candidateIdJ:int
            votesOfCandidateJ:int
            for candidateIdJ, votesOfCandidateJ in resultsI.items():
                #print("candidateIdJ: " + str(candidateIdJ))
                #print("votesOfCandidateJ: " + str(votesOfCandidateJ))

                penalizedResultDictI[candidateIdJ] = votesOfCandidateJ / (1 + penalties.get(candidateIdJ, 0))

            penalizedResultsDict[methodIdI] = Series(penalizedResultDictI, name="rating")

        return penalizedResultsDict



    @staticmethod
    def getPenaltyPosition(position:int, maxPenalty:float):
        """
        Simpler version of penalty for 1/1+p position discount
        """
        return maxPenalty / (1 + position)

    @staticmethod
    def getPenaltyLinear2(xDistanceInHistory:int, maxPenaltyValue:float, minPenaltyValue:float, lengthOfHistory:int):
        if (xDistanceInHistory < 0):
            raise ValueError("xDistanceInHistory must be greater than zero")
        if (maxPenaltyValue < minPenaltyValue):
            raise ValueError("MaxPenaltyValue must be greater than or equal to minPenaltyValue")

        if xDistanceInHistory > lengthOfHistory:
            return 0
        slope:float = (maxPenaltyValue -minPenaltyValue) / lengthOfHistory
        return maxPenaltyValue -slope*xDistanceInHistory


    @staticmethod
    def getPenaltyLinear(timeDiff:float, minTimeDiff:float, maxTimeDiff:float, minPenalty:float, maxPenalty:float):
        '''
        computes linear penalty based on a line equation
        y = m*x + c   ~   penalty = m*timeDiff + c
        the line is defined by two points:
        (minTimeDiff, maxPenalty), (maxTimeDiff, minPenalty)
        :param timeDiff: time since the recommendation (in seconds)
        :param minTimeDiff: penalty starts to decrease after this time
        :param maxTimeDiff: penalty remains minimal after this time
        :param minPenalty: minimal penalty given (for timeDiff >= maxTimeDiff)
        :param maxPenalty: maximal penalty given (for timeDiff <= minTimeDiff)
        :return: computed penalty
        '''
        if (timeDiff <= 0):
            raise ValueError("timeDiff must not be negative")
        if (minTimeDiff < 0):
            raise ValueError("minTimeDiff must not be negative")
        if (maxTimeDiff <= minTimeDiff):
            raise ValueError("maxTimeDiff must be greater than minTimeDiff")
        if (minPenalty < 0):
            raise ValueError("minPenalty must not be negative")
        if (maxPenalty < minPenalty):
            raise ValueError("maxPenalty must be greater than or equal to minPenalty")

        m = (minPenalty - maxPenalty) / (maxTimeDiff - minTimeDiff)
        c = maxPenalty - (m * maxTimeDiff)
        res = (m * timeDiff) + c
        if res < minPenalty:
            res = minPenalty
        elif res > maxPenalty:
            res = maxPenalty

        return res