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

from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.aPenalization import APenalization #class


class PenalUsingReduceRelevance(APenalization):

    def __init__(self, penalPositionFnc, argumentsPositionList:List, penalHistoryFnc, argumentsHistoryList:List):
        if type(argumentsPositionList) is not list:
            raise ValueError("Type of argumentsPositionList isn't list.")
        if type(argumentsHistoryList) is not list:
            raise ValueError("Type of argumentsHistoryList isn't list.")

        self._penalPositionFnc = penalPositionFnc
        self._argumentsPositionList = argumentsPositionList
        self._penalHistoryFnc = penalHistoryFnc
        self._argumentsHistoryList = argumentsHistoryList

    # methodsResultDict:dict[int,Series[int,str]]
    def runPenalization(self, userID:int, methodsResultDict:dict, history:AHistory):

        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Type of userID isn't int.")
        if type(methodsResultDict) is not dict:
            raise ValueError("Type of methodsResultDict isn't dict.")
        if not isinstance(history, AHistory):
            raise ValueError("Type of history isn't AHistory.")

        itemIDs: List[int] = list(set(itertools.chain(*[rI.index for mI, rI in methodsResultDict.items()])))

        penalties: dict = {}
        for itemIdI in itemIDs:
            prevRecomendations: List[tuple] = history.getPreviousRecomOfUserAndItem(userID, itemIdI, 100)
            prevRecomendations.reverse()

            penaltyI: float = 0
            i: int = 0
            for indexJ, userIdJ, itemIdJ, positionJ, clickedJ, timestampJ in prevRecomendations:
                penaltyPositionJ:float = self._penalPositionFnc(positionJ, *self._argumentsPositionList)

                penaltyHistoryJ:float = self._penalHistoryFnc(i, *self._argumentsHistoryList)

                penaltyI += penaltyPositionJ * penaltyHistoryJ
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




def penaltyStatic(xDistance:int, value:float):
    return value


def penaltyLinear(xDistance:int, maxPenaltyValue:float, minPenaltyValue:float, lengthOfInterval:int):
    if (xDistance < 0):
        raise ValueError("xDistance must be greater than zero")
    if (maxPenaltyValue < minPenaltyValue):
        raise ValueError("maxPenaltyValue must be greater than or equal to minPenaltyValue")

    if xDistance > lengthOfInterval:
        return 0
    slope: float = (maxPenaltyValue - minPenaltyValue) / lengthOfInterval
    return maxPenaltyValue - slope * xDistance





def __getPenaltyPosition(position:int, maxPenalty:float):
    """
    Simpler version of penalty for 1/1+p position discount
    """
    return maxPenalty / (1 + position)


def __getPenaltyLinear(timeDiff:float, minTimeDiff:float, maxTimeDiff:float, minPenalty:float, maxPenalty:float):
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