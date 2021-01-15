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

from aggregation.negImplFeedback.aPenalization import APenalization #class

from sklearn.preprocessing import normalize


class PenalUsingReduceRelevance(APenalization):

    def __init__(self, penalPositionFnc, argumentsPositionList:List,
                 penalHistoryFnc, argumentsHistoryList:List,
                 lengthOfHistory:int):
        if type(argumentsPositionList) is not list:
            raise ValueError("Type of argumentsPositionList isn't list.")
        if type(argumentsHistoryList) is not list:
            raise ValueError("Type of argumentsHistoryList isn't list.")
        if type(lengthOfHistory) is not int:
            raise ValueError("Argument lengthOfHistory isn't type int.")

        self._penalPositionFnc = penalPositionFnc
        self._argumentsPositionList = argumentsPositionList
        self._penalHistoryFnc = penalHistoryFnc
        self._argumentsHistoryList = argumentsHistoryList

        self._lengthOfHistory = lengthOfHistory

    # methodsResultDict:dict[int,Series[int,str]]
    def runPenalization(self, userID:int, methodsResultDict:dict, history:AHistory):

        if type(userID) is not int and type(userID) is not np.int64:
            print(type(userID))
            raise ValueError("Type of userID isn't int.")
        if type(methodsResultDict) is not dict:
            raise ValueError("Type of methodsResultDict isn't dict.")
        if not isinstance(history, AHistory):
            raise ValueError("Type of history isn't AHistory.")

        itemIDs:List[int] = list(set(itertools.chain(*[rI.index for mI, rI in methodsResultDict.items()])))

        # dictionary of float penalization indexed by itemIDs
        penalties:dict = self.__getPenaltiesOfItemIDs(userID, itemIDs, history)


        penalizedResultsDict:dict = {}

        methodIdI:str
        resultsI:Series
        for methodIdI, resultsI in methodsResultDict.items():
            penalizedResultDictI:dict = {}

            itemIdJ:int
            ratingJ:int
            for itemIdJ, ratingJ in resultsI.items():
                #print("candidateIdJ: " + str(candidateIdJ))
                #print("votesOfCandidateJ: " + str(votesOfCandidateJ))

                penalizedResultDictI[itemIdJ] = ratingJ / (1 + penalties.get(itemIdJ, 0))

            penalizedResultsDict[methodIdI] = Series(penalizedResultDictI, name="rating")

        return penalizedResultsDict


    def runOneMethodPenalization(self, userID:int, methodsResultSrs:Series, history:AHistory):

        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Type of userID isn't int.")
        if type(methodsResultSrs) is not Series:
            raise ValueError("Type of methodsResultSrs isn't Series.")
        if not isinstance(history, AHistory):
            raise ValueError("Type of history isn't AHistory.")

        itemIDs:List[int] = methodsResultSrs.keys()

        # dictionary of float penalization indexed by itemIDs
        penalties:dict = self.__getPenaltiesOfItemIDs(userID, itemIDs, history)

        penalizedRatings:List[float] = []
        itemIdI:int
        ratingI:int
        for itemIdI, ratingI in methodsResultSrs.items():
            penalizedRatings.append(ratingI / (1 + penalties.get(itemIdI, 0)))

        penalizedItemIDswithRatingsSrs:Series = Series(penalizedRatings, index=methodsResultSrs.keys())

        sortedPenalizedItemIDswithRatingsSrs:Series = penalizedItemIDswithRatingsSrs.sort_values(
            axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last')

        normalizedNewMethodsResultSrs:Series = Series(
            normalize(sortedPenalizedItemIDswithRatingsSrs.values[:, np.newaxis], axis=0).ravel(),
            index=sortedPenalizedItemIDswithRatingsSrs.index)

        return normalizedNewMethodsResultSrs
        # return Series<(rating:int, itemID:int)>

    def __getPenaltiesOfItemIDs(self, userID:int, itemIDs:List[int], history:AHistory):
        prevRecomendations:List[tuple] = history.getPreviousRecomOfUser(userID, self._lengthOfHistory)
        i:int = len(prevRecomendations)   #self._lengthOfHistory#
        #print(prevRecomendations)
        penalties:dict = {}
        for indexJ, userIdJ, itemIdJ, positionJ, clickedJ, timestampJ in prevRecomendations:
            actPenalty = penalties.get(itemIdJ, 0) #init with one
            penaltyPositionJ:float = self._penalPositionFnc(positionJ, *self._argumentsPositionList)
            penaltyHistoryJ:float = self._penalHistoryFnc(i, *self._argumentsHistoryList)
            
            
            prob_implicit_rejection:float = self._penalPositionFnc(positionJ, *self._argumentsPositionList)
            prob_stable_pref:float = self._penalHistoryFnc(i, *self._argumentsHistoryList)
            prob_couldBeRelevant = (1-(prob_implicit_rejection * prob_stable_pref)  )

            penalties[itemIdJ]  =  actPenalty + (penaltyPositionJ * penaltyHistoryJ)
            
            i = i-1
        print(penalties)
        return penalties
        
    """    
    def __getPenaltiesOfItemIDs(self, userID:int, itemIDs:List[int], history:AHistory):

        penalties:dict = {}
        for itemIdI in itemIDs:
            prevRecomendations:List[tuple] = history.getPreviousRecomOfUserAndItem(userID, itemIdI, self._lengthOfHistory)
            prevRecomendations.reverse()

            penaltyI: float = 0
            i:int = 0
            #print(prevRecomendations)
            for indexJ, userIdJ, itemIdJ, positionJ, clickedJ, timestampJ in prevRecomendations:
                penaltyPositionJ:float = self._penalPositionFnc(positionJ, *self._argumentsPositionList)

                penaltyHistoryJ:float = self._penalHistoryFnc(indexJ, *self._argumentsHistoryList)

                penaltyI += penaltyPositionJ * penaltyHistoryJ
                i += 1
            penalties[itemIdI] = penaltyI

        return penalties
     """


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