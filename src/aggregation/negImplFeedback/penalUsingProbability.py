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


class PenalUsingProbability(APenalization):

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
        prob_couldBeRelevant:dict = self.__getPenaltiesOfItemIDs(userID, itemIDs, history)


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

                penalizedResultDictI[itemIdJ] = ratingJ * prob_couldBeRelevant.get(itemIdJ, 1) #The only difference in application of penalty for prob vs. reduce relevance

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
        prob_couldBeRelevant:dict = self.__getPenaltiesOfItemIDs(userID, itemIDs, history)

        penalizedRatings:List[float] = []
        itemIdI:int
        ratingI:int
        for itemIdI, ratingI in methodsResultSrs.items():
            penalizedRatings.append(ratingI * prob_couldBeRelevant.get(itemIdI, 1))

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
        probabilities:dict = {}
        for indexJ, userIdJ, itemIdJ, positionJ, clickedJ, timestampJ in prevRecomendations:
            actProbability = probabilities.get(itemIdJ, 1) #init with one
            
            prob_implicit_rejection:float = self._penalPositionFnc(positionJ, *self._argumentsPositionList)
            prob_stable_pref:float = self._penalHistoryFnc(i, *self._argumentsHistoryList)
            prob_couldBeRelevant = (1-(prob_implicit_rejection * prob_stable_pref)  )

            probabilities[itemIdJ]  =  actProbability*prob_couldBeRelevant
            
            i = i-1
        print(probabilities)
        return probabilities
        
        
        """
        print(self._penalHistoryFnc)
        print(self._argumentsHistoryList)
        penalties:dict = {}
        for itemIdI in itemIDs:
            print(history.getPreviousRecomOfUserAndItem(userID, itemIdI, self._lengthOfHistory))
            prevRecomendations:List[tuple] = history.getPreviousRecomOfUserAndItem(userID, itemIdI, self._lengthOfHistory)
            #prevRecomendations.reverse()

            prob_couldBeRelevant: float = 1
            i:int = 0
            #print(prevRecomendations)
            for indexJ, userIdJ, itemIdJ, positionJ, clickedJ, timestampJ in prevRecomendations:
                #major differences in application of calculated penalties for prob vs. reduced relevance
                #prob_implicit_rejection: probability that by ignoring an item, user expressed an implicit rejection
                prob_implicit_rejection:float = self._penalPositionFnc(positionJ, *self._argumentsPositionList)
                #prob_changed_mind: probability that during the time between now and the implicit rejection event, user changed his/her mind
                prob_changed_mind:float = self._penalHistoryFnc(indexJ, *self._argumentsHistoryList)
                
                
                prob_couldBeRelevant *= (1-(prob_implicit_rejection * (1-prob_changed_mind))  )
                #print(itemIdI, positionJ, indexJ, prob_implicit_rejection, prob_changed_mind, (1-(prob_implicit_rejection * (1-prob_changed_mind))))
                
                i += 1
            print(itemIdI, prob_couldBeRelevant)
            penalties[itemIdI] = prob_couldBeRelevant
            
        return penalties
        """