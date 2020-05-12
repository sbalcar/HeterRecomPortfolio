#!/usr/bin/python3
import random

import numpy as np
import pandas as pd

from numpy.random import beta
from typing import List

from pandas.core.frame import DataFrame  # class

from aggregation.aAggregation import AAgregation  # class

from aggregation.tools.responsibilityDHont import countDHontResponsibility #function

class AggrDHontRandomized(AAgregation):

    def __init__(self, argumentsDict:dict):

       if type(argumentsDict) is not dict:
          raise ValueError("Argument argumentsDict is not type dict.")

       self._argumentsDict = argumentsDict


    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # modelDF:pd.DataFrame[numberOfVotes:int], numberOfItems:int
    # differencAmplificatorExponent : int / float
    def run(self, methodsResultDict:dict, modelDF:DataFrame, differenceAmplificatorExponent:float, numberOfItems:int=20):

        # testing types of parameters
        if type(methodsResultDict) is not dict:
            raise ValueError("Type of methodsResultDict is not dict.")

        if type(modelDF) is not DataFrame:
            raise ValueError("Type of methodsParamsDF is not DataFrame.")
        if list(modelDF.columns) != ['votes']:
            raise ValueError("Argument methodsParamsDF doen't contain rights columns.")

        if type(numberOfItems) is not int:
            raise ValueError("Type of numberOfItems is not int.")

        if sorted([mI for mI in modelDF.index]) != sorted([mI for mI in methodsResultDict.keys()]):
            raise ValueError("Arguments methodsResultDict and methodsParamsDF have to define the same methods.")
        for mI in methodsResultDict.keys():
            if modelDF.loc[mI] is None:
                raise ValueError("Argument modelDF contains in ome method an empty list of items.")
        if numberOfItems < 0:
            raise ValueError("Argument numberOfItems must be positive value.")


        candidatesOfMethods = np.asarray([cI.keys() for cI in methodsResultDict.values()])
        uniqueCandidatesI = list(set(np.concatenate(candidatesOfMethods)))
        # print("UniqueCandidatesI: ", uniqueCandidatesI)

        # numbers of elected candidates of parties
        electedOfPartyDictI = {mI: 1 for mI in modelDF.index}
        # print("ElectedForPartyI: ", electedOfPartyDictI)

        # votes number of parties
        votesOfPartiesDictI = {mI: modelDF.votes.loc[mI] for mI in modelDF.index}
        # print("VotesOfPartiesDictI: ", votesOfPartiesDictI)

        recommendedItemIDs = []

        for iIndex in range(0, numberOfItems):
            # print("iIndex: ", iIndex)

            if len(uniqueCandidatesI) == 0:
                return recommendedItemIDs[:numberOfItems]

            # coumputing of votes of remaining candidates
            actVotesOfCandidatesDictI = {}
            for candidateIDJ in uniqueCandidatesI:
                votesOfCandidateJ = 0
                for parityIDK in modelDF.index:
                    partyAffiliationOfCandidateKJ = methodsResultDict[parityIDK].get(candidateIDJ, 0)
                    votesOfPartyK = votesOfPartiesDictI.get(parityIDK)
                    votesOfCandidateJ += partyAffiliationOfCandidateKJ * votesOfPartyK
                actVotesOfCandidatesDictI[candidateIDJ] = votesOfCandidateJ ** differenceAmplificatorExponent
            # print(actVotesOfCandidatesDictI)

            # ROULETTE SELECTION (selecting random candidate according vote distribution)
            # compute sum of all votes
            voteSum = sum(actVotesOfCandidatesDictI.values())

            # random number in range <0, voteSum)
            rnd = random.randrange(0, (int)(voteSum), 1)

            # find random candidate
            for candidate in actVotesOfCandidatesDictI.keys():
                if (rnd < actVotesOfCandidatesDictI[candidate]):
                    selectedCandidateI = candidate
                    break
                rnd -= actVotesOfCandidatesDictI[candidate]

            # add new selected candidate in results
            recommendedItemIDs.append(selectedCandidateI);

            # removing elected candidate from list of candidates
            uniqueCandidatesI.remove(selectedCandidateI)

            # updating number of elected candidates of parties
            electedOfPartyDictI = {
                partyIDI: electedOfPartyDictI[partyIDI] + methodsResultDict[partyIDI].get(selectedCandidateI, 0) for
                partyIDI in electedOfPartyDictI.keys()}
            # print("DevotionOfPartyDictI: ", devotionOfPartyDictI)

            # updating number of votes of parties
            votesOfPartiesDictI = {partyI: modelDF.votes.loc[partyI] / electedOfPartyDictI.get(partyI) for
                                   partyI in modelDF.index}
            # print("VotesOfPartiesDictI: ", votesOfPartiesDictI)

        return recommendedItemIDs[:numberOfItems]


    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # modelDF:DataFrame<(methodID:str, votes:int)>, numberOfItems:int
    def runWithResponsibility(self, methodsResultDict:dict, modelDF:DataFrame, numberOfItems:int=20):

        # testing types of parameters
        if type(methodsResultDict) is not dict:
            raise ValueError("Type of methodsResultDict is not dict.")
        for methI in methodsResultDict.values():
            if type(methI) is not pd.Series:
                raise ValueError("Type of methodsParamsDF doen't contain Series.")
        if type(modelDF) is not DataFrame:
            raise ValueError("Type of methodsParamsDF is not DataFrame.")
        if list(modelDF.columns) != ['votes']:
            raise ValueError("Argument methodsParamsDF doen't contain rights columns.")
        if type(numberOfItems) is not int:
            raise ValueError("Type of numberOfItems is not int.")

        if sorted([mI for mI in modelDF.index]) != sorted([mI for mI in methodsResultDict.keys()]):
            raise ValueError("Arguments methodsResultDict and methodsParamsDF have to define the same methods.")
        for mI in methodsResultDict.keys():
            if modelDF.loc[mI] is None:
                raise ValueError("Argument modelDF contains in ome method an empty list of items.")
        if numberOfItems < 0:
            raise ValueError("Argument numberOfItems must be positive value.")


        aggregatedItemIDs:List[int] = self.run(methodsResultDict, modelDF, numberOfItems)

        itemsWithResposibilityOfRecommenders:List = countDHontResponsibility(
            aggregatedItemIDs, methodsResultDict, modelDF, numberOfItems)

        # list<(itemID:int, Series<(rating:int, methodID:str)>)>
        return itemsWithResposibilityOfRecommenders


