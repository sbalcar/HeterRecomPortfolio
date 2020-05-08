#!/usr/bin/python3
import random

import numpy as np
import pandas as pd

from numpy.random import beta
from typing import List

from pandas.core.frame import DataFrame  # class

from configuration.arguments import Arguments  # class
from configuration.argument import Argument  # class

from recommendation.resultOfRecommendation import ResultOfRecommendation  # class
from recommendation.resultsOfRecommendations import ResultsOfRecommendations  # class

from aggregation.aAggregation import AAgregation  # class

from evaluationOfRecommender.evaluationOfRecommenders import EvaluationOfRecommenders  # class

from aggregation.tools.responsibilityDHont import countDHontResponsibility #function

class AggrDHontRandomized(AAgregation):

    def __init__(self, arguments: Arguments):

        def __init__(self, argumentsDict: dict):
            if type(argumentsDict) is not dict:
                raise ValueError("Argument argumentsDict is not type dict.")

            self._argumentsDict = argumentsDict

    # userDef:DataFrame<(methodID, votes)>
    def run(self, resultsOfRecommendations: ResultsOfRecommendations, userDef: DataFrame, numberOfItems: int = 20):

        if type(resultsOfRecommendations) is not ResultsOfRecommendations:
            raise ValueError("Argument resultsOfRecommendations is not type ResultsOfRecommendations.")

        if type(userDef) is not DataFrame:
            raise ValueError("Argument userDef isn't type DataFrame.")

        if type(numberOfItems) is not int:
            raise ValueError("Argument numberOfItems isn't type int.")

        methodsResultDict = resultsOfRecommendations.exportAsDictionaryOfSeries()
        # print(methodsResultDict)

        return self.aggrRandomizedElectionsRun(methodsResultDict, userDef, topK=numberOfItems)


    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # methodsParamsDF:pd.DataFrame[numberOfVotes:int], numberOfItems:int
    # differencAmplificatorExponent : int / float
    def aggrRandomizedElectionsRun(self, methodsResultDict:dict, methodsParamsDF:DataFrame, differenceAmplificatorExponent:float, numberOfItems:int=20):

        if sorted([mI for mI in methodsParamsDF.index]) != sorted([mI for mI in methodsResultDict.keys()]):
            raise ValueError("Arguments methodsResultDict and methodsParamsDF have to define the same methods.")

        if np.prod([len(methodsResultDict.get(mI)) for mI in methodsResultDict]) == 0:
            raise ValueError("Argument methodsParamsDF contains in ome method an empty list of items.")

        if numberOfItems < 0:
            raise ValueError("Argument topK must be positive value.")

        candidatesOfMethods = np.asarray([cI.keys() for cI in methodsResultDict.values()])
        uniqueCandidatesI = list(set(np.concatenate(candidatesOfMethods)))
        # print("UniqueCandidatesI: ", uniqueCandidatesI)

        # numbers of elected candidates of parties
        electedOfPartyDictI = {mI: 1 for mI in methodsParamsDF.index}
        # print("ElectedForPartyI: ", electedOfPartyDictI)

        # votes number of parties
        votesOfPartiesDictI = {mI: methodsParamsDF.votes.loc[mI] for mI in methodsParamsDF.index}
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
                for parityIDK in methodsParamsDF.index:
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
            votesOfPartiesDictI = {partyI: methodsParamsDF.votes.loc[partyI] / electedOfPartyDictI.get(partyI) for
                                   partyI in methodsParamsDF.index}
            # print("VotesOfPartiesDictI: ", votesOfPartiesDictI)

        return recommendedItemIDs[:numberOfItems]


    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # methodsParamsDF:DataFrame<(methodID:str, votes:int)>, topK:int
    def runWithResponsibility(self, methodsResultDict:dict, methodsParamsDF:DataFrame, numberOfItems:int=20):

        aggregatedItemIDs:List[int] = self.aggrElectionsRun(methodsResultDict, methodsParamsDF, numberOfItems)

        itemsWithResposibilityOfRecommenders:List = countDHontResponsibility(
            aggregatedItemIDs, methodsResultDict, methodsParamsDF, numberOfItems)

        # list<(itemID:int, Series<(rating:int, methodID:str)>)>
        return itemsWithResposibilityOfRecommenders


