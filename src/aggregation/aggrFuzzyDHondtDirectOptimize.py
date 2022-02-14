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
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector  # class

from history.aHistory import AHistory  # class
from abc import ABC, abstractmethod

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription  # class


class AggrFuzzyDHondtDirectOptimize(AAgregation):

    ARG_SELECTOR: str = "selector"

    def __init__(self, history: AHistory, argumentsDict: Dict[str, object]):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        self._history = history
        self._selector = argumentsDict[self.ARG_SELECTOR]

    def update(self, ratingsUpdateDF: DataFrame, argumentsDict: Dict[str, object]):
        pass

    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # modelDF:pd.DataFrame[numberOfVotes:int], numberOfItems:int
    def run(self, methodsResultDict: Dict, modelDF: DataFrame, userID: int, numberOfItems: int,
            argumentsDict: Dict[str, object] = {}):

        # testing types of parameters
        if type(methodsResultDict) is not dict:
            raise ValueError("Type of methodsResultDict isn't dict.")
        if type(modelDF) is not DataFrame:
            raise ValueError("Type of methodsParamsDF isn't DataFrame.")
        if list(modelDF.columns) != ['votes']:
            raise ValueError("Argument methodsParamsDF doen't contain rights columns.")
        if type(numberOfItems) is not int:
            raise ValueError("Type of numberOfItems isn't int.")

        if sorted([mI for mI in modelDF.index]) != sorted([mI for mI in methodsResultDict.keys()]):
            raise ValueError("Arguments methodsResultDict and methodsParamsDF have to define the same methods.")
        for mI in methodsResultDict.keys():
            if modelDF.loc[mI] is None:
                raise ValueError("Argument modelDF contains in ome method an empty list of items.")
        if numberOfItems < 0:
            raise ValueError("Argument numberOfItems must be positive value.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")
        # print(methodsResultDict)
        candidatesOfMethods = [np.array(list(cI.keys())) for cI in methodsResultDict.values()]
        # print(candidatesOfMethods)

        uniqueCandidatesI: List[int] = list(set(np.concatenate(candidatesOfMethods)))
        uniqueCandidatesI: list(map(int, uniqueCandidatesI))  # failsafe - in some cases it returns float

        # print("UniqueCandidatesI: ", uniqueCandidatesI)

        # sum of preference the elected candidates have for each party
        electedOfPartyDictI: dict[str, float] = {mI: 0.0 for mI in modelDF.index}
        # print("ElectedForPartyI: ", electedOfPartyDictI)

        # votes number of parties
        votesOfPartiesDictI: dict[str, float] = {mI: modelDF.votes.loc[mI] for mI in modelDF.index}
        totalVotes = modelDF["votes"].sum()
        # print("VotesOfPartiesDictI: ", votesOfPartiesDictI, totalVotes)

        recommendedItemIDs: List[int] = []

        totalSelectedCandidatesVotes: float = 0.0

        iIndex: int
        for iIndex in range(0, numberOfItems):
            # print("iIndex: ", iIndex)

            if len(uniqueCandidatesI) == 0:
                return recommendedItemIDs[:numberOfItems]

            # coumputing of votes of remaining candidates
            actVotesOfCandidatesDictI: dict[
                int, int] = {}  # calculates the proportonal improvement of the output if this candidate is included
            candidateIDJ: int
            for candidateIDJ in uniqueCandidatesI:
                votesOfCandidateJ: float = 0.0

                candidateVotesPerParty: dict[str, float] = {mI: methodsResultDict[mI].get(candidateIDJ, 0) for mI in
                                                            modelDF.index}
                candidateTotalVotes: float = np.sum(list(candidateVotesPerParty.values()))
                totalVotesPlusProspected: float = totalSelectedCandidatesVotes + candidateTotalVotes

                for parityIDK in modelDF.index:
                    # get the fraction of under-representation for the party
                    # check how much proportional representation the candidate adds
                    # sum over all parties & select the highest sum
                    votes_fraction_per_party = votesOfPartiesDictI[parityIDK] / totalVotes
                    notRepresentedVotesPerParty = max(0, (votes_fraction_per_party * totalVotesPlusProspected) -
                                                      electedOfPartyDictI[parityIDK])  # max(w_i*(A+C) - a_i,0)
                    # print("Unrepresented,",parityIDK,candidateIDJ,notRepresentedVotesPerParty)
                    votesOfCandidateJ += min(notRepresentedVotesPerParty, candidateVotesPerParty[
                        parityIDK])  # only account the amount of votes that does not exceed proportional representation

                actVotesOfCandidatesDictI[candidateIDJ] = votesOfCandidateJ
            # print(actVotesOfCandidatesDictI)

            # select candidate with highest number of votes
            # selectedCandidateI:int = AggrDHont.selectorOfTheMostVotedItem(actVotesOfCandidatesDictI)
            selectedCandidateI: int = self._selector.select(actVotesOfCandidatesDictI)

            # add new selected candidate in results
            recommendedItemIDs.append(selectedCandidateI);

            # removing elected candidate from list of candidates
            try:
                uniqueCandidatesI.remove(selectedCandidateI)
            except:
                print("Cannot remove" + str(selectedCandidateI) + " from " + str(uniqueCandidatesI))
                print("candidate votes")
                print(actVotesOfCandidatesDictI)
                # exit(1)

            # updating number of elected candidates of parties
            electedOfPartyDictI: dict = {
                partyIDI: electedOfPartyDictI[partyIDI] + methodsResultDict[partyIDI].get(selectedCandidateI, 0) for
                partyIDI in electedOfPartyDictI.keys()}
            totalSelectedCandidatesVotes = np.sum(list(electedOfPartyDictI.values()))
            # print("electedOfPartyDictI: ", electedOfPartyDictI, totalSelectedCandidatesVotes)

        # list<int>
        return recommendedItemIDs[:numberOfItems]

    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # modelDF:DataFrame<(methodID:str, votes:int)>, numberOfItems:int
    def runWithResponsibility(self, methodsResultDict: Dict, modelDF: DataFrame, userID: int, numberOfItems: int,
                              argumentsDict: Dict[str, object] = {}):
        # print(argumentsDict["pageType"])
        # testing types of parameters
        if type(methodsResultDict) is not dict:
            raise ValueError("Type of methodsResultDict isn't dict.")
        for methI in methodsResultDict.values():
            if type(methI) is not pd.Series:
                raise ValueError("Type of methodsParamsDF doesn't contain Series.")
        if type(modelDF) is not DataFrame:
            raise ValueError("Type of methodsParamsDF isn't DataFrame.")
        if list(modelDF.columns) != ['votes']:
            raise ValueError("Argument methodsParamsDF doesn't contain rights columns.")
        if type(numberOfItems) is not int:
            raise ValueError("Type of numberOfItems isn't int.")

        if sorted([mI for mI in modelDF.index]) != sorted([mI for mI in methodsResultDict.keys()]):
            raise ValueError("Arguments methodsResultDict and methodsParamsDF have to define the same methods.")
        for mI in methodsResultDict.keys():
            if modelDF.loc[mI] is None:
                raise ValueError("Argument modelDF contains in ome method an empty list of items.")
        if numberOfItems < 0:
            raise ValueError("Argument numberOfItems must be positive value.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        aggregatedItemIDs: List[int] = self.run(methodsResultDict, modelDF, userID, numberOfItems, argumentsDict)

        itemsWithResposibilityOfRecommenders: List[int, np.Series[int, str]] = countDHontResponsibility(
            aggregatedItemIDs, methodsResultDict, modelDF, numberOfItems)

        # list<(itemID:int, Series<(rating:int, methodID:str)>)>
        return itemsWithResposibilityOfRecommenders