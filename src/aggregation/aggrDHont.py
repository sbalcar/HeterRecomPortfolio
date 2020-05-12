#!/usr/bin/python3
import random

import numpy as np
import pandas as pd

from numpy.random import beta
from typing import List
#from pandas.Series

from pandas.core.frame import DataFrame #class

from aggregation.tools.responsibilityDHont import countDHontResponsibility #function

from aggregation.aAggregation import AAgregation #class


class AggrDHont(AAgregation):

    def __init__(self, argumentsDict:dict):

       if type(argumentsDict) is not dict:
          raise ValueError("Argument argumentsDict is not type dict.")

       self._argumentsDict = argumentsDict


    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # modelDF:pd.DataFrame[numberOfVotes:int], numberOfItems:int
    def run(self, methodsResultDict:dict, modelDF:DataFrame, numberOfItems:int = 20):

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

      candidatesOfMethods:np.asarray[str] = np.asarray([cI.keys() for cI in methodsResultDict.values()])
      uniqueCandidatesI:List[int] = list(set(np.concatenate(candidatesOfMethods)))
      #print("UniqueCandidatesI: ", uniqueCandidatesI)

      # numbers of elected candidates of parties
      electedOfPartyDictI:dict[str,int] = {mI:1 for mI in modelDF.index}
      #print("ElectedForPartyI: ", electedOfPartyDictI)

      # votes number of parties
      votesOfPartiesDictI:dict[str,float] = {mI:modelDF.votes.loc[mI] for mI in modelDF.index}
      #print("VotesOfPartiesDictI: ", votesOfPartiesDictI)

      recommendedItemIDs:List[int] = []

      iIndex:int
      for iIndex in range(0, numberOfItems):
        #print("iIndex: ", iIndex)

        if len(uniqueCandidatesI) == 0:
            return recommendedItemIDs[:numberOfItems]

        # coumputing of votes of remaining candidates
        actVotesOfCandidatesDictI:dict[int,int] = {}
        candidateIDJ:int
        for candidateIDJ in uniqueCandidatesI:
           votesOfCandidateJ:int = 0
           parityIDK:str
           for parityIDK in modelDF.index:
              partyAffiliationOfCandidateKJ:float = methodsResultDict[parityIDK].get(candidateIDJ, 0)
              votesOfPartyK:int = votesOfPartiesDictI.get(parityIDK)
              votesOfCandidateJ += partyAffiliationOfCandidateKJ * votesOfPartyK
           actVotesOfCandidatesDictI[candidateIDJ] = votesOfCandidateJ
        #print(actVotesOfCandidatesDictI)

        # get the highest number of votes of remaining candidates
        maxVotes:float = max(actVotesOfCandidatesDictI.values())
        #print("MaxVotes: ", maxVotes)

        # select candidate with highest number of votes
        selectedCandidateI:int = [votOfCandI for votOfCandI in actVotesOfCandidatesDictI.keys() if actVotesOfCandidatesDictI[votOfCandI] == maxVotes][0]
        #print("SelectedCandidateI: ", selectedCandidateI)

        # add new selected candidate in results
        recommendedItemIDs.append(selectedCandidateI);

        # removing elected candidate from list of candidates
        uniqueCandidatesI.remove(selectedCandidateI)

        # updating number of elected candidates of parties
        electedOfPartyDictI = {partyIDI:electedOfPartyDictI[partyIDI] + methodsResultDict[partyIDI].get(selectedCandidateI, 0) for partyIDI in electedOfPartyDictI.keys()}
        #print("DevotionOfPartyDictI: ", devotionOfPartyDictI)

        # updating number of votes of parties
        votesOfPartiesDictI = {partyI: modelDF.votes.loc[partyI] / electedOfPartyDictI.get(partyI) for partyI in modelDF.index}
        #print("VotesOfPartiesDictI: ", votesOfPartiesDictI)

      # list<int>
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

        itemsWithResposibilityOfRecommenders:List[int,np.Series[int,str]] = countDHontResponsibility(
            aggregatedItemIDs, methodsResultDict, modelDF, numberOfItems)

        # list<(itemID:int, Series<(rating:int, methodID:str)>)>
        return itemsWithResposibilityOfRecommenders
