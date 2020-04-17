#!/usr/bin/python3
import random

import numpy as np
import pandas as pd

from numpy.random import beta
from typing import List

from pandas.core.frame import DataFrame #class

from configuration.arguments import Arguments #class
from configuration.argument import Argument #class

from recommendation.resultOfRecommendation import ResultOfRecommendation #class
from recommendation.resultsOfRecommendations import ResultsOfRecommendations #class

from aggregation.aaggregation import AAgregation #class

from evaluationOfRecommender.evaluationOfRecommenders import EvaluationOfRecommenders #class


class AggrDHont(AAgregation):

    def __init__(self, arguments:Arguments):

       if type(arguments) is not Arguments:
          raise ValueError("Argument arguments is not type Arguments.")

       self._arguments = arguments;


    # userDef:DataFrame<(methodID, votes)>
    def run(self, resultsOfRecommendations:ResultsOfRecommendations, userDef:DataFrame, numberOfItems:int=20):

        if type(resultsOfRecommendations) is not ResultsOfRecommendations:
             raise ValueError("Argument resultsOfRecommendations is not type ResultsOfRecommendations.")

        if type(userDef) is not DataFrame:
             raise ValueError("Argument userDef isn't type DataFrame.")

        if type(numberOfItems) is not int:
             raise ValueError("Argument numberOfItems isn't type int.")

        methodsResultDict = resultsOfRecommendations.exportAsDictionaryOfSeries()
        #print(methodsResultDict)

        return self.aggrElectionsRun(methodsResultDict, userDef, topK=numberOfItems)


#    def __run(self, resultsOfRecommendations:ResultsOfRecommendations, evaluationOfRecommenders:EvaluationOfRecommenders, numberOfItems:int=20):
#
#        if type(resultsOfRecommendations) is not ResultsOfRecommendations:
#             raise ValueError("Argument resultsOfRecommendations is not type ResultsOfRecommendations.")
#
#        if type(numberOfItems) is not int:
#             raise ValueError("Argument numberOfItems is not type int.")
#
#        methodsResultDict = resultsOfRecommendations.exportAsDictionaryOfSeries()
#        #print(methodsResultDict)
#
#        methodsParamsDF = evaluationOfRecommenders.exportAsParamsDF()
#        #print(methodsParamsDF)
#
#        return self.aggrElectionsRun(methodsResultDict, methodsParamsDF, topK=numberOfItems)


    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # methodsParamsDF:pd.DataFrame[numberOfVotes:int], topK:int
    def aggrElectionsRun(self, methodsResultDict, methodsParamsDF:DataFrame, topK:int = 20):

      if sorted([mI for mI in methodsParamsDF.index]) != sorted([mI for mI in methodsResultDict.keys()]):
        raise ValueError("Arguments methodsResultDict and methodsParamsDF have to define the same methods.")

      if np.prod([len(methodsResultDict.get(mI)) for mI in methodsResultDict]) == 0:
        raise ValueError("Argument methodsParamsDF contains in ome method an empty list of items.")

      if topK < 0 :
        raise ValueError("Argument topK must be positive value.")

      candidatesOfMethods = np.asarray([cI.keys() for cI in methodsResultDict.values()])
      uniqueCandidatesI = list(set(np.concatenate(candidatesOfMethods)))
      #print("UniqueCandidatesI: ", uniqueCandidatesI)

      # numbers of elected candidates of parties
      electedOfPartyDictI = {mI:1 for mI in methodsParamsDF.index}
      #print("ElectedForPartyI: ", electedOfPartyDictI)

      # votes number of parties
      votesOfPartiesDictI = {mI:methodsParamsDF.votes.loc[mI] for mI in methodsParamsDF.index}
      #print("VotesOfPartiesDictI: ", votesOfPartiesDictI)

      recommendedItemIDs = []

      for iIndex in range(0, topK):
        #print("iIndex: ", iIndex)

        if len(uniqueCandidatesI) == 0:
            return recommendedItemIDs[:topK]

        # coumputing of votes of remaining candidates
        actVotesOfCandidatesDictI = {}
        for candidateIDJ in uniqueCandidatesI:
           votesOfCandidateJ = 0
           for parityIDK in methodsParamsDF.index:
              partyAffiliationOfCandidateKJ = methodsResultDict[parityIDK].get(candidateIDJ, 0)
              votesOfPartyK = votesOfPartiesDictI.get(parityIDK)
              votesOfCandidateJ += partyAffiliationOfCandidateKJ * votesOfPartyK
           actVotesOfCandidatesDictI[candidateIDJ] = votesOfCandidateJ
        #print(actVotesOfCandidatesDictI)

        # get the highest number of votes of remaining candidates
        maxVotes = max(actVotesOfCandidatesDictI.values())
        #print("MaxVotes: ", maxVotes)

        # select candidate with highest number of votes
        selectedCandidateI = [votOfCandI for votOfCandI in actVotesOfCandidatesDictI.keys() if actVotesOfCandidatesDictI[votOfCandI] == maxVotes][0]
        #print("SelectedCandidateI: ", selectedCandidateI)

        # add new selected candidate in results
        recommendedItemIDs.append(selectedCandidateI);

        # removing elected candidate from list of candidates
        uniqueCandidatesI.remove(selectedCandidateI)

        # updating number of elected candidates of parties
        electedOfPartyDictI = {partyIDI:electedOfPartyDictI[partyIDI] + methodsResultDict[partyIDI].get(selectedCandidateI, 0) for partyIDI in electedOfPartyDictI.keys()}
        #print("DevotionOfPartyDictI: ", devotionOfPartyDictI)

        # updating number of votes of parties
        votesOfPartiesDictI = {partyI:methodsParamsDF.votes.loc[partyI] / electedOfPartyDictI.get(partyI)  for partyI in methodsParamsDF.index}
        #print("VotesOfPartiesDictI: ", votesOfPartiesDictI)

      # list<int>
      return recommendedItemIDs[:topK]

    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # methodsParamsDF:pd.DataFrame[numberOfVotes:int], topK:int
    # differencAmplificatorExponent : int / float
    def aggrRandomizedElectionsRun(self, methodsResultDict, methodsParamsDF, differenceAmplificatorExponent, topK=20):

        if sorted([mI for mI in methodsParamsDF.index]) != sorted([mI for mI in methodsResultDict.keys()]):
            raise ValueError("Arguments methodsResultDict and methodsParamsDF have to define the same methods.")

        if np.prod([len(methodsResultDict.get(mI)) for mI in methodsResultDict]) == 0:
            raise ValueError("Argument methodsParamsDF contains in ome method an empty list of items.")

        if topK < 0:
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

        for iIndex in range(0, topK):
            # print("iIndex: ", iIndex)

            if len(uniqueCandidatesI) == 0:
                return recommendedItemIDs[:topK]

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

        return recommendedItemIDs[:topK]


    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # methodsParamsDF:DataFrame<(methodID:str, votes:int)>, topK:int
    def runWithResponsibility(self, methodsResultDict, methodsParamsDF, topK=20):
      
      # recommendedItemIDs:int[]
      recommendedItemIDs = self.aggrElectionsRun(methodsResultDict, methodsParamsDF, topK)
      votesOfPartiesDictI = {mI:methodsParamsDF.votes.loc[mI] for mI in methodsParamsDF.index}
      
      candidatesOfMethods = np.asarray([cI.keys() for cI in methodsResultDict.values()])
      uniqueCandidatesI = list(set(np.concatenate(candidatesOfMethods)))

      candidateOfdevotionOfPartiesDictDict = {}
      for candidateIDI in recommendedItemIDs:
      #for candidateIDI in uniqueCandidatesI:
         devotionOfParitiesDict = {}
         for parityIDJ in methodsParamsDF.index:
            devotionOfParitiesDict[parityIDJ] = methodsResultDict[parityIDJ].get(candidateIDI, 0)  * votesOfPartiesDictI[parityIDJ]
         candidateOfdevotionOfPartiesDictDict[candidateIDI] = devotionOfParitiesDict
      #print(candidateOfdevotionOfPartiesDictDict)

      # selectedCandidate:list<(itemID:int, Series<(rating:int, methodID:str)>)>
      selectedCandidate = [(candidateI, candidateOfdevotionOfPartiesDictDict[candidateI]) for candidateI in recommendedItemIDs]

      # list<(itemID:int, Series<(rating:int, methodID:str)>)>
      return selectedCandidate



if __name__== "__main__":
  print("Running Elections:")

  # number of recommended items
  N = 120

  # method results, items=[1,2,4,5,6,7,8,12,32,64,77]
  methodsResultDict = {
          "metoda1":pd.Series([0.2,0.1,0.3,0.3,0.1],[32,2,8,1,4],name="rating"),
          "metoda2":pd.Series([0.1,0.1,0.2,0.3,0.3],[1,5,32,6,7],name="rating"),
          "metoda3":pd.Series([0.3,0.1,0.2,0.3,0.1],[7,2,77,64,12],name="rating")
          }
  #print(methodsResultDict)


  # methods parametes
  methodsParamsData = [['metoda1',100], ['metoda2',80], ['metoda3',60]]
  methodsParamsDF = pd.DataFrame(methodsParamsData, columns=["methodID","votes"])
  methodsParamsDF.set_index("methodID", inplace=True)

  #print(methodsParamsDF)

  args = Arguments([Argument("arg1", 0)])
  aggregator = AggrDHont(args)

  print("aggrElectionsRun:")
  print(aggregator.aggrElectionsRun(methodsResultDict, methodsParamsDF, N))

  print("aggrRandomizedElectionsRun:")
  print(aggregator.aggrRandomizedElectionsRun(methodsResultDict, methodsParamsDF, 5, N))

  #print("aggrElectionsRunWithResponsibility:")
  #print(aggregator.aggrElectionsRunWithResponsibility(methodsResultDict, methodsParamsDF, N))

