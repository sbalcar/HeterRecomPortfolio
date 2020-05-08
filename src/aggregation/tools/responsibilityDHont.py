#!/usr/bin/python3
import random

import numpy as np
import pandas as pd

from numpy.random import beta
from typing import List

from pandas.core.frame import DataFrame  # class

from recommendation.resultOfRecommendation import ResultOfRecommendation  # class
from recommendation.resultsOfRecommendations import ResultsOfRecommendations  # class

from aggregation.aAggregation import AAgregation  # class

from evaluationOfRecommender.evaluationOfRecommenders import EvaluationOfRecommenders  # class


# methodsResultDict:{String:Series(rating:float[], itemID:int[])},
# methodsParamsDF:DataFrame<(methodID:str, votes:int)>, topK:int
def countDHontResponsibility(aggregatedItemIDs:List[int], methodsResultDict:dict, methodsParamsDF:DataFrame, numberOfItems:int=20):

    #sumOfAllVotes:int = sum(methodsParamsDF["votes"].values)

    votesOfPartiesDictI:dict[str,int] = {mI:methodsParamsDF.votes.loc[mI] for mI in methodsParamsDF.index}

    candidatesOfMethods:np.asarray[str] = np.asarray([cI.keys() for cI in methodsResultDict.values()])
    uniqueCandidatesI:List[str] = list(set(np.concatenate(candidatesOfMethods)))

    candidateOfdevotionOfPartiesDictDict:dict = {}

    candidateIDI:int
    for candidateIDI in aggregatedItemIDs:
        # for candidateIDI in uniqueCandidatesI:
        devotionOfParitiesDict:dict = {}

        parityIDJ:str
        for parityIDJ in methodsParamsDF.index:
            devotionOfParitiesDict[parityIDJ] = methodsResultDict[parityIDJ].get(candidateIDI, 0)  * votesOfPartiesDictI[parityIDJ]
        candidateOfdevotionOfPartiesDictDict[candidateIDI] = devotionOfParitiesDict
    # print(candidateOfdevotionOfPartiesDictDict)

    # selectedCandidate:list<(itemID:int, Series<(rating:int, methodID:str)>)>
    selectedCandidate:List[int, pd.Series[str,int]] = [(candidateI, candidateOfdevotionOfPartiesDictDict[candidateI]) for candidateI in aggregatedItemIDs]

    # list<(itemID:int, Series<(rating:int, methodID:str)>)>
    return selectedCandidate[:numberOfItems]