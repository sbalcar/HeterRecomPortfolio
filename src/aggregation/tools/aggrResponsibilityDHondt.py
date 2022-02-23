#!/usr/bin/python3
import random
import os

import numpy as np
import pandas as pd

from numpy.random import beta
from typing import List

from pandas.core.frame import DataFrame  # class

from aggregation.aAggregation import AAgregation  # class


# methodsResultDict:{String:Series(rating:float[], itemID:int[])},
def countAggrDHondtResponsibility(methodsResult:List[tuple], modelDF:DataFrame):
    if type(methodsResult) is not list:
        raise ValueError("Type of methodsResultDict isn't list.")
    if type(modelDF) is not DataFrame:
        raise ValueError("Type of methodsParamsDF isn't DataFrame.")
    if list(modelDF.columns) != ['votes']:
        print(modelDF)
        raise ValueError("Argument methodsParamsDF doen't contain rights columns.")

    numberOfVotes:int = modelDF['votes'].sum()
    #print(numberOfVotes)

    result:List[tuple] = []
    for itemIdI, responsDictI in methodsResult:
        weightedRelevances:List[float] = []
        #print(responsDictI)
        for methIdKeyJ in responsDictI:
            #print(methIdKeyJ)
            wIJ: float = modelDF.loc[methIdKeyJ, 'votes'] / numberOfVotes
            weightedRelevances.append(responsDictI[methIdKeyJ] * wIJ)

        weightedRelevanceJ:float = sum(weightedRelevances)
        result.append((itemIdI, weightedRelevanceJ))



    return result



def normalizationOfDHondtResponsibility(rItemIdsWitRespons:List[tuple]):
    result:List[tuple] = []

    for itemIDI, resposDistI in rItemIdsWitRespons:
        sumOfAll:float = sum(resposDistI.values())
        resposDistJ = {}
        for methodI, resposibilityI in resposDistI.items():
            resposDistJ[methodI] = resposibilityI / sumOfAll
        result.append((itemIDI, resposDistJ))

    return result