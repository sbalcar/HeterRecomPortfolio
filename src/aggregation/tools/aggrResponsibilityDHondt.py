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
        print(responsDictI)
        for methIdKeyJ in responsDictI:
            print(methIdKeyJ)
            wIJ: float = modelDF.loc[methIdKeyJ, 'votes'] / numberOfVotes
            weightedRelevances.append(responsDictI[methIdKeyJ] * wIJ)

        weightedRelevanceJ:float = sum(weightedRelevances)
        result.append((itemIdI, weightedRelevanceJ))



    return result



if __name__ == "__main__":
    os.chdir("..")

    rItemIdsWitResponsDict = [(11, {'metoda1': 0.7, 'metoda2': 0.5, 'metoda3': 0.1}),
                        (21, {'metoda1': 0.7, 'metoda2': 0.1, 'metoda3': 0.5})]
    print(rItemIdsWitResponsDict)

    # methods parametes
    portfolioModelData:List[tuple] = [['metoda1',100], ['metoda2',80], ['metoda3',60]]
    modelDF:DataFrame = pd.DataFrame(portfolioModelData, columns=["methodID","votes"])
    modelDF.set_index("methodID", inplace=True)

    print("Model:")
    print(modelDF)
    print("")

    result:List[tuple] = countAggrDHondtResponsibility(rItemIdsWitResponsDict, modelDF)
    print("Result:")
    print(result)