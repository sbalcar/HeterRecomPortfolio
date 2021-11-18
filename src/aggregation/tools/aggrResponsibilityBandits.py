#!/usr/bin/python3
import random
import os

import numpy as np
import pandas as pd

from numpy.random import beta
from typing import List

from pandas.core.frame import DataFrame  # class
from pandas.core.frame import Series # class

from aggregation.aAggregation import AAgregation  # class

from batchDefinition.modelDefinition import ModelDefinition

from sklearn.preprocessing import normalize


# methodsResultDict:{String:Series(rating:float[], itemID:int[])},
def countAggrBanditsResponsibility(methodsResult:List[tuple], modelDF:DataFrame):

    #print(methodsResult)

    result:List[tuple] = []
    for itemIdI, methodIdI in methodsResult:
        wIJ:float = modelDF.loc[methodIdI, 'r'] / modelDF.loc[methodIdI, 'n']
        result.append((itemIdI,wIJ))

    itemsIDs:List[int] = [x[0] for x in result]
    scores:List[float] = [x[1] for x in result]
    resultSer:Series = Series(scores, index=itemsIDs)

    finalScores = normalize(np.expand_dims(resultSer.values, axis=0))[0, :]
    resultNorm:List[tuple] = zip(resultSer.index, finalScores.tolist())

    return resultNorm



if __name__ == "__main__":
    os.chdir("..")

    rItemIdsWitRespons:List[tuple] = [(11, 'metoda1'), (21, 'metoda2')]


    modelData:List[tuple] = [['metoda1', 5, 10, 1, 1], ['metoda2', 1, 10, 1, 1], ['metoda3', 2, 20, 1, 1]]
    modelDF:DataFrame = pd.DataFrame(modelData, columns=["methodID", "r", "n", "alpha0", "beta0"])
    modelDF.set_index("methodID", inplace=True)

    modelDF:DataFrame = ModelDefinition.createBanditModel(['metoda1', 'metoda2', 'metoda3'])

    print("Model:")
    print(modelDF)
    print("")

    result:List[tuple] = countAggrBanditsResponsibility(rItemIdsWitRespons, modelDF)
    print("Result:")
    print(result)