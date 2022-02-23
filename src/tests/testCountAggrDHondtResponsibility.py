#!/usr/bin/python3
import random
import os

import numpy as np
import pandas as pd

from numpy.random import beta
from typing import List

from pandas.core.frame import DataFrame  # class

from aggregation.aAggregation import AAgregation  # class
from aggregation.tools.aggrResponsibilityDHondt import countAggrDHondtResponsibility #function
from aggregation.tools.aggrResponsibilityDHondt import normalizationOfDHondtResponsibility #function


def test01():
    print("Test 01")

    rItemIdsWitRespons = [(11, {'metoda1': 0.7, 'metoda2': 0.5, 'metoda3': 0.1}),
                        (21, {'metoda1': 0.7, 'metoda2': 0.1, 'metoda3': 0.5})]
    print("rItemIdsWitRespons:")
    print(rItemIdsWitRespons)

    rItemIdsWitResponsNorm = normalizationOfDHondtResponsibility(rItemIdsWitRespons)
    print("rItemIdsWitResponsNorm:")
    print(rItemIdsWitResponsNorm)

    # methods parametes
    portfolioModelData:List[tuple] = [['metoda1',100], ['metoda2',80], ['metoda3',60]]
    modelDF:DataFrame = pd.DataFrame(portfolioModelData, columns=["methodID","votes"])
    modelDF.set_index("methodID", inplace=True)

    print("Model:")
    print(modelDF)
    print("")

    result:List[tuple] = countAggrDHondtResponsibility(rItemIdsWitResponsNorm, modelDF)
    print("Result:")
    print(result)



if __name__ == "__main__":
    os.chdir("..")

    test01()