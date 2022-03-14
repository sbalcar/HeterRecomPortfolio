#!/usr/bin/python3

import os
from typing import List

import pandas as pd
from pandas import DataFrame

from portfolioModel.pModelDHondt import PModelDHondt #class


def test01():
    print("Test 01")

    m = PModelDHondt(["r1", "r2"])
    print(m.head(10))


def test02():
    print("Test 02")

    # methods parametes
    portfolioModelData1:List[tuple] = [['metoda1',100], ['metoda2',80], ['metoda3',60]]
    portfolioModel1:DataFrame = pd.DataFrame(portfolioModelData1, columns=["methodID","votes"])
    portfolioModel1.set_index("methodID", inplace=True)
    portfolioModel1.__class__ = PModelDHondt

    #portfolioModel1.linearNormalizing()

    portfolioModelData2:List[tuple] = [['metoda1',0], ['metoda2',20], ['metoda3',40]]
    portfolioModel2:DataFrame = pd.DataFrame(portfolioModelData2, columns=["methodID","votes"])
    portfolioModel2.set_index("methodID", inplace=True)
    portfolioModel2.__class__ = PModelDHondt

    #print(portfolioModel1.head())
    #print(portfolioModel2.head())

    rModel = PModelDHondt.sumModels(portfolioModel1, portfolioModel2)
    print(rModel)
    print()

    PModelDHondt.linearNormalizingPortfolioModelDHondt(rModel)
    print(rModel)


if __name__ == "__main__":
    os.chdir("..")

    #test01()
    test02()