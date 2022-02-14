#!/usr/bin/python3

import os

from portfolioModel.pModelBandit import PModelBandit #class

import pandas as pd
from pandas import DataFrame


def test01():
    print("Test 01")

    m = PModelBandit(["r1", "r2"])
    print(m.head(10))
    print("")


    df = DataFrame([[0,1,1,1,1]], columns=PModelBandit.getColumns())
    df.set_index(PModelBandit.COL_METHOD_ID, inplace=True)
    df.__class__ = PModelBandit


if __name__ == "__main__":
    os.chdir("..")

    test01()