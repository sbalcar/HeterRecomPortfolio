#!/usr/bin/python3

import os

from portfolioModel.pModelDHondt import PModelDHondt #class

import pandas as pd
from pandas import DataFrame


def test01():
    print("Test 01")

    m = PModelDHondt(["r1", "r2"])
    print(m.head(10))


if __name__ == "__main__":
    os.chdir("..")

    test01()