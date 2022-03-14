#!/usr/bin/python3

import os
from typing import List

import pandas as pd
from pandas import DataFrame

from evaluationTool.evalToolDHondt import EvalToolDHondt

from portfolioModel.pModelDHondt import PModelDHondt #class
from portfolioModel.pModelDHondtPersonalisedStat import PModelDHondtPersonalisedStat #class
from portfolioModel.pModelHybrid import PModelHybrid #class

from batchDefinition.inputRecomRRDefinition import InputRecomRRDefinition #class


def test01():
    print("Test 01")

    rIDs, rDescs = InputRecomRRDefinition.exportPairOfRecomIdsAndRecomDescrs()

    mGlobal:DataFrame = PModelDHondt(rIDs)
    mPerson:DataFrame = PModelDHondtPersonalisedStat(rIDs)
    mh:DataFrame = PModelHybrid(mGlobal, mPerson)
    mh.getModel(1)



if __name__ == '__main__':
    os.chdir("..")

    test01()
#    test02()