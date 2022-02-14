#!/usr/bin/python3

import os
from typing import List
from typing import Dict

from pandas.core.frame import DataFrame #class

import pandas as pd

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class
from evaluationTool.evalToolDHondtPersonal import EToolDHondtPersonal #class

from batchDefinition.inputRecomSTDefinition import InputRecomSTDefinition #class

from portfolioModel.pModelDHondt import PModelDHondt #class
from portfolioModel.pModelDHondtPersonalised import PModelDHondtPersonalised #class


def test01():
    print("Test 01")

    #print("Running Two paralel History Databases:")

    # method results, items=[1,2,4,5,6,7,8,12,32,64,77]
    methodsResultDict:dict = {
        "metoda1": pd.Series([0.2, 0.1, 0.3, 0.3, 0.1], [32, 2, 8, 1, 4], name="rating"),
        "metoda2": pd.Series([0.1, 0.1, 0.2, 0.3, 0.3], [1, 5, 32, 6, 7], name="rating"),
        "metoda3": pd.Series([0.3, 0.1, 0.2, 0.3, 0.1], [7, 2, 77, 64, 12], name="rating")
    }

    rItemIDsWithResponsibility:List = [(7, {'metoda1': 0, 'metoda2': 24.0, 'metoda3': 18.0}), (1, {'metoda1': 30.0, 'metoda2': 8.0, 'metoda3': 0}), (32, {'metoda1': 20.0, 'metoda2': 16.0, 'metoda3': 0}), (8, {'metoda1': 30.0, 'metoda2': 0, 'metoda3': 0}), (6, {'metoda1': 0, 'metoda2': 24.0, 'metoda3': 0}), (64, {'metoda1': 0, 'metoda2': 0, 'metoda3': 18.0}), (2, {'metoda1': 10.0, 'metoda2': 0, 'metoda3': 6.0}), (77, {'metoda1': 0, 'metoda2': 0, 'metoda3': 12.0}), (4, {'metoda1': 10.0, 'metoda2': 0, 'metoda3': 0}), (5, {'metoda1': 0, 'metoda2': 8.0, 'metoda3': 0}), (12, {'metoda1': 0, 'metoda2': 0, 'metoda3': 6.0})]

    # methods parametes
    portfolioModelData:List[tuple] = [['metoda1',100], ['metoda2',80], ['metoda3',60]]
    portfolioModelDF:DataFrame = pd.DataFrame(portfolioModelData, columns=[PModelDHondt.COL_METHOD_ID, PModelDHondt.COL_VOTES])
    portfolioModelDF.set_index(PModelDHondt.COL_METHOD_ID, inplace=True)
    portfolioModelDF.__class__ = PModelDHondtPersonalised

    userID = 1

    tool = EToolDHondtPersonal({})
    tool.click(userID, rItemIDsWithResponsibility, [101], portfolioModelDF, {})


if __name__ == '__main__':
    os.chdir("..")

    test01()
#    test02()