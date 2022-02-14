#!/usr/bin/python3

import os
from typing import List
from typing import Dict

from pandas.core.frame import DataFrame #class

import pandas as pd

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class

from batchDefinition.inputRecomSTDefinition import InputRecomSTDefinition #class

from portfolioModel.pModelDHondt import PModelDHondt #class


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
    portfolioModelDF:DataFrame = pd.DataFrame(portfolioModelData, columns=["methodID","votes"])
    portfolioModelDF.set_index("methodID", inplace=True)

    print("Definition:")
    print(portfolioModelDF)
    print()

    # linearly normalizing to unit sum of votes
    EvalToolDHondt.linearNormalizingPortfolioModelDHont(portfolioModelDF)

    print("Linearly normalizing:")
    print(portfolioModelDF)
    print()


    evaluationDict:dict = {}

    print("Clicked:")
    evalTool:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS:0.1, EvalToolDHondt.ARG_LEARNING_RATE_VIEWS:0.1,})
    evalTool.click(rItemIDsWithResponsibility, 7, portfolioModelDF, evaluationDict)
    evalTool.click(rItemIDsWithResponsibility, 1, portfolioModelDF, evaluationDict)
    evalTool.click(rItemIDsWithResponsibility, 7, portfolioModelDF, evaluationDict)
    print()

    print("Displayed - start:")
    for i in range(100):
        rItemIDsWithResponsibility1:List = [(7, {'metoda1': 0, 'metoda2': 24.0, 'metoda3': 18.0})]
        evalTool.displayed(rItemIDsWithResponsibility1, portfolioModelDF, evaluationDict)
    print(portfolioModelDF)
    print("Displayed - end:")
    print()

    print("Clicked:")
    evalTool.click(rItemIDsWithResponsibility, 4, portfolioModelDF, evaluationDict)
    print()


def test02():
    print("Test 02")

    rIDs, rDescr = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

    model:DataFrame = PModelDHondt(rIDs)
    print(model)

    userID:int = 1

    rItemIDsWithResponsibility:List[(int, Dict)] = [(1,
            {rIDs[0]:0.05, rIDs[1]:0.05, rIDs[2]:0.05, rIDs[3]:0.05,
             rIDs[4]:0.05, rIDs[5]:0.05, rIDs[6]:0.05, rIDs[7]:0.65})
            ]

    lrClick:float = 0.03
    lrView:float = lrClick / 500
    evalTool:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS:lrClick, EvalToolDHondt.ARG_LEARNING_RATE_VIEWS:lrView})
    evalTool.click(userID, rItemIDsWithResponsibility, 1, model, {})

    for i in range(555):
        evalTool.displayed(userID, rItemIDsWithResponsibility, model, {})

    print(model)


if __name__ == '__main__':
    os.chdir("..")

#    test01()
    test02()