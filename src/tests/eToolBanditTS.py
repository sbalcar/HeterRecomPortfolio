#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class

import pandas as pd

from evaluationTool.evalToolDHondt import EvalToolDHondt #class
from evaluationTool.evalToolBanditTS import EvalToolBanditTS #class


def test01():
    print("Test 01")

    print("Running EvalToolBanditTS:")


    rItemIDsWithResponsibility:List = [(1, 'metoda1'), (32, 'metoda1'), (2, 'metoda1'), (8, 'metoda1'), (6, 'metoda2'), (4, 'metoda1'), (7, 'metoda3'), (5, 'metoda2'), (64, 'metoda3'), (77, 'metoda3'), (12, 'metoda3')]

    # methods parametes
    portfolioModelData:List[tuple] = [['metoda1', 5, 10, 1, 1], ['metoda2', 5, 12, 1, 1], ['metoda3', 6, 13, 1, 1]]
    portfolioModelDF:DataFrame = pd.DataFrame(portfolioModelData, columns=["methodID", "r", "n", "alpha0", "beta0"])
    portfolioModelDF.set_index("methodID", inplace=True)


    print("Definition:")
    print(portfolioModelDF)
    print()


    evaluationDict:dict = {}

    print("Clicked:")
    EvalToolBanditTS.click(rItemIDsWithResponsibility, 1, portfolioModelDF, evaluationDict)
    print()

    print("Clicked:")
    EvalToolBanditTS.click(rItemIDsWithResponsibility, 32, portfolioModelDF, evaluationDict)
    print()

    print("Clicked:")
    EvalToolBanditTS.click(rItemIDsWithResponsibility, 6, portfolioModelDF, evaluationDict)
    print()

    print("Displayed:")
    rItemIDsWithResponsibility1:List[tuple] = [(1, 'metoda1'), (32, 'metoda1'), (2, 'metoda1'), (8, 'metoda1')]
    for i in [0,1,2]:
        EvalToolBanditTS.displayed(rItemIDsWithResponsibility1, portfolioModelDF, evaluationDict)
        print(portfolioModelDF)


test01()