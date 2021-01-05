#!/usr/bin/python3

from typing import List

import os

from pandas.core.frame import DataFrame #class

import pandas as pd

from evaluationTool.evalToolContext import EvalToolContext #class

from datasets.ml.users import Users #class
from datasets.ml.items import Items #class

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

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
    portfolioModelData = [['metoda1',100], ['metoda2',80], ['metoda3',60]]
    portfolioModelDF:DataFrame = pd.DataFrame(portfolioModelData, columns=["methodID","votes"])
    portfolioModelDF.set_index("methodID", inplace=True)


    itemsDF: DataFrame = Items.readFromFileMl1m()
    usersDF: DataFrame = Users.readFromFileMl1m()

    print("Definition:")
    print(portfolioModelDF)
    print(sumMethods(portfolioModelDF))
    print()

    userID = 1
    itemID = 2
    historyDF: AHistory = HistoryDF("test01")
    historyDF.insertRecommendation(userID, itemID, 1, True, 10)
    historyDF.insertRecommendation(userID, itemID, 1, True, 20)
    historyDF.insertRecommendation(userID, itemID, 1, True, 30)
    historyDF.insertRecommendation(userID, itemID, 1, False, 40)


    evaluationDict:dict = {EvalToolContext.ARG_USER_ID: userID,
                           EvalToolContext.ARG_RELEVANCE: methodsResultDict}
    evalToolDHondt = EvalToolContext(
        {EvalToolContext.ARG_USERS: usersDF,
         EvalToolContext.ARG_ITEMS: itemsDF,
         EvalToolContext.ARG_DATASET: "ml",
         EvalToolContext.ARG_HISTORY: historyDF}
    )

    print("Clicked:")
    evalToolDHondt.click(rItemIDsWithResponsibility, 7, portfolioModelDF, evaluationDict)
    evalToolDHondt.click(rItemIDsWithResponsibility, 1, portfolioModelDF, evaluationDict)
    evalToolDHondt.click(rItemIDsWithResponsibility, 7, portfolioModelDF, evaluationDict)
    print(portfolioModelDF)
    print(sumMethods(portfolioModelDF))
    print()

    print("Displayed - start:")
    for i in range(100):
        evalToolDHondt.displayed(rItemIDsWithResponsibility, portfolioModelDF, evaluationDict)
        print(portfolioModelDF)
        print(sumMethods(portfolioModelDF))
        print()
    print(portfolioModelDF)
    print(sumMethods(portfolioModelDF))
    print("Displayed - end:")
    print()

    print("Clicked:")
    evalToolDHondt.click(rItemIDsWithResponsibility, 4, portfolioModelDF, evaluationDict)
    print(portfolioModelDF)
    print(sumMethods(portfolioModelDF))
    print()

def sumMethods(methods):
    sum = 0
    for index, m in methods.iterrows():
        sum += m[0]
    return sum


if __name__ == "__main__":
    os.chdir("..")
    test01()