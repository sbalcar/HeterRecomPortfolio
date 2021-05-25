#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame  # class
from pandas.core.series import Series  # class

from aggregation.aggrFAI import AggrFAI  # class

import pandas as pd
from history.historyDF import HistoryDF  # class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription  # class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc  # function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc  # function

from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector  # class


def test01():
    print("Test 01")

    print("Running AggrRandomRecsSwitching:")

    # number of recommended items
    N = 7

    # method results, items=[1,2,4,5,6,7,8,12,32,64,77]
    methodsResultDict: dict = {
        "metoda1": pd.Series([0.2, 0.5, 0.3, 0.3, 0.1], [32, 2, 8, 1, 4], name="rating"),
        "metoda2": pd.Series([0.5, 0.1, 0.2, 0.3, 0.3], [2, 5, 32, 6, 7], name="rating"),
        "metoda3": pd.Series([0.3, 0.5, 0.2, 0.3, 0.1], [7, 2, 77, 64, 12], name="rating"),
        "metoda4": pd.Series([], [], name="rating")
    }
    # print(methodsResultDict)

    aggr: AggrFAI = AggrFAI(HistoryDF(""), {})

    userID: int = 101
    itemIDs: List[tuple] = aggr.runWithResponsibility(methodsResultDict, {}, userID, N)
    print(itemIDs)

    itemIDs: List[tuple] = aggr.runWithResponsibility(methodsResultDict, {}, userID, N)
    print(itemIDs)

    userID: int = 102
    itemIDs: List[tuple] = aggr.runWithResponsibility(methodsResultDict, {}, userID, N)
    print(itemIDs)


if __name__ == "__main__":
    print("AggrFAI algorithm")

    test01()
