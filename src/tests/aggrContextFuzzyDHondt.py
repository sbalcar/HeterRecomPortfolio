#!/usr/bin/python3

import os

from typing import List
from pandas.core.frame import DataFrame #class

from aggregation.aggrContextFuzzyDHondt import AggrContextFuzzyDHondt #class

import pandas as pd
from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

from datasets.ml.users import Users #class
from datasets.ml.items import Items #class


from aggregation.negImplFeedback.penalUsingReduceRelevance import penaltyLinear #function

from aggregation.negImplFeedback.penalUsingReduceRelevance import PenalUsingReduceRelevance #class
from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

def test01():
    print("Test 01")

    # number of recommended items
    N = 120

    methodsResultDict:dict[str,pd.Series] = {
          "metoda1":pd.Series([0.2,0.1,0.3,0.3,0.1],[32,2,8,1,4],name="rating"),
          "metoda2":pd.Series([0.1,0.1,0.2,0.3,0.3],[1,5,32,6,7],name="rating"),
          "metoda3":pd.Series([0.3,0.1,0.2,0.3,0.1],[7,2,77,64,12],name="rating")
          }

    itemsDF: DataFrame = Items.readFromFileMl1m()
    usersDF: DataFrame = Users.readFromFileMl1m()

    # methods parametes
    methodsParamsData:List[tuple] = [['metoda1',100], ['metoda2',80], ['metoda3',60]]
    methodsParamsDF:DataFrame = pd.DataFrame(methodsParamsData, columns=["methodID", "votes"])
    methodsParamsDF.set_index("methodID", inplace=True)

    userID:int = 0
    itemID:int = 7

    historyDF:AHistory = HistoryDF("test01")
    historyDF.insertRecommendation(userID, itemID, 1, True, 10)
    historyDF.insertRecommendation(userID, itemID, 1, True, 20)
    historyDF.insertRecommendation(userID, itemID, 1, True, 30)
    historyDF.insertRecommendation(userID, itemID, 1, False, 40)
    historyDF.print()

    # TODO: What is ARG_SELECTOR?
    aggr:AggrContextFuzzyDHondt = AggrContextFuzzyDHondt(historyDF, {AggrContextFuzzyDHondt.ARG_SELECTOR:TheMostVotedItemSelector({}),
                                                                     AggrContextFuzzyDHondt.ARG_USERS:usersDF,
                                                                     AggrContextFuzzyDHondt.ARG_ITEMS:itemsDF})
    itemIDs = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, userID, N)
    print(itemIDs)
    itemIDs = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, userID, N)
    print(itemIDs)


if __name__ == "__main__":
    os.chdir("..")
    test01()