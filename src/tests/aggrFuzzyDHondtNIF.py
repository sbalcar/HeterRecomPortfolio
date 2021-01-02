#!/usr/bin/python3

import os
from typing import List
from pandas.core.frame import DataFrame #class

from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt #class
from aggregation.aggrFuzzyDHondtINF import AggrFuzzyDHondtINF #class

import pandas as pd
from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function

#from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.penalUsingReduceRelevance import PenalizationOfResultsByNegImpFeedbackUsingReduceRelevance #class
from aggregation.negImplFeedback.penalUsingReduceRelevance import PenalUsingReduceRelevance #class

from aggregation.negImplFeedback.penalUsingReduceRelevance import penaltyLinear #function

from aggregation.negImplFeedback.penalUsingReduceRelevance import PenalUsingReduceRelevance #class
from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

def test01():
    print("Test 01")

    # number of recommended items
    N = 120

    #a = observationalLinearProbabilityFnc(0.1, 0.9, 5)
    #print(a)

    uBehaviourDesc:UserBehaviourDescription = UserBehaviourDescription(observationalLinearProbabilityFnc, [0.1, 0.9])

    # method results, items=[1,2,4,5,6,7,8,12,32,64,77]
    methodsResultDict:dict[str,pd.Series] = {
          "metoda1":pd.Series([0.2,0.1,0.3,0.3,0.1],[32,2,8,1,4],name="rating"),
          "metoda2":pd.Series([0.1,0.1,0.2,0.3,0.3],[1,5,32,6,7],name="rating"),
          "metoda3":pd.Series([0.3,0.1,0.2,0.3,0.1],[7,2,77,64,12],name="rating")
          }
    #print(methodsResultDict)

    # methods parametes
    methodsParamsData:List[tuple] = [['metoda1',100], ['metoda2',80], ['metoda3',60]]
    methodsParamsDF:DataFrame = pd.DataFrame(methodsParamsData, columns=["methodID","votes"])
    methodsParamsDF.set_index("methodID", inplace=True)
    #print(methodsParamsDF)

    userID:int = 0
    itemID:int = 7

    historyDF:AHistory = HistoryDF("test01")
    historyDF.insertRecommendation(userID, itemID, 1, True, None)
    historyDF.insertRecommendation(userID, itemID, 1, True, None)
    historyDF.insertRecommendation(userID, itemID, 1, True, None)
    historyDF.print()

    #ignoringValue:float = historyDF.getIgnoringValue(userID, itemID, limit=3)
    #print("IgnoringValue: " + str(ignoringValue))

    penalization:APenalization = PenalUsingReduceRelevance(penaltyLinear, [0.8, 0.2, 3], penaltyLinear, [1.0, 0.2, 3], 3)

    aggr:AggrFuzzyDHondt = AggrFuzzyDHondtINF(historyDF, {AggrFuzzyDHondtINF.ARG_SELECTOR:TheMostVotedItemSelector({}),
                                                          AggrFuzzyDHondtINF.ARG_PENALTY_TOOL:penalization})

    itemIDs:List[tuple] = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, userID, N)
    print(itemIDs)



if __name__ == '__main__':
    os.chdir("..")

    test01()
