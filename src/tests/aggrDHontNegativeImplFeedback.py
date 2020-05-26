#!/usr/bin/python3

from typing import List
from pandas.core.frame import DataFrame #class

from aggregation.aggrDHont import AggrDHont #class
from aggregation.aggrDHontNegativeImplFeedback import AggrDHontNegativeImplFeedback #class

import pandas as pd
from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class
from history.historySQLite import HistorySQLite #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function

from aggregation.tools.penalizationOfResultsByNegImpFeedbackUsingReduceRelevance import PenalizationOfResultsByNegImpFeedbackUsingReduceRelevance #class


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
    historyDF.insertRecommendation(userID, itemID, 1, 0.9, True)
    historyDF.insertRecommendation(userID, itemID, 1, 0.9, True)
    historyDF.insertRecommendation(userID, itemID, 1, 0.9, True)
    historyDF.print()

    ignoringValue:float = historyDF.getIgnoringValue(userID, itemID, limit=3)
    print("IgnoringValue: " + str(ignoringValue))

    aggr:AggrDHont = AggrDHontNegativeImplFeedback(historyDF, {AggrDHontNegativeImplFeedback.ARG_SELECTORFNC:(AggrDHontNegativeImplFeedback.selectorOfTheMostVotedItem,[]),
                                                               AggrDHontNegativeImplFeedback.AGR_LENGTH_OF_HISTORY:10,
                                                               AggrDHontNegativeImplFeedback.AGR_BORDER_NEGATIVE_FEEDBACK:1.0})

    itemIDs:List[tuple] = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, userID, N)
    print(itemIDs)


def test02():
    print("Test 02")

    methodsResultDict:dict[str,pd.Series] = {
          "metoda1":pd.Series([0.2,0.1,0.3,0.3,0.1],[32,2,8,1,4],name="rating"),
          "metoda2":pd.Series([0.1,0.1,0.2,0.3,0.3],[1,5,32,6,7],name="rating"),
          "metoda3":pd.Series([0.3,0.1,0.2,0.3,0.1],[7,2,77,64,12],name="rating")
          }
    print(methodsResultDict)
    print()

    userID:int = 0
    itemID:int = 1

    historyDF:AHistory = HistoryDF("test01")
    historyDF.insertRecommendation(userID, itemID, 0, 0.9, False)
    historyDF.insertRecommendation(userID, itemID, 0, 0.9, False)
    historyDF.insertRecommendation(userID, itemID, 0, 0.9, False)
    #historyDF.print()

    ###################
    maxPenaltyValue:float = 1.2
    minPenaltyValue:float = 0.2
    lengthOfHistory:int = 5

    p = PenalizationOfResultsByNegImpFeedbackUsingReduceRelevance(historyDF, maxPenaltyValue, minPenaltyValue, lengthOfHistory)
    methodsResultDict:dict[str, pd.Series] = p.proportionalRelevanceReduction(methodsResultDict, userID)
    print("methodsResultDict")
    print(methodsResultDict)


    ###################
    i:int = 2
    maxPenaltyValue:float = 1.2
    minPenaltyValue:float = 0.2
    lengthOfHistory:int = 5

    value:float = PenalizationOfResultsByNegImpFeedbackUsingReduceRelevance.getPenaltyLinear2(i, maxPenaltyValue, minPenaltyValue, lengthOfHistory)
    print("value: " + str(value))


    ###################
    minTimeDiff:float = 1.0
    maxTimeDiff:float = 1.5
    minPenalty:float = 0.0
    maxPenalty:float = 1.0

    timeDiff:float = minTimeDiff + 0.25

    value:float = PenalizationOfResultsByNegImpFeedbackUsingReduceRelevance.getPenaltyLinear(timeDiff, minTimeDiff, maxTimeDiff, minPenalty, maxPenalty)
    print("value: " + str(value))

#test01()
test02()