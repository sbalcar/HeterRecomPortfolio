#!/usr/bin/python3

from typing import List
from pandas.core.frame import DataFrame #class

from aggregation.aggrDHont import AggrDHont #class
from aggregation.aggrDHontNegativeImplFeedback import AggrDHontNegativeImplFeedback #class

import pandas as pd
from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function


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

    historyDF:AHistory = HistoryDF()
    historyDF.addRecommendation(1, [7], [True])
    historyDF.addRecommendation(1, [7], [True])
    historyDF.addRecommendation(1, [7], [True])
    historyDF.print()

    ignoringValue:float = historyDF.getIgnoringValue(7, uBehaviourDesc, numberOfItems=N, lengthOfHistory=3)
    print("IgnoringValue: " + str(ignoringValue))

    aggr:AggrDHont = AggrDHontNegativeImplFeedback(uBehaviourDesc, historyDF, {AggrDHontNegativeImplFeedback.ARG_SELECTORFNC:(AggrDHontNegativeImplFeedback.selectorOfTheMostVotedItem,[])})

    itemIDs:List[tuple] = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, N)
    print(itemIDs)


test01()
