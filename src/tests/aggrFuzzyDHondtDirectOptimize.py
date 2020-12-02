#!/usr/bin/python3

from typing import List
from pandas.core.frame import DataFrame #class

from aggregation.aggrFuzzyDHondtDirectOptimize import AggrFuzzyDHondtDirectOptimize #class

import pandas as pd
from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function

from aggregation.negImplFeedback.penalUsingReduceRelevance import penaltyLinear #function

from aggregation.negImplFeedback.penalUsingReduceRelevance import PenalUsingReduceRelevance #class
from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

def test01():
    print("Test 01")

    # number of recommended items
    N:int = 5

    methodsResultDict:dict = {
          "method1":pd.Series([0.5, 0.5, 0.4, 0.2],[1,2,3,4],name="rating"),
          "method2":pd.Series([0.9, 0.5, 0.1],[5,1,6],name="rating"),
          "method3":pd.Series([0.7, 0.3, 0.9], [3, 4, 6], name="rating")
    }


    # methods parametes
    modelDF:DataFrame = DataFrame({"votes": [0.5, 0.4, 0.1]}, index=["method1", "method2", "method3"])
    #print(modelDF)

    userID:int = 0

    aggr:AggrFuzzyDHondtDirectOptimize = AggrFuzzyDHondtDirectOptimize({
                    AggrFuzzyDHondtDirectOptimize.ARG_SELECTOR:TheMostVotedItemSelector({})
                    })

    #itemIDs:List[tuple] = aggr.runWithResponsibility(methodsResultDict, modelDF, userID, N)

    # [1, 3, 5, 2, 4]
    itemIDs:List[tuple] = aggr.run(methodsResultDict, modelDF, userID, N)

    print("itemIDs:" + str(itemIDs))



test01()
