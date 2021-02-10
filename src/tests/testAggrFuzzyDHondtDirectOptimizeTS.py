#!/usr/bin/python3

import os
from typing import List
from pandas.core.frame import DataFrame #class

from aggregation.aggrDHondtDirectOptimizeThompsonSampling import AggrDHondtDirectOptimizeThompsonSampling #class

import pandas as pd
import math
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
          "method1":pd.Series([0.9, 0.5, 0.4, 0.2],[1,2,3,4],name="rating"),
          "method2":pd.Series([0.2, 0.9, 0.01],[3,4,5],name="rating"),
          "method3":pd.Series([0.01, 0.9, 0.9], [5, 6, 7], name="rating")
    }


    # methods parametes
    methodsParamsData:List[tuple] = [['method1', 5, 10, 1, 1], ['method2', 5, 12, 1, 1], ['method3', 6, 13, 1, 1]]
    modelDF:DataFrame = pd.DataFrame(methodsParamsData, columns=["methodID", "r", "n", "alpha0", "beta0"])
    modelDF.set_index("methodID", inplace=True)    
    
    
    #modelDF:DataFrame = DataFrame({"votes": [0.5, 0.4, 0.1]}, index=["method1", "method2", "method3"])
    #print(modelDF)

    userID:int = 0

    aggr:AggrDHondtDirectOptimizeThompsonSampling = AggrDHondtDirectOptimizeThompsonSampling(
        HistoryDF(""),
        {AggrDHondtDirectOptimizeThompsonSampling.ARG_SELECTOR:TheMostVotedItemSelector({}), AggrDHondtDirectOptimizeThompsonSampling.ARG_DISCOUNT_FACTOR:"uniform"}
        )
    itemIDs:List[tuple] = aggr.run(methodsResultDict, modelDF, userID, N)
    print("itemIDs:" + str(itemIDs))

    aggr:AggrDHondtDirectOptimizeThompsonSampling = AggrDHondtDirectOptimizeThompsonSampling(
        HistoryDF(""),
        {AggrDHondtDirectOptimizeThompsonSampling.ARG_SELECTOR:TheMostVotedItemSelector({}), AggrDHondtDirectOptimizeThompsonSampling.ARG_DISCOUNT_FACTOR:"DCG"}
        )
    itemIDs:List[tuple] = aggr.run(methodsResultDict, modelDF, userID, N)
    print("itemIDs:" + str(itemIDs))

    aggr:AggrDHondtDirectOptimizeThompsonSampling = AggrDHondtDirectOptimizeThompsonSampling(
        HistoryDF(""),
        {AggrDHondtDirectOptimizeThompsonSampling.ARG_SELECTOR:TheMostVotedItemSelector({}), AggrDHondtDirectOptimizeThompsonSampling.ARG_DISCOUNT_FACTOR:"PowerLaw"}
        )
    itemIDs:List[tuple] = aggr.run(methodsResultDict, modelDF, userID, N)
    print("itemIDs:" + str(itemIDs))    

if __name__ == '__main__':


    test01()
