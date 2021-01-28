#!/usr/bin/python3

from typing import List
from pandas.core.frame import DataFrame #class

from aggregation.aggrWeightedAVG import AggrWeightedAVG #class

import pandas as pd
from history.historyDF import HistoryDF #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function

from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

from input.inputAggrDefinition import InputAggrDefinition  #class
from input.modelDefinition import ModelDefinition

from aggregation.negImplFeedback.aPenalization import APenalization #class


def test01():
    print("Test 01")

    # number of recommended items
    N = 120

    # method results, items=[1,2,4,5,6,7,8,12,32,64,77]
    methodsResultDict:dict[str,pd.Series] = {
          "metoda1":pd.Series([0.2,0.1,0.3,0.3,0.1],[32,2,8,1,4],name="rating"),
          "metoda2":pd.Series([0.1,0.1,0.2,0.3,0.3],[1,5,32,6,7],name="rating"),
          "metoda3":pd.Series([0.3,0.1,0.2,0.3,0.1],[7,2,77,64,12],name="rating")
          }
    #print(methodsResultDict)

    # methods parametes
    portfolioModelData:List[tuple] = [['metoda1',100], ['metoda2',80], ['metoda3',60]]
    portfolioModel:DataFrame = pd.DataFrame(portfolioModelData, columns=["methodID","votes"])
    portfolioModel.set_index("methodID", inplace=True)

    sumMethodsVotes: float = portfolioModel["votes"].sum()
    for methodIdI in portfolioModel.index:
        portfolioModel.loc[methodIdI, "votes"] = portfolioModel.loc[methodIdI, "votes"] / sumMethodsVotes

    userID:int = 1

    aggr:AggrWeightedAVG = AggrWeightedAVG(HistoryDF(""), {})
    #itemIDs:int = aggr.run(methodsResultDict, methodsParamsDF, N)
    #print(itemIDs)
    itemIDs:List[tuple] = aggr.runWithResponsibility(methodsResultDict, portfolioModel, userID, N)
    print(itemIDs)



if __name__ == "__main__":
    print("Weighted AVG aggregation")

    test01()
    # [(1, {'metoda1': 0.0, 'metoda2': 0.0}), (2, {'metoda1': 0, 'metoda2': 0.0}), (3, {'metoda1': 0.0, 'metoda2': 0}), (4, {'metoda1': 0, 'metoda2': 0.0}), (5, {'metoda1': 0.0, 'metoda2': 0}), (6, {'metoda1': 0, 'metoda2': 0.0}), (7, {'metoda1': 0.0, 'metoda2': 0}), (8, {'metoda1': 0, 'metoda2': 0.0}), (9, {'metoda1': 0.0, 'metoda2': 0}), (10, {'metoda1': 0, 'metoda2': 0.0})]


