#!/usr/bin/python3

from typing import List
from pandas.core.frame import DataFrame #class

from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt #class
from aggregation.aggrFuzzyDHondtINF import AggrFuzzyDHondtINF #class

import pandas as pd
from history.historyDF import HistoryDF #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function

from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

from input.inputAggrDefinition import InputAggrDefinition, ModelDefinition  #class

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


    aggr:AggrFuzzyDHondt = AggrFuzzyDHondt(HistoryDF(""), {AggrFuzzyDHondt.ARG_SELECTOR:TheMostVotedItemSelector({})})
    #itemIDs:int = aggr.run(methodsResultDict, methodsParamsDF, N)
    #print(itemIDs)
    itemIDs:List[tuple] = aggr.runWithResponsibility(methodsResultDict, portfolioModel, N)
    print(itemIDs)



def test02():
    print("Test 02")

    # number of recommended items
    N = 120

    # method results, items=[1,2,3,4,5,6,7,8,9,10]
    methodsResultDict = {
          "metoda1":pd.Series([0.2,0.2,0.2,0.2,0.2],[1,3,5,7,9],name="rating"),
          "metoda2":pd.Series([0.2,0.2,0.2,0.2,0.2],[2,4,6,8,10],name="rating"),
          }

    # methods parametes
    #methodsParamsData:List[tuple] = [['metoda1',0], ['metoda2',0]]
    methodsParamsData:List[tuple] = [['metoda1',1], ['metoda2',1]]
    methodsParamsDF:DataFrame = pd.DataFrame(methodsParamsData, columns=["methodID","votes"])
    methodsParamsDF.set_index("methodID", inplace=True)

    #aggr:AggrDHont = AggrDHont(HistoryDF(), {AggrDHont.ARG_SELECTORFNC:(AggrDHont.selectorOfTheMostVotedItem,[])})
    #aggr:AggrDHont = AggrDHont(HistoryDF(), {AggrDHont.ARG_SELECTORFNC:(AggrDHont.selectorOfRouletteWheelRatedItem,[])})
    #aggr:AggrDHont = AggrDHont(HistoryDF(), {AggrDHont.ARG_SELECTORFNC:(AggrDHont.selectorOfRouletteWheelExpRatedItem,[1])})

    pToolOLin0802HLin1002: APenalization = InputAggrDefinition.exportPenaltyToolOLin0802HLin1002(20)

    aggr:AggrFuzzyDHondt = AggrFuzzyDHondtINF(HistoryDF(""),
                                              {AggrFuzzyDHondtINF.ARG_SELECTOR:RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT:1}),
                                               AggrFuzzyDHondtINF.ARG_PENALTY_TOOL:pToolOLin0802HLin1002})

    ##itemIDs:int = aggr.run(methodsResultDict, methodsParamsDF, N)
    itemIDs:int = aggr.run(methodsResultDict, methodsParamsDF, N)
    #itemIDs:List[tuple] = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, N)
    print(itemIDs)


if __name__ == "__main__":
    print("D'Hondt algorithm")

    test01()
    # [(7, {'metoda1': 0, 'metoda2': 24.0, 'metoda3': 18.0}), (1, {'metoda1': 30.0, 'metoda2': 8.0, 'metoda3': 0}), (32, {'metoda1': 20.0, 'metoda2': 16.0, 'metoda3': 0}), (8, {'metoda1': 30.0, 'metoda2': 0, 'metoda3': 0}), (6, {'metoda1': 0, 'metoda2': 24.0, 'metoda3': 0}), (64, {'metoda1': 0, 'metoda2': 0, 'metoda3': 18.0}), (2, {'metoda1': 10.0, 'metoda2': 0, 'metoda3': 6.0}), (77, {'metoda1': 0, 'metoda2': 0, 'metoda3': 12.0}), (4, {'metoda1': 10.0, 'metoda2': 0, 'metoda3': 0}), (5, {'metoda1': 0, 'metoda2': 8.0, 'metoda3': 0}), (12, {'metoda1': 0, 'metoda2': 0, 'metoda3': 6.0})]

    test02()
    # [(1, {'metoda1': 0.0, 'metoda2': 0.0}), (2, {'metoda1': 0, 'metoda2': 0.0}), (3, {'metoda1': 0.0, 'metoda2': 0}), (4, {'metoda1': 0, 'metoda2': 0.0}), (5, {'metoda1': 0.0, 'metoda2': 0}), (6, {'metoda1': 0, 'metoda2': 0.0}), (7, {'metoda1': 0.0, 'metoda2': 0}), (8, {'metoda1': 0, 'metoda2': 0.0}), (9, {'metoda1': 0.0, 'metoda2': 0}), (10, {'metoda1': 0, 'metoda2': 0.0})]


