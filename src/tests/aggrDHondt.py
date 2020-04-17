#!/usr/bin/python3

from typing import List

from pandas.core.series import Series #class

from aggregation.aggrElections import AggrElections #class

from configuration.arguments import Arguments #class

from recommendation.resultOfRecommendation import ResultOfRecommendation #class

from simulation.evaluationTool.simplePositiveFeedback import SimplePositiveFeedback #class

import numpy as np
import pandas as pd

from pandas.core.frame import DataFrame #class

import os


def test01():
    print("Test 01")

    # number of recommended items
    N = 120

    # method results, items=[1,2,4,5,6,7,8,12,32,64,77]
    methodsResultDict = {
          "metoda1":pd.Series([0.2,0.1,0.3,0.3,0.1],[32,2,8,1,4],name="rating"),
          "metoda2":pd.Series([0.1,0.1,0.2,0.3,0.3],[1,5,32,6,7],name="rating"),
          "metoda3":pd.Series([0.3,0.1,0.2,0.3,0.1],[7,2,77,64,12],name="rating")
          }
    #print(methodsResultDict)


    # methods parametes
    methodsParamsData = [['metoda1',100], ['metoda2',80], ['metoda3',60]]
    methodsParamsDF = pd.DataFrame(methodsParamsData, columns=["methodID","votes"])
    methodsParamsDF.set_index("methodID", inplace=True)
    #print(methodsParamsDF)

    aggr:AggrElections = AggrElections(Arguments([]))
    #itemIDs:int = aggr.run(methodsResultDict, methodsParamsDF, N)
    #print(itemIDs)
    itemIDs:List[tuple] = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, N)
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
    #print(methodsResultDict)


    # methods parametes
    methodsParamsData = [['metoda1',0], ['metoda2',0]]
    methodsParamsDF = pd.DataFrame(methodsParamsData, columns=["methodID","votes"])
    methodsParamsDF.set_index("methodID", inplace=True)
    #print(methodsParamsDF)

    aggr:AggrElections = AggrElections(Arguments([]))
    #itemIDs:int = aggr.run(methodsResultDict, methodsParamsDF, N)
    #print(itemIDs)
    itemIDs:List[tuple] = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, N)
    print(itemIDs)

def main():
    print("D'Hondt algorithm")

    #test01()
    test02()


if __name__ == "__main__":
    main()