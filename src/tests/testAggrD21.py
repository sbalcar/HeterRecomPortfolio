#!/usr/bin/python3

from typing import List
from typing import Dict #class

from pandas.core.frame import DataFrame  #class
from pandas.core.series import Series  #class

from aggregation.aggrFAI import AggrFAI  #class

import pandas as pd
from history.aHistory import AHistory  #class
from history.historyDF import HistoryDF  #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription  #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc  #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc  #function

from aggregation.aggrD21 import AggrD21 #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector  #class


def test01():
    print("Test 01")

    # result items=[21,3,4,5,6,7,8]
    inputItemIDsDict:dict[str,pd.Series] = {
          "input1":pd.Series([1.0,1.0,1.0,1.0,1.0,1.0],[21,3,4,6,7,8],name="rating"),
          "input2":pd.Series([1.0,1.0,1.0,1.0,1.0],[21,2,4,5,9],name="rating"),
          "negative":pd.Series([1.0,1.0],[21,2],name="rating")
          }
    #print(inputItemIDsDict)

    modelDF:DataFrame = DataFrame()


    history:AHistory = HistoryDF("test")

    userID:int = 1
    numberOfItems:int = 20
    argumentsDict:Dict[str, object] = {AggrD21.ARG_RATING_THRESHOLD_FOR_NEG: 1.0}

    aggrD21 = AggrD21(history, argumentsDict)
    aggrD21.runWithResponsibility(inputItemIDsDict, modelDF, userID, numberOfItems, argumentsDict)


if __name__ == "__main__":
    print("AggrFAI algorithm")

    test01()