#!/usr/bin/python3

import os

from typing import Dict #class

from pandas.core.series import Series #class

from aggregation.negImplFeedback.penalUsingProbability import PenalUsingProbability #class
from aggregation.negImplFeedback.aPenalization import APenalization #class

from history.aHistory import AHistory #class
from history.historyHierDF import HistoryHierDF #class

from aggregation.negImplFeedback.penalUsingReduceRelevance import penaltyLinear #function


def test01():
    print("Test 01")
    os.chdir("..")

    userID:int = 1

    history:AHistory = HistoryHierDF("databse1")

    # userID, itemID, position, observation, clicked
    history.insertRecommendation(userID, 100, 1, 0.5, False)
    history.insertRecommendation(userID, 100, 1, 0.5, True)
    history.insertRecommendation(userID, 100, 1, 0.5, True)
    history.insertRecommendation(userID, 100, 1, 0.5, True)


    recommendationDict:Dict[int,float] = {100:0.35, 125:0.25, 95:0.15, 45:0.1, 78:0.05, 68:0.05, 32:0.02, 6:0.01, 18:0.01, 47:0.01}
    recommendationSrs:Series = Series(recommendationDict)

    penalty:APenalization = PenalUsingProbability(penaltyLinear, [0.8, 0.2, 100], penaltyLinear, [1.0, 0.2, 100], 100)
    pRecommendationSrs:Series = penalty.runOneMethodPenalization(userID, recommendationSrs, history)

    print(pRecommendationSrs)



if __name__ == "__main__":
    os.chdir("..")

    test01()
