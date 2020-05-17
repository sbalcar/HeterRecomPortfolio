#!/usr/bin/python3

import time
from typing import List

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

import pandas as pd
from history.historySQLite import HistorySQLite #class
from history.historyDF import HistoryDF #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function


def test01():
    print("Test 01")

    print("Running Two paralel History Databases:")

    history1 = HistorySQLite("databse1")
    history2 = HistorySQLite("databse2")

    # userID, itemID, position, observation, clicked
    history1.insertRecommendation(1, 1, 1, 0.5, True)
    history2.insertRecommendation(2, 1, 1, 0.5, True)

    history1.insertRecommendation(1, 2, 1, 0.5, True)
    history2.insertRecommendation(2, 2, 1, 0.5, True)

    history1.insertRecommendation(1, 3, 1, 0.5, True)
    history2.insertRecommendation(2, 3, 1, 0.5, True)

    # userID, limit
    #r1 = history1.getInteractionCount(0, 10)
    #print(r1)

    # userID, limit=100
    p1 = history1.getPreviousRecomOfUser(1)
    print(p1)

    p2 = history2.getPreviousRecomOfUser(2)
    print(p2)


def test02():
    print("Test 02")

    print("Running of comparing Dataframe vs. Database based History:")

    history1 = HistorySQLite("databse1")
    history2 = HistoryDF("databse2")

    start1 = time.time()

    for i in range(100):
        # userID, itemID, position, observation, clicked
        history1.insertRecommendation(1, i, 1, 0.5, False)

    end1 = time.time()


    start2 = time.time()

    for i in range(100):
        # userID, itemID, position, observation, clicked
        history2.insertRecommendation(1, i, 1, 0.5, False)

    end2 = time.time()


    print()
    print("Time HistorySQLite: " + format(end1 - start1, '.5f') + " s")
    print("Time HistoryDF: " + format(end2 - start2, '.5f') + " s")

    #print(history1.getPreviousRecomOfUser(1))
    print(history1.getPreviousRecomOfUser(1)[0])
    #print(history2.getPreviousRecomOfUser(1))
    print(history2.getPreviousRecomOfUser(1)[0])


#test01()
test02()