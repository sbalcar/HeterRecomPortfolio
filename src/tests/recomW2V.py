#!/usr/bin/python3

import os
import time
from typing import List

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.ratings import Ratings #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderW2V import RecommenderW2V #class

import pandas as pd
from history.historySQLite import HistorySQLite #class
from history.historyDF import HistoryDF #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function

from recommender.recommenderCosineCB import RecommenderCosineCB #class

def test01():
    print("Test 01")
    os.chdir("..")

    print("Running RecommenderW2V:")

    ratingsDF: DataFrame = Ratings.readFromFileMl1m()

    ratingsDFTrain:DataFrame = ratingsDF.iloc[0:50000]

    rec:ARecommender = RecommenderW2V({RecommenderW2V.ARG_TRAIN_VARIANT:"all"})
    rec.train(HistoryDF("w2v"), ratingsDFTrain, pd.DataFrame(), pd.DataFrame())

    ratingsDFUpdate:DataFrame = ratingsDF.iloc[50003:50004]
    rec.update(ratingsDFUpdate)

    print(len(rec.userProfiles[331]))


    print("max")
    r:Series = rec.recommend(331, 50, "max")
    print(r)

    print("mean")
    r:Series = rec.recommend(10000, 50, "mean")
    print(r)


test01()