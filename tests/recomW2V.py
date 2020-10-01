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

    ratingsDF:DataFrame = Ratings.readFromFileMl1m()

    ratingsDFTrain:DataFrame = ratingsDF.iloc[0:50000]

    id:str = "ml1mDiv90"
    #id:str = "test"

    rec:ARecommender = RecommenderW2V(id, {RecommenderW2V.ARG_TRAIN_VARIANT:"posneg"})
    #rec:ARecommender = RecommenderW2V(id, {RecommenderW2V.ARG_TRAIN_VARIANT:"positive"})
    rec.train(HistoryDF("w2v"), ratingsDFTrain, pd.DataFrame(), pd.DataFrame())

    ratingsDFUpdate:DataFrame = ratingsDF.iloc[50003:50004]
    rec.update(ratingsDFUpdate)

    print(len(rec.userProfiles[331]))


    r:Series = rec.recommend(331, 50, {RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"max"})
    print("max")
    print(type(r))
    print(r)


    r:Series = rec.recommend(331, 50, {RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"window10"})
    print("window10")
    print(type(r))
    print(r)


    r:Series = rec.recommend(10000, 50, {RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"window10"})
    print("mean")
    print(type(r))
    print(r)


test01()