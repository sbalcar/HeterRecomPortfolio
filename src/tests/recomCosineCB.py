#!/usr/bin/python3

import os
import time
from typing import List

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.ratings import Ratings #class

from recommender.aRecommender import ARecommender #class

import pandas as pd
from history.historySQLite import HistorySQLite #class
from history.historyDF import HistoryDF #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function

from recommender.recommenderCosineCB import RecommenderCosineCB #class

def test01():
    print("Test 01")

    print("Running RecommenderCosineCB:")

    cbDataPath:str = ".."  + os.sep + ".." + os.sep + "data" + os.sep + "cbDataOHE.txt"


    #ratingsFile:str = ".." + os.sep + ".." + os.sep + "datasets" + os.sep + "ml-100k" + os.sep + "u.data"

    #ratingsDF:DataFrame = pd.read_csv(ratingsFile, sep='\t', header=None)
    #ratingsDF.columns = [Ratings.COL_USERID, Ratings.COL_MOVIEID, Ratings.COL_RATING, Ratings.COL_TIMESTAMP]

    ratingsFile:str = ".." + os.sep + ".." + os.sep + "datasets" + os.sep + "ml-1m" + os.sep + "ratings.dat"

    ratingsDF: DataFrame = pd.read_csv(ratingsFile, sep=':', usecols=[0, 2, 4, 6], header=None)
    ratingsDF.columns = [Ratings.COL_USERID, Ratings.COL_MOVIEID, Ratings.COL_RATING, Ratings.COL_TIMESTAMP]


    #ratingsDF:DataFrame = Ratings.readFromFileMl100k()
    ratingsDFTrain:DataFrame = ratingsDF.iloc[0:50000]

    rec:ARecommender = RecommenderCosineCB({RecommenderCosineCB.ARG_CB_DATA_PATH:cbDataPath})
    rec.train(pd.DataFrame(), ratingsDFTrain, pd.DataFrame(), pd.DataFrame())

    ratingsDFUpdate:DataFrame = ratingsDF.iloc[50003:50004]
    rec.update(ratingsDFUpdate)

    print(len(rec.userProfiles[331]))

    #r:Series = rec.recommend(331, 50, "max")
    #print(r)

    r:Series =rec.recommend(10000, 50, "mean")
    print(r)


test01()