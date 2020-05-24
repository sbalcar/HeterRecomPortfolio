#!/usr/bin/python3

import os

from configuration.configuration import Configuration #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.ratings import Ratings #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class

import pandas as pd


def test01():
    print("Test 01")
    os.chdir("..")

    print("Running RecommenderCosineCB:")

    cbDataPath:str = Configuration.cbDataFileWithPathTFIDF

    #ratingsDF:DataFrame = Ratings.readFromFileMl100k()
    ratingsDF: DataFrame = Ratings.readFromFileMl1m()


    ratingsDFTrain:DataFrame = ratingsDF.iloc[0:50000]

    rec:ARecommender = RecommenderCosineCB({RecommenderCosineCB.ARG_CB_DATA_PATH:cbDataPath})
    rec.train(pd.DataFrame(), ratingsDFTrain, pd.DataFrame(), pd.DataFrame())

    ratingsDFUpdate:DataFrame = ratingsDF.iloc[50003:50004]
    rec.update(ratingsDFUpdate)

    print(len(rec.userProfiles[331]))

    print("max")
    r:Series = rec.recommend(331, 50, "max")
    print(type(r))
    print(r)

    print("mean")
    r:Series =rec.recommend(10000, 50, "mean")
    print(type(r))
    print(r)


test01()