#!/usr/bin/python3

import os

from configuration.configuration import Configuration #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.ratings import Ratings #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class

import pandas as pd


def test01():
    print("Test 01")
    os.chdir("..")

    print("Running RecommenderTheMostPopular:")

    cbDataPath:str = Configuration.cbDataFileWithPathTFIDF

    #ratingsDF:DataFrame = Ratings.readFromFileMl100k()
    ratingsDF: DataFrame = Ratings.readFromFileMl1m()


    ratingsDFTrain:DataFrame = ratingsDF.iloc[0:50000]

    rec:ARecommender = RecommenderTheMostPopular("test", {})
    rec.train(pd.DataFrame(), ratingsDFTrain, pd.DataFrame(), pd.DataFrame())

    ratingsDFUpdate:DataFrame = ratingsDF.iloc[50003:50004]
    rec.update(ratingsDFUpdate)

    r:Series = rec.recommend(331, 50, {})
    print(type(r))
    print(r)

    # testing of a non-existent user
    r:Series =rec.recommend(10000, 50, {})
    print(type(r))
    print(r)


test01()