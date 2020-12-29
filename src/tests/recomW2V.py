#!/usr/bin/python3

import os

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.ml.ratings import Ratings #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderW2V import RecommenderW2V #class

import pandas as pd
from history.historyDF import HistoryDF #class


def test01():
    print("Test 01")

    print("Running RecommenderW2V:")

    ratingsDF: DataFrame = Ratings.readFromFileMl1m()

    ratingsDFTrain:DataFrame = ratingsDF.iloc[0:50000]

    trainDataset:ADataset = DatasetML("test", ratingsDFTrain, pd.DataFrame(), pd.DataFrame())


    rec:ARecommender = RecommenderW2V("RecommenderW2V",{
                            RecommenderW2V.ARG_ITERATIONS: 50000,
                            RecommenderW2V.ARG_TRAIN_VARIANT: 'positive',
                            RecommenderW2V.ARG_USER_PROFILE_SIZE: -1,
                            RecommenderW2V.ARG_USER_PROFILE_STRATEGY: 'weightedMean',
                            RecommenderW2V.ARG_VECTOR_SIZE: 128,
                            RecommenderW2V.ARG_WINDOW_SIZE: 5})

    rec.train(HistoryDF("w2v"), trainDataset)

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


if __name__ == "__main__":
    os.chdir("..")
    #os.chdir("..")

    test01()