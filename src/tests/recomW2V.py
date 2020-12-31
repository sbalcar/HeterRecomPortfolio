#!/usr/bin/python3

import os

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

from datasets.ml.ratings import Ratings #class
from datasets.ml.items import Items #class
from datasets.ml.users import Users #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderW2V import RecommenderW2V #class

import pandas as pd
from history.historyDF import HistoryDF #class


def test01():
    print("Test 01")

    print("Running RecommenderW2V ML:")

    ratingsDF:DataFrame = Ratings.readFromFileMl1m()

    ratingsDFTrain:DataFrame = ratingsDF.iloc[0:50000]

    trainDataset:ADataset = DatasetML("ml1mDiv90", ratingsDFTrain, pd.DataFrame(), pd.DataFrame())

    argsDict:dict = {
        RecommenderW2V.ARG_ITERATIONS: 50000,
        RecommenderW2V.ARG_TRAIN_VARIANT: 'positive',
        RecommenderW2V.ARG_USER_PROFILE_SIZE: -1,
        RecommenderW2V.ARG_USER_PROFILE_STRATEGY: 'weightedMean',
        RecommenderW2V.ARG_VECTOR_SIZE: 128,
        RecommenderW2V.ARG_WINDOW_SIZE: 5}
    rec:ARecommender = RecommenderW2V("RecommenderW2V", argsDict)

    rec.train(HistoryDF("w2v"), trainDataset)

    ratingsDFUpdate:DataFrame = ratingsDF.iloc[50003:50004]
    rec.update(ratingsDFUpdate)

    print(len(rec.userProfiles[331]))


    r:Series = rec.recommend(331, 50, argsDict)
    print("max")
    print(type(r))
    print(r)


    r:Series = rec.recommend(10000, 50, argsDict)
    print("mean")
    print(type(r))
    print(r)


def test02():
    print("Test 02")

    print("Running RecommenderW2V RR:")

    dataset:DatasetRetailRocket = DatasetRetailRocket.readDatasets()

    trainDataset:DatasetRetailRocket = dataset

    eventsDF:DataFrame = dataset.eventsDF

    # train recommender
    argsDict: dict = {
        RecommenderW2V.ARG_ITERATIONS: 50000,
        RecommenderW2V.ARG_TRAIN_VARIANT: 'posneg',
        RecommenderW2V.ARG_USER_PROFILE_SIZE: -1,
        RecommenderW2V.ARG_USER_PROFILE_STRATEGY: 'weightedMean',
        RecommenderW2V.ARG_VECTOR_SIZE: 128,
        RecommenderW2V.ARG_WINDOW_SIZE: 5}
    rec: ARecommender = RecommenderW2V("RecommenderW2V", argsDict)

    rec.train(HistoryDF("test03"), trainDataset)

    uDF:DataFrame = DataFrame([eventsDF.iloc[9000]])
    print(uDF)
    rec.update(uDF)

    r:Series = rec.recommend(23, 50, {})
    print(r)


def test03():
    print("Test 03")

    print("Running RecommenderW2V ST:")

    dataset:DatasetST = DatasetST.readDatasets()

    trainDataset:DatasetST = dataset

    eventsDF:DataFrame = dataset.eventsDF

    # train recommender
    argsDict: dict = {
        RecommenderW2V.ARG_ITERATIONS: 50000,
        RecommenderW2V.ARG_TRAIN_VARIANT: 'posneg',
        RecommenderW2V.ARG_USER_PROFILE_SIZE: -1,
        RecommenderW2V.ARG_USER_PROFILE_STRATEGY: 'weightedMean',
        RecommenderW2V.ARG_VECTOR_SIZE: 128,
        RecommenderW2V.ARG_WINDOW_SIZE: 5}
    rec: ARecommender = RecommenderW2V("RecommenderW2V", argsDict)

    rec.train(HistoryDF("test03"), trainDataset)

    uDF:DataFrame = DataFrame([eventsDF.iloc[9000]])
    print(uDF)
    rec.update(uDF)

    r:Series = rec.recommend(23, 50, {})
    print(r)





if __name__ == "__main__":
    os.chdir("..")

#    test01()
#    test02()
    test03()