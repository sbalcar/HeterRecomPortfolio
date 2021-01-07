#!/usr/bin/python3

import os

from configuration.configuration import Configuration #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

from datasets.ml.ratings import Ratings #class

from history.historyDF import HistoryDF #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class

import pandas as pd


def test01():
    print("Test 01")

    print("Running RecommenderCosineCB ML:")

    #cbDataPath:str = Configuration.cbDataFileWithPathTFIDF
    cbDataPath:str = Configuration.cbML1MDataFileWithPathOHE

    ratingsDF:DataFrame = Ratings.readFromFileMl1m()

    ratingsDFTrain:DataFrame = ratingsDF.iloc[0:50000]

    trainDataset:ADataset = DatasetML("test", ratingsDFTrain, pd.DataFrame(), pd.DataFrame())

    args:dict = {
            RecommenderCosineCB.ARG_CB_DATA_PATH: Configuration.cbML1MDataFileWithPathTFIDF,
            RecommenderCosineCB.ARG_USER_PROFILE_SIZE: 5,
            RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY: "max"}
    rec:ARecommender = RecommenderCosineCB("test", args)

    rec.train(HistoryDF("test"), trainDataset)

    ratingsDFUpdate:DataFrame = ratingsDF.iloc[50003:50004]
    #ratingsDFUpdate:DataFrame = ratingsDF.iloc[3:4]
    rec.update(ratingsDFUpdate)

    print(len(rec.userProfiles[331]))

    print("max")
    r:Series = rec.recommend(331, 50, args)
    print(type(r))
    print(r)

    # testing of a non-existent user
    print("mean")
    r:Series =rec.recommend(10000, 50, args)
    print(type(r))
    print(r)



def test03():
    print("Test 03")

    print("Running RecommenderCosineCB ST:")

    dataset:DatasetST = DatasetST.readDatasets()


    args:dict = {
            RecommenderCosineCB.ARG_CB_DATA_PATH: Configuration.cbSTDataFileWithPathTFIDF,
#            RecommenderCosineCB.ARG_CB_DATA_PATH: Configuration.cbSTDataFileWithPathOHE,
            RecommenderCosineCB.ARG_USER_PROFILE_SIZE: 5,
            RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY: "max"}
    rec:ARecommender = RecommenderCosineCB("test", args)

    rec.train(HistoryDF("test"), dataset)

    eventsDFDFUpdate:DataFrame = dataset.eventsDF.iloc[5003:5004]
    rec.update(eventsDFDFUpdate)

    # user with very outdated profile - no recent objects
    r:Series = rec.recommend(3500678, 20, args)
    print(type(r))
    print(r)

    # user with very outdated profile - no recent objects
    r:Series = rec.recommend(3325463, 20, args)
    print(type(r))
    print(r)

    # testing of a non-existent user
    r:Series =rec.recommend(10000, 50, args)
    print(type(r))
    print(r)


if __name__ == "__main__":
    os.chdir("..")

#    test01()
    test03()