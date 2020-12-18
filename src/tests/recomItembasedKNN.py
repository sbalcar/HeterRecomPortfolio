#!/usr/bin/python3

import os

from configuration.configuration import Configuration #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.ml.ratings import Ratings #class
from datasets.ml.items import Items #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderItemBasedKNN import RecommenderItemBasedKNN #class

from history.historyDF import HistoryDF #class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class

import pandas as pd


def test01():
    print("Test 01")

    print("Running RecommenderItemBasedKNN:")

    ratingsDF:DataFrame = Ratings.readFromFileMl1m()

    filmsDF:DataFrame = Items.readFromFileMl1m()

    # Take only first 50k
    ratingsDFTrain:DataFrame = ratingsDF.iloc[0:50000]

    trainDataset:ADataset = DatasetML(ratingsDFTrain, pd.DataFrame(), filmsDF)

    # train recommender
    rec:ARecommender = RecommenderItemBasedKNN("test", {})
    rec.train(HistoryDF("test01"), trainDataset)

    # get one rating for update
    ratingsDFUpdate:DataFrame = ratingsDF.iloc[50005:50006]

    # get recommendations:
    print("Recommendations before update")
    r:Series = rec.recommend(ratingsDFUpdate['userId'].iloc[0], 50, {})

    rec.update(ratingsDFUpdate)

    print("Recommendations after update")
    r: Series = rec.recommend(ratingsDFUpdate['userId'].iloc[0], 50, {})

    print("Test for non-existent user:")
    r:Series =rec.recommend(10000, 50, {})
    print(r)
    print("================== END OF TEST 01 ======================\n\n\n\n\n")


def test02():
    print("Test 02")

    print("Running RecommenderItemBasedKNN:")

    ratingsDF:DataFrame = Ratings.readFromFileMl1m()

    filmsDF:DataFrame = Items.readFromFileMl1m()

    ratingsDFTrain:DataFrame = ratingsDF.iloc[0:1000000]

    trainDataset:ADataset = DatasetML(ratingsDFTrain, pd.DataFrame(), filmsDF)

    # train recommender
    rec:ARecommender = RecommenderItemBasedKNN("test", {})
    rec.train(HistoryDF("test02"), trainDataset)

    r:Series = rec.recommend(1, 50, {})
    print(r)
    print("================== END OF TEST 02 ======================\n\n\n\n\n")


def test03():
    print("Test 03")

#    userID: 23
#    currentItemID: 196
#    repetition: 0

    print("Running RecommenderItemBasedKNN:")

    ratingsDF:DataFrame = Ratings.readFromFileMl1m()
    ratingsSortedDF:DataFrame = ratingsDF.sort_values(by=Ratings.COL_TIMESTAMP)

    filmsDF:DataFrame = Items.readFromFileMl1m()

    print(len(ratingsSortedDF))
    ratingsDFTrain:DataFrame = ratingsSortedDF.iloc[0:900000]

    trainDataset:ADataset = DatasetML(ratingsDFTrain, pd.DataFrame(), filmsDF)

    # train recommender
    rec:ARecommender = RecommenderItemBasedKNN("test", {})
    rec.train(HistoryDF("test03"), trainDataset)

    r:Series = rec.recommend(23, 10, {})
    print(r)
    #print("================== END OF TEST 03 ======================\n\n\n\n\n")

if __name__ == "__main__":
    os.chdir("..")
    test01()
    test02()
    test03()