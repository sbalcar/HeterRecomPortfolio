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
from datasets.datasetRetailRocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

import pandas as pd


def test01():
    print("Test 01")

    print("Running RecommenderItemBasedKNN ML:")

    ratingsDF:DataFrame = Ratings.readFromFileMl1m()

    filmsDF:DataFrame = Items.readFromFileMl1m()

    # Take only first 50k
    ratingsDFTrain:DataFrame = ratingsDF.iloc[0:50000]

    trainDataset:ADataset = DatasetML("test", ratingsDFTrain, pd.DataFrame(), filmsDF)

    # train recommender
    rec:ARecommender = RecommenderItemBasedKNN("test", {})
    rec.train(HistoryDF("test01"), trainDataset)

    # get one rating for update
    ratingsDFUpdate:DataFrame = ratingsDF.iloc[50005:50006]

    # get recommendations:
    print("Recommendations before update")
    r:Series = rec.recommend(ratingsDFUpdate['userId'].iloc[0], 50, {})

    rec.update(ratingsDFUpdate, {})

    print("Recommendations after update")
    r: Series = rec.recommend(ratingsDFUpdate['userId'].iloc[0], 50, {})

    print("Test for non-existent user:")
    r:Series =rec.recommend(10000, 50, {})
    print(r)
    print("================== END OF TEST 01 ======================\n\n\n\n\n")


def test02():
    print("Test 02")

    print("Running RecommenderItemBasedKNN ML:")

    ratingsDF:DataFrame = Ratings.readFromFileMl1m()

    filmsDF:DataFrame = Items.readFromFileMl1m()

    ratingsDFTrain:DataFrame = ratingsDF.iloc[0:1000000]

    trainDataset:ADataset = DatasetML("test", ratingsDFTrain, pd.DataFrame(), filmsDF)

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

    print("Running RecommenderItemBasedKNN ML:")

    ratingsDF:DataFrame = Ratings.readFromFileMl1m()
    ratingsSortedDF:DataFrame = ratingsDF.sort_values(by=Ratings.COL_TIMESTAMP)

    filmsDF:DataFrame = Items.readFromFileMl1m()

    print(len(ratingsSortedDF))
    ratingsDFTrain:DataFrame = ratingsSortedDF[0:900000]
    ratingsDFTrain: DataFrame = ratingsDFTrain[ratingsDFTrain[Ratings.COL_USERID] != 23]
    ratingsDFTrain: DataFrame = ratingsDFTrain[ratingsDFTrain[Ratings.COL_MOVIEID] != 10]


    print(ratingsDFTrain.head(25))

    trainDataset:ADataset = DatasetML("test", ratingsDFTrain, pd.DataFrame(), filmsDF)


    # train recommender
    rec:ARecommender = RecommenderItemBasedKNN("test1", {})
    rec.train(HistoryDF("test03"), trainDataset)


    uDdata = [[23, 10, 4, 10000]]
    uDF: DataFrame = pd.DataFrame(uDdata, columns=[Ratings.COL_USERID, Ratings.COL_MOVIEID, Ratings.COL_RATING, Ratings.COL_TIMESTAMP])

    rec.update(uDF, {})


    r:Series = rec.recommend(23, 10, {})
    print(r)
    print("\n")

    r:Series = rec.recommend(23, 10, {})
    print(r)

    print("================== END OF TEST 03 ======================\n\n\n\n\n")


def test11():
    print("Test 11")

    print("Running RecommenderItemBasedKNN RR:")

    from datasets.retailrocket.events import Events  # class
    #eventsDF:DataFrame = Events.readFromFile()
    eventsDF:DataFrame = Events.readFromFileWithFilter(minEventCount=50)

    dataset:ADataset = DatasetRetailRocket("test", eventsDF, DataFrame(), DataFrame())

    rec:ARecommender = RecommenderItemBasedKNN("test", {})
    print("train")
    rec.train(HistoryDF("test"), dataset)

    uDF:DataFrame = DataFrame([eventsDF.iloc[9000]])
    print(uDF)
    rec.update(uDF, {})

    recommendation = rec.recommend(1093035, 20, {})
    print("Recommendation:")
    print(recommendation)

    print("================== END OF TEST 04 ======================\n\n\n\n\n")


def test21():
    print("Test 21")

    print("Running RecommenderItemBasedKNN ST:")

    from datasets.slantour.events import Events  # class
    eventsDF:DataFrame = Events.readFromFile()

    dataset:ADataset = DatasetST("test", eventsDF, DataFrame())

    rec:ARecommender = RecommenderItemBasedKNN("test", {})
    rec.train(HistoryDF("test"), dataset)

    uDF:DataFrame = DataFrame([eventsDF.iloc[9000]])
    print(uDF)
    rec.update(uDF, {})

    r = rec.recommend(3325463, 20, {})
    print(r)

    print("================== END OF TEST 05 ======================\n\n\n\n\n")


def test22():
    print("Test 22")

    print("Running RecommenderItemBasedKNN ST:")

    from datasets.slantour.events import Events  # class

    userID1:int = 1
    userID2:int = 2
    userID3:int = 3

    trainEventsDF:DataFrame = DataFrame(columns=[Events.COL_USER_ID, Events.COL_OBJECT_ID])

    trainEventsDF.loc[0] = [userID1, 101]
    trainEventsDF.loc[1] = [userID1, 102]
    trainEventsDF.loc[2] = [userID1, 103]
    trainEventsDF.loc[3] = [userID1, 104]
    trainEventsDF.loc[4] = [userID2, 101]
    trainEventsDF.loc[5] = [userID2, 102]

    print(trainEventsDF.head(10))

    trainDataset:ADataset = DatasetST("test", trainEventsDF, DataFrame())

    rec:ARecommender = RecommenderItemBasedKNN("test", {})
    rec.train(HistoryDF("test"), trainDataset)


    print("update 1:")
    updateEvents1DF:DataFrame = DataFrame(columns=trainEventsDF.columns)
    updateEvents1DF.loc[0] = [userID1, 105]
    print(updateEvents1DF.head())
    rec.update(updateEvents1DF, {})

    print("update 2:")
    updateEvents2DF:DataFrame = DataFrame(columns=trainEventsDF.columns)
    updateEvents2DF.loc[0] = [userID3, 106]
    print(updateEvents2DF.head())
    rec.update(updateEvents2DF, {})


    print("recommend:")
    r = rec.recommend(userID2, 10, {})
    print(r)

    print("================== END OF TEST 06 ======================\n\n\n\n\n")


if __name__ == "__main__":
    os.chdir("..")

    # ML
    #test01()
    #test02()
    #test03()

    # RR
    test11()

    # ST
    #test21()
    #test22()