#!/usr/bin/python3

import os

from pandas.core.frame import DataFrame #class

from configuration.configuration import Configuration #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.ml.ratings import Ratings #class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

from history.historyDF import HistoryDF #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class

import pandas as pd



def test01():
    print("Test 01")

    print("Running RecommenderTheMostPopular ML:")

    ratingsDF: DataFrame = Ratings.readFromFileMl1m()

    ratingsDFTrain:DataFrame = ratingsDF.iloc[0:50000]

    trainDataset:ADataset = DatasetML("test", ratingsDFTrain, pd.DataFrame(), pd.DataFrame())

    rec:ARecommender = RecommenderTheMostPopular("test", {})
    rec.train(HistoryDF("test"), trainDataset)

    ratingsDFUpdate:DataFrame = ratingsDF.iloc[50003:50004]
    rec.update(ratingsDFUpdate)

    r:Series = rec.recommend(331, 50, {})
    print(type(r))
    print(r)

    # testing of a non-existent user
    r:Series =rec.recommend(10000, 50, {})
    print(type(r))
    print(r)


def test02():
    print("Test 02")

    print("Running RecommenderTheMostPopular RR:")

    from datasets.retailrocket.events import Events  # class
    eventsDF:DataFrame = Events.readFromFile()

    dataset:ADataset = DatasetRetailRocket("test", eventsDF, DataFrame(), DataFrame())

    rec:ARecommender = RecommenderTheMostPopular("rTheMostPopular", {})
    rec.train(HistoryDF("test"), dataset)

    recommendation = rec.recommend(1, 20, {})
    print(recommendation)


def test03():
    print("Test 03")

    print("Running RecommenderTheMostPopular ST:")

    from datasets.slantour.events import Events  # class
    eventsDF:DataFrame = Events.readFromFile()

    dataset:ADataset = DatasetST("test", eventsDF, DataFrame())

    rec:ARecommender = RecommenderTheMostPopular("rTheMostPopular", {})
    rec.train(HistoryDF("test"), dataset)

    recommendation = rec.recommend(1, 20, {})
    print(recommendation)


if __name__ == "__main__":

    os.chdir("..")

    #test01()
    #test02()
    test03()

