#!/usr/bin/python3

import os
import time

from typing import List #class
from typing import Dict #class
from configuration.configuration import Configuration #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.ml.ratings import Ratings #class
from datasets.ml.items import Items #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderVSKNN import RecommenderVMContextKNN #class

from history.historyDF import HistoryDF #class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

import pandas as pd


def test01():
    print("Test 01")

    print("Running RecommenderItemBasedKNN ML:")

    ratingsDF:DataFrame = Ratings.readFromFileMl1m()

    filmsDF:DataFrame = Items.readFromFileMl1m()

    # Take only first 50k

    ratingsDFTrain:DataFrame = ratingsDF.iloc[0:800000]


    trainDataset:ADataset = DatasetML("test", ratingsDFTrain, pd.DataFrame(), filmsDF)

    # train recommender
    rec:ARecommender = RecommenderVMContextKNN("test", {RecommenderVMContextKNN.ARG_K:50})
    start = time.time()
    rec.train(HistoryDF("test01"), trainDataset)
    end = time.time()
    print("Time to train: " + str(end - start))

    # get one rating for update
    ratingsDFUpdate:DataFrame = ratingsDF.iloc[800006:800007]


    # get recommendations:
    print("Recommendations before update")
    start = time.time()
    r:Series = rec.recommend(ratingsDFUpdate['userId'].iloc[0], 50, {})
    end = time.time()
    print("Time to train: " + str(end - start))
    
    rec.update(ratingsDFUpdate)

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
    rec:ARecommender = RecommenderVMContextKNN("test", {RecommenderVMContextKNN.ARG_K:50})
    start = time.time()
    rec.train(HistoryDF("test02"), trainDataset)
    end = time.time()
    print("Time to train: " + str(end - start))

    r:Series = rec.recommend(1, 50, {})
    print(r)
    print("================== END OF TEST 02 ======================\n\n\n\n\n")




def test03():
    print("Test 03")

    print("Running RecommenderVSKNN ST:")

    from datasets.slantour.events import Events  # class

    dataset:DatasetST = DatasetST.readDatasets()
    dataset.eventsDF = dataset.eventsDF.iloc[0:200000]

    maxUserID:List[int] = dataset.eventsDF[Events.COL_USER_ID].max()

    # train recommender
    rec:ARecommender = RecommenderVMContextKNN("test", {RecommenderVMContextKNN.ARG_K:50})
    start = time.time()
    rec.train(HistoryDF("test03"), dataset)
    end = time.time()
    print("Time to train: " + str(end - start))   

    eventsDFDFUpdate:DataFrame = dataset.eventsDF.iloc[5003:5004]
    #rec.update(eventsDFDFUpdate, {})


    r:Series = rec.recommend(3342336, 20, {})
    print(type(r))
    print(r)

    r:Series = rec.recommend(2035310, 20, {})
    print(type(r))
    print(r)

    r:Series = rec.recommend(3342341, 20, {})
    print(type(r))
    print(r)

    # testing of a non-existent user
    r:Series =rec.recommend(maxUserID+1, 50, {})
    print(type(r))
    print(r)

if __name__ == "__main__":
    os.chdir("..")

    #test01()
    #test02()
    test03()