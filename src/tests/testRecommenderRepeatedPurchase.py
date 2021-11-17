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
from recommender.recommenderRepeatedPurchase import RecommenderRepeatedPurchase #class

import pandas as pd



def test01():
    print("Test 01")

    print("Running RecommenderRepeatedPurchase RR:")

    from datasets.retailrocket.events import Events  # class
    #eventsDF:DataFrame = Events.readFromFile()
    eventsDF:DataFrame = Events.readFromFileWithFilter(minEventCount=50)
    #print(eventsDF)

    userID:int = 1

    trainSer = pd.Series([1433221523348, userID, Events.EVENT_ADDTOCART, 350688, "Nan"], index=[Events.COL_TIME_STAMP, Events.COL_VISITOR_ID, Events.COL_EVENT, Events.COL_ITEM_ID, Events.EVENT_TRANSACTION])
    trainDF = pd.DataFrame([trainSer])

    dataset:ADataset = DatasetRetailRocket("test", eventsDF, DataFrame(), DataFrame())

    rec:ARecommender = RecommenderRepeatedPurchase("rRepeatedPurchase", {})
    rec.train(HistoryDF("test"), dataset)

    # nejcasteji opakovane kupovane itemy: 119736, 119736, 119736, 213834, 119736, 227311, 382885, 119736, 213834, 119736, 432171, 183756, 119736, 305675, 320130
    update1Ser = pd.Series([1433221523348, userID, Events.EVENT_ADDTOCART, 119736, "Nan"], index=[Events.COL_TIME_STAMP, Events.COL_VISITOR_ID, Events.COL_EVENT, Events.COL_ITEM_ID, Events.EVENT_TRANSACTION])
    update1DF:DataFrame = pd.DataFrame([update1Ser])

    update2Ser = pd.Series([1433221523348, userID, Events.EVENT_ADDTOCART, 213834, "Nan"], index=[Events.COL_TIME_STAMP, Events.COL_VISITOR_ID, Events.COL_EVENT, Events.COL_ITEM_ID, Events.EVENT_TRANSACTION])
    update2DF:DataFrame = pd.DataFrame([update2Ser])

    rec.update(update1DF, {})
    rec.update(update2DF, {})

    recommendationSer:Series = rec.recommend(userID, 20, {})

    print("Recommendation:")
    print(recommendationSer)



if __name__ == "__main__":
    os.chdir("..")
    print(os
          .getcwd())

    test01()
    #test02()
    #test03()
