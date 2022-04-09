#!/usr/bin/python3

import os

from typing import List
from typing import Dict

from pandas.core.frame import DataFrame #class

from configuration.configuration import Configuration #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.ml.ratings import Ratings #class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailRocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

from history.historyDF import HistoryDF #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.recommenderRepeatedPurchase import RecommenderRepeatedPurchase #class

from recommender.recommenderClusterBaserd import RecommenderClusterBased #class

import pandas as pd



def test01():
    print("Test 01")

    dataset:ADataset = DatasetML.readDatasets()
    itemsDF:DataFrame = dataset.itemsDF

    from datasets.ml.items import Items #class
    #print(Items.getAllGenres())
    #print(itemsDF.head())

    #print(itemsDF[itemsDF[Items.COL_GENRES] == Items.GENRE_COMEDY].head())
    #print(itemsDF[itemsDF[Items.COL_GENRES].str.contains(Items.GENRE_COMEDY)].head())

    dataset.getTheMostPopularOfGenre(Items.GENRE_COMEDY)



    argsDict:dict = {RecommenderClusterBased.ARG_RECOMMENDER_NUMERIC_ID:1}
    r:ARecommender = RecommenderClusterBased("test", argsDict)
    r.train(HistoryDF("test01"), dataset)

    userID:int = 1
    rItemIds:List[int] = r.recommend(userID, 20, argsDict)

    print(rItemIds)





if __name__ == "__main__":
    os.chdir("..")
    print(os
          .getcwd())

    test01()
    #test02()
    #test03()
