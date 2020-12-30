#!/usr/bin/python3
import sys 
  
# appending a path 
sys.path.append('..')
 
import os

from configuration.configuration import Configuration #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class

from datasets.ml.ratings import Ratings #class
from datasets.ml.items import Items #class
from datasets.ml.users import Users #class

from history.historyDF import HistoryDF #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderBPRMF import RecommenderBPRMF #class

import pandas as pd


def test01():
    print("Test 01")

    print("Running Recommender BPRMF on ML:")

    dataset:DatasetML = DatasetML.readDatasets()

    # Take only first 500k
    trainDataset:DatasetML = DatasetML("test", dataset.ratingsDF.iloc[0:499965], dataset.usersDF, dataset.itemsDF)

    # train recommender
    rec:ARecommender = RecommenderBPRMF("test",{
                    RecommenderBPRMF.ARG_FACTORS: 20,
                    RecommenderBPRMF.ARG_ITERATIONS: 50,
                    RecommenderBPRMF.ARG_LEARNINGRATE: 0.003,
                    RecommenderBPRMF.ARG_REGULARIZATION: 0.003})

    rec.train(HistoryDF("test01"), trainDataset)

    # get one rating for update
    ratingsDFUpdate:DataFrame = dataset.ratingsDF.iloc[499965:503006]

    # get recommendations:
    print("Recommendations before update")
    print(rec._movieFeaturesMatrixLIL[:,ratingsDFUpdate['userId'].iloc[0]  ].getnnz() )
    
    r:Series = rec.recommend(ratingsDFUpdate['userId'].iloc[0], 50, {})
    print(r);
    for i in range(ratingsDFUpdate.shape[0]):
        rUp = ratingsDFUpdate.iloc[i:i+1,:]
        rec.update(rUp)

    print("Recommendations after update")
    print(rec._movieFeaturesMatrixLIL[:,ratingsDFUpdate['userId'].iloc[0]  ].getnnz() )
    
    r: Series = rec.recommend(ratingsDFUpdate['userId'].iloc[0], 50, {})
    print(r);
    
    print("Test for non-existent user:")
    r:Series =rec.recommend(10000, 50, {})
    print(r);
    
    print("================== END OF TEST 01 ======================\n\n\n\n\n")


def test02():
    print("Test 02")

    print("Running Recommender BPRMF on ML:")

    dataset:DatasetML = DatasetML.readDatasets()

    # Take only first 500k
    trainDataset:DatasetML = DatasetML("test", dataset.ratingsDF.iloc[0:800000], dataset.usersDF, dataset.itemsDF)

    print(dataset.ratingsDF.iloc[655924:655926])

    # train recommender
    rec:ARecommender = RecommenderBPRMF("test",{
                    RecommenderBPRMF.ARG_FACTORS: 20,
                    RecommenderBPRMF.ARG_ITERATIONS: 50,
                    RecommenderBPRMF.ARG_LEARNINGRATE: 0.003,
                    RecommenderBPRMF.ARG_REGULARIZATION: 0.003})
    rec.train(HistoryDF("test02"), trainDataset)

    # get recommendations:
    print("Recommendations before update")
    r:Series = rec.recommend(23, 50, {})
    print(r)

    print("================== END OF TEST 02 ======================\n\n\n\n\n")


def test03():
    print("Test 03")

    print("Running Recommender BPRMF on RR:")

    dataset:DatasetRetailRocket = DatasetRetailRocket.readDatasets()

    trainDataset:DatasetRetailRocket = dataset

    # train recommender
    rec:ARecommender = RecommenderBPRMF("test",{
                    RecommenderBPRMF.ARG_FACTORS: 20,
                    RecommenderBPRMF.ARG_ITERATIONS: 50,
                    RecommenderBPRMF.ARG_LEARNINGRATE: 0.003,
                    RecommenderBPRMF.ARG_REGULARIZATION: 0.003})
    rec.train(HistoryDF("test03"), trainDataset)

    r:Series = rec.recommend(23, 50, {})
    print(r)



if __name__ == "__main__":
    os.chdir("..")

#    test01()
#    test02()
    test03()