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
from datasets.datasetST import DatasetST #class

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
    
    r:Series = rec.recommend(ratingsDFUpdate['userId'].iloc[0], 50, {})
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

    eventsDF:DataFrame = dataset.eventsDF

    # train recommender
    rec:ARecommender = RecommenderBPRMF("test",{
                    RecommenderBPRMF.ARG_FACTORS: 20,
                    RecommenderBPRMF.ARG_ITERATIONS: 50,
                    RecommenderBPRMF.ARG_LEARNINGRATE: 0.003,
                    RecommenderBPRMF.ARG_REGULARIZATION: 0.003})
    rec.train(HistoryDF("test03"), trainDataset)

    uDF:DataFrame = DataFrame([eventsDF.iloc[9000]])
    print(uDF)
    rec.update(uDF)

    r:Series = rec.recommend(23, 50, {})
    print(r)


def test04():
    print("Test 04")

    print("Running Recommender BPRMF on SL:")
    from datasets.slantour.events import Events  # class

    dataset:DatasetST = DatasetST.readDatasets()

    trainDataset:DatasetST = dataset

    eventsDF:DataFrame = dataset.eventsDF

    uIDMax:int = eventsDF[Events.COL_USER_ID].max()
    print("uIDMax: " + str(uIDMax))

    iIDMax:int = eventsDF[Events.COL_OBJECT_ID].max()
    print("iIDMax: " + str(iIDMax))

    # train recommender
    rec:ARecommender = RecommenderBPRMF("test",{
                    RecommenderBPRMF.ARG_FACTORS: 20,
                    RecommenderBPRMF.ARG_ITERATIONS: 50,
                    RecommenderBPRMF.ARG_LEARNINGRATE: 0.003,
                    RecommenderBPRMF.ARG_REGULARIZATION: 0.003})
    rec.train(HistoryDF("test04"), trainDataset)

    uDF1:DataFrame = DataFrame([eventsDF.iloc[9000]])
    print(uDF1)
    rec.update(uDF1)

    userID1:int = uIDMax + 1
    itemID1:int = iIDMax + 1
    itemID2:int = iIDMax + 2

    # update with unknown user and unknown item
    uDF2:DataFrame = DataFrame(columns=[Events.COL_USER_ID, Events.COL_OBJECT_ID])
    uDF2.loc[0] = [userID1, itemID1]
    print(uDF2)
    rec.update(uDF2)

    # update with unknown item
    uDF3:DataFrame = DataFrame(columns=[Events.COL_USER_ID, Events.COL_OBJECT_ID])
    uDF3.loc[0] = [userID1, itemID2]
    print(uDF3)
    rec.update(uDF3)

    r:Series = rec.recommend(23, 50, {})
    print(r)



def test05():
    print("Test 05")

    import numpy as np
    from scipy.sparse import csr_matrix
    csr_matrix((3, 4), dtype=np.int8).toarray()

    #row = np.array([0, 0, 1, 2, 2, 2])
    row = [0, 0, 1, 2, 2, 2]
    #col = np.array([0, 2, 2, 0, 1, 2])
    col = [0, 2, 2, 0, 1, 2]
    #data = np.array([1, 2, 3, 4, 5, 6])
    data = [1, 2, 3, 4, 5, 6]

    #sparseRatingsCSR:csr_matrix
    sparseRatingsCSR:csr_matrix = csr_matrix((data, (row, col)), shape=(30, 30)) #.toarray()
    print(type(sparseRatingsCSR))
    print(sparseRatingsCSR)
    print()

    sparseRatingsCSR[10,10] = -1
    print(sparseRatingsCSR)
    print()

    itemFeaturesMatrixLIL = sparseRatingsCSR.tolil()
    print(type(itemFeaturesMatrixLIL))
    print(sparseRatingsCSR.toarray())



if __name__ == "__main__":
    os.chdir("..")

#    test01()
#    test02()
#    test03()
    test04()
#    test05()