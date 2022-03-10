#!/usr/bin/python3

import os
from tqdm import tqdm
import scipy.sparse as sp

import pickle
import numpy as np

from typing import List # class
from typing import Dict  # class

from pandas.core.frame import DataFrame  # class
from pandas.core.series import Series  # class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderBPRMF import RecommenderBPRMF #class
from recommender.recommenderBPRMFImplicit import RecommenderBPRMFImplicit #class

from history.aHistory import AHistory #class
from history.historyHierDF import HistoryHierDF  # class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailRocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

from datasets.ml.ratings import Ratings  # class

from typing import Dict
from numpy import ndarray

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class
from batchDefinition.inputRecomRRDefinition import InputRecomRRDefinition #class
from batchDefinition.inputRecomSTDefinition import InputRecomSTDefinition #class


def test01():
    print("Test 01")

    print("Running Recommender BPRMF on ML:")

    batchID:str = "batchID"

    dataset:ADataset = DatasetML.readDatasets()

    history:AHistory = HistoryHierDF(["aa"])

    argumentsDict: Dict[str, object] = {
        RecommenderBPRMF.ARG_EPOCHS: 2,
        RecommenderBPRMF.ARG_FACTORS: 10,
        RecommenderBPRMF.ARG_LEARNINGRATE: 0.05,
        RecommenderBPRMF.ARG_UREGULARIZATION: 0.0025,
        RecommenderBPRMF.ARG_BREGULARIZATION: 0,
        RecommenderBPRMF.ARG_PIREGULARIZATION: 0.0025,
        RecommenderBPRMF.ARG_NIREGULARIZATION: 0.00025}

    r:ARecommender = RecommenderBPRMF(batchID, argumentsDict)
    r.train(history, dataset)

    numberOfItems:int = 20
    userId:int = 1
    res = r.recommend(userId, numberOfItems, argumentsDict)
    print(res)
    userId:int = 2
    res = r.recommend(userId, numberOfItems, argumentsDict)
    print(res)


def test02():
    print("Test 02")

    print("Running Recommender BPRMF on RR:")

    batchID:str = "batchID"

    dataset:ADataset = DatasetRetailRocket.readDatasetsWithFilter(50)

    history:AHistory = HistoryHierDF(["aa"])

    argumentsDict:Dict[str, object] = {
        RecommenderBPRMF.ARG_EPOCHS: 2,
        RecommenderBPRMF.ARG_FACTORS: 10,
        RecommenderBPRMF.ARG_LEARNINGRATE: 0.05,
        RecommenderBPRMF.ARG_UREGULARIZATION: 0.0025,
        RecommenderBPRMF.ARG_BREGULARIZATION: 0,
        RecommenderBPRMF.ARG_PIREGULARIZATION: 0.0025,
        RecommenderBPRMF.ARG_NIREGULARIZATION: 0.00025}

    r:ARecommender = RecommenderBPRMF(batchID, argumentsDict)
    r.train(history, dataset)

    numberOfItems:int = 20
    userId:int = 1118731
    res = r.recommend(userId, numberOfItems, argumentsDict)
    print(res)
    userId:int = 85734
    res = r.recommend(userId, numberOfItems, argumentsDict)
    print(res)


def test03():
    print("Test 03")

    print("Running Recommender BPRMF on ST:")

    batchID:str = "batchID"

    dataset:ADataset = DatasetST.readDatasets()

    history:AHistory = HistoryHierDF(["aa"])

    argumentsDict:Dict[str, object] = {
        RecommenderBPRMF.ARG_EPOCHS: 2,
        RecommenderBPRMF.ARG_FACTORS: 10,
        RecommenderBPRMF.ARG_LEARNINGRATE: 0.05,
        RecommenderBPRMF.ARG_UREGULARIZATION: 0.0025,
        RecommenderBPRMF.ARG_BREGULARIZATION: 0,
        RecommenderBPRMF.ARG_PIREGULARIZATION: 0.0025,
        RecommenderBPRMF.ARG_NIREGULARIZATION: 0.00025}

    r:ARecommender = RecommenderBPRMF(batchID, argumentsDict)
    r.train(history, dataset)

    numberOfItems:int = 20
    userId:int = 62302
    res = r.recommend(userId, numberOfItems, argumentsDict)
    print(res)
    userId:int = 3462303
    res = r.recommend(userId, numberOfItems, argumentsDict)
    print(res)


def test11():
    print("Test 11")

    print("Running Recommender BPRMF on ML:")

    batchID:str = "batchID"

    trainDataset:ADataset
    testDataset:ADataset
    trainDataset, testDataset = DatasetML.readDatasets().divideDataset(90)

    testUserIDs:ndarray = testDataset.ratingsDF[Ratings.COL_USERID].unique()

    history:AHistory = HistoryHierDF(["aa"])

    numberOfItems:int = 20

    rd:RecommenderDescription = InputRecomMLDefinition.exportRDescBPRMF()
    #rd:RecommenderDescription = InputRecomMLDefinition.exportRDescBPRMFIMPLf20i20lr0003r01()
    #rd:RecommenderDescription = InputRecomMLDefinition.exportRDescTheMostPopular()
    #rd:RecommenderDescription = InputRecomMLDefinition.exportRDescKNN()
    #rd:RecommenderDescription = InputRecomMLDefinition.exportRDescCosineCBcbdOHEupsweightedMeanups3()
    #rd:RecommenderDescription = InputRecomMLDefinition.exportRDescW2Vtpositivei50000ws1vs32upsweightedMeanups3()
    #rd:RecommenderDescription = InputRecomMLDefinition.exportRDescW2Vtpositivei50000ws1vs64upsweightedMeanups7()

    r:ARecommender = rd.exportRecommender("aaa")
    argumentsDict:Dict = rd.getArguments()

    r.train(history, trainDataset)

    numberOfHit:int = 0
    for userIdI in testUserIDs:
        recI:Series = r.recommend(int(userIdI), numberOfItems, argumentsDict)
        recItemIDsI:List[int] = [i for i in recI.keys()]

        windowItemIds:List[int] = testDataset.ratingsDF.loc[testDataset.ratingsDF[Ratings.COL_USERID] == userIdI][Ratings.COL_MOVIEID].unique()
        itemIdsHitted:List[int] = list(set(recItemIDsI) & set(windowItemIds))
        numberOfHit += len(itemIdsHitted)

    print("")
    print("numberOfHit: " + str(numberOfHit))



def test12():
    print("Test 12")

    print("Running Recommender BPRMF on RR:")
    from datasets.retailrocket.events import Events  # class

    batchID:str = "batchID"

    trainDataset:ADataset
    testDataset:ADataset
    trainDataset, testDataset = DatasetRetailRocket.readDatasetsWithFilter(50).divideDataset(90)

    testUserIDs:ndarray = testDataset.eventsDF[Events.COL_VISITOR_ID].unique()

    history:AHistory = HistoryHierDF(["aa"])

    numberOfItems:int = 20

    rd:RecommenderDescription = InputRecomRRDefinition.exportRDescBPRMFIMPL()
    #rd:RecommenderDescription = InputRecomRRDefinition.exportRDescBPRMF()
    #rd:RecommenderDescription = InputRecomRRDefinition.exportRDescTheMostPopular()
    rd:RecommenderDescription = InputRecomRRDefinition.exportRDescKNN()
    #rd:RecommenderDescription = InputRecomRRDefinition.exportRDescCosineCBcbdOHEupsweightedMeanups3()
    #rd:RecommenderDescription = InputRecomRRDefinition.exportRDescW2Vtpositivei50000ws1vs32upsweightedMeanups3()
    #rd:RecommenderDescription = InputRecomRRDefinition.exportRDescW2Vtpositivei50000ws1vs64upsweightedMeanups7()

    r:ARecommender = rd.exportRecommender("aaa")
    argumentsDict:Dict = rd.getArguments()

    r.train(history, trainDataset)

    numberOfHit:int = 0
    for userIdI in testUserIDs[0:500]:
        recI:Series = r.recommend(int(userIdI), numberOfItems, argumentsDict)
        recItemIDsI:List[int] = [i for i in recI.keys()]

        windowItemIds:List[int] = testDataset.eventsDF.loc[testDataset.eventsDF[Events.COL_VISITOR_ID] == userIdI][Events.COL_ITEM_ID].unique()
        itemIdsHitted:List[int] = list(set(recItemIDsI) & set(windowItemIds))
        numberOfHit += len(itemIdsHitted)

    print("")
    print("numberOfHit: " + str(numberOfHit))


def test13():
    print("Test 13")

    print("Running Recommender BPRMF on ST:")
    from datasets.slantour.events import Events  # class

    batchID:str = "batchID"

    trainDataset:ADataset
    testDataset:ADataset
    trainDataset, testDataset = DatasetST.readDatasets().divideDataset(90)

    testUserIDs:ndarray = testDataset.eventsDF[Events.COL_USER_ID].unique()

    history:AHistory = HistoryHierDF(["aa"])

    numberOfItems:int = 20

    rd:RecommenderDescription = InputRecomSTDefinition.exportRDescBPRMFIMPLf50i20lr01r003()
    rd:RecommenderDescription = InputRecomSTDefinition.exportRDescBPRMF()
    #rd:RecommenderDescription = InputRecomSTDefinition.exportRDescTheMostPopular()
    #rd:RecommenderDescription = InputRecomSTDefinition.exportRDescKNN()
    #rd:RecommenderDescription = InputRecomSTDefinition.exportRDescCosineCBcbdOHEupsweightedMeanups5()
    #rd:RecommenderDescription = InputRecomSTDefinition.exportRDescW2Vtalli100000ws1vs32upsmaxups1()
    #rd:RecommenderDescription = InputRecomSTDefinition.exportRDescW2talli200000ws1vs64upsweightedMeanups5()

    r:ARecommender = rd.exportRecommender("aaa")
    argumentsDict:Dict = rd.getArguments()

    r.train(history, trainDataset)

    numberOfHit:int = 0
    for userIdI in testUserIDs[0:1000]:
        recI:Series = r.recommend(int(userIdI), numberOfItems, argumentsDict)
        recItemIDsI:List[int] = [i for i in recI.keys()]

        windowItemIds:List[int] = testDataset.eventsDF.loc[testDataset.eventsDF[Events.COL_USER_ID] == userIdI][Events.COL_OBJECT_ID].unique()
        itemIdsHitted:List[int] = list(set(recItemIDsI) & set(windowItemIds))
        numberOfHit += len(itemIdsHitted)

    print("")
    print("numberOfHit: " + str(numberOfHit))



if __name__ == "__main__":
    os.chdir("..")

    #test01()
    #test02()
    #test03()

    #test11()
    #test12()
    test13()