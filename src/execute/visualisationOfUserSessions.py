#!/usr/bin/python3

import os
import random
import numpy as np

from typing import List

from pandas.core.frame import DataFrame #class

from configuration.configuration import Configuration #class
#from datasets.behaviours import Behaviours #class

import matplotlib.pyplot as plt

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailRocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

from datasets.ml.ratings import Ratings #class
from datasets.ml.users import Users #class
from datasets.ml.items import Items #class
from datasets.ml.behavioursML import BehavioursML #class
from datasets.retailrocket.behavioursRR import BehavioursRR #class
from datasets.slantour.behavioursST import BehavioursST #class

from pandas.core.frame import DataFrame #class


def visualizationML():
    print("visualizationML")

    from datasets.ml.ratings import Ratings  # class

    # dataset reading
    dataset:ADataset = DatasetML.readDatasets()
    print(dataset.ratingsDF.head())
    userIdsWithDuplicites:List[int] = dataset.ratingsDF[Ratings.COL_USERID].tolist()
    userIds:List[int] = list(set(userIdsWithDuplicites))

    plt.hist(userIdsWithDuplicites, len(userIds), None, fc='none', lw=1.5, histtype='step')
    plt.ticklabel_format(style='plain')
    plt.show()


def visualizationRR():
    print("visualizationRR")

    from datasets.retailrocket.events import Events  # class

    # dataset reading
    dataset:ADataset = DatasetRetailRocket.readDatasetsWithFilter(minEventCount=50)
    print(dataset.eventsDF.head())
    userIdsWithDuplicites:List[int] = dataset.eventsDF[Events.COL_VISITOR_ID].tolist()
    userIds:List[int] = list(set(userIdsWithDuplicites))

    plt.hist(userIdsWithDuplicites, len(userIds), None, fc='none', lw=1.5, histtype='step')
    plt.ticklabel_format(style='plain')
    plt.show()

def visualizationST():
    print("visualizationST")

    from datasets.slantour.events import Events  # class

    # dataset reading
    dataset:ADataset = DatasetST.readDatasets()
    #dataset:ADataset = DatasetST.readDatasetsSkipOutlierUsers(500)
    print(dataset.eventsDF.head())
    userIdsWithDuplicites:List[int] = dataset.eventsDF[Events.COL_USER_ID].tolist()
    userIds:List[int] = list(set(userIdsWithDuplicites))

    plt.hist(userIdsWithDuplicites, len(userIds), None, fc='none', lw=1.5, histtype='step')
    plt.ticklabel_format(style='plain')
    plt.show()

def visualizationST2():
    print("visualizationST2")

    from datasets.slantour.events import Events  # class

    dataset:ADataset = DatasetST.readDatasets()
    eventsDF:Events = dataset.eventsDF

    eventsDF.sort_values(by=[Events.COL_START_DATE_TIME], inplace=True)


    userIdsWithDuplicites:List[int] = eventsDF[Events.COL_USER_ID].tolist()
    userIds:List[int] = list(set(userIdsWithDuplicites))
    print("userIdsCount: " + str(len(userIds)))

    #return
    for userIdI in userIds:
        indexesOfUserI:List[int] = eventsDF[eventsDF[Events.COL_USER_ID] == userIdI].index.tolist()
        print(indexesOfUserI)
        distanceI:int = max(indexesOfUserI) - min(indexesOfUserI)
        print(distanceI)

if __name__ == "__main__":
    os.chdir("..")
    print(os.getcwd())

    #visualizationML()
    #visualizationRR()
    visualizationST2()

