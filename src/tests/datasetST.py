#!/usr/bin/python3

import os
import sys
from typing import List

from pandas.core.frame import DataFrame #class

import pandas as pd

from evaluationTool.evalToolDHondt import EvalToolDHondt #class
from evaluationTool.evalToolBanditTS import EvalToolBanditTS #class

from datasets.aDataset import ADataset #class
from datasets.datasetST import DatasetST #class

from datasets.ml.items import Items #class
from datasets.slantour.serials import Serials #class
from datasets.slantour.events import Events #class

def test01():
    print("Test 01")

    dataset:ADataset = DatasetST.readDatasets()

    print(dataset.serialsDF.columns)
    print(dataset.serialsDF.head(10))

    print(dataset.eventsDF.columns)
    print(dataset.eventsDF.head(10))


def test02():
    print("Test 02")

    dataset:ADataset = DatasetST.readDatasets()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    eventsDF:DataFrame = dataset.eventsDF
    eventsDF = eventsDF.loc[eventsDF[Events.COL_OBJECT_ID] != 0]

    print(eventsDF.columns)
    print(eventsDF[[Events.COL_VISIT_ID, Events.COL_USER_ID, Events.COL_OBJECT_ID]].head(100))

#0    1        3325463  104
#1    2        3293771  3282
#2    3        3342115  6874
#4    5        3342137  1613
#9    10       3333510  6933
#12   13       3342114  7106

def test03():
    print("Test 03")

    dataset:ADataset = DatasetST.readDatasets()

    print(dataset.getTheMostSold())

    #eventsDF:DataFrame = dataset.eventsDF
    #eventsDF = eventsDF.loc[eventsDF[Events.COL_OBJECT_ID] != 0]



if __name__ == "__main__":
    os.chdir("..")

    #test01()
    #test02()
    test03()