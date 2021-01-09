#!/usr/bin/python3

import os
import sys
from typing import List

from pandas.core.frame import DataFrame #class

import pandas as pd

from evaluationTool.evalToolDHondt import EvalToolDHondt #class
from evaluationTool.evalToolBanditTS import EvalToolBanditTS #class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class

from datasets.ml.items import Items #class


def test01():
    print("Test 01")

    dataset:ADataset = DatasetML.readDatasets()

    #12::Dracula: Dead and Loving It(1995)::Comedy | Horror
    itemsDF = dataset.itemsDF[dataset.itemsDF[Items.COL_MOVIEID] == 12]
    print(itemsDF)

    #458::Geronimo: An American Legend(1993)::Drama | Western
    itemsDF = dataset.itemsDF[dataset.itemsDF[Items.COL_MOVIEID] == 458]
    print(itemsDF)


if __name__ == "__main__":
    os.chdir("..")

    print(sys.version)
    test01()