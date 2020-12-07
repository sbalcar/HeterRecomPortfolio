#!/usr/bin/python3

import csv
from typing import List

from pandas.core.frame import DataFrame  # class

import pandas as pd

import os


class ItemProperties:
    COL_TIME_STAMP = "timestamp"
    COL_ITEM_ID = "itemid"
    COL_PROPERTY = "property"
    COL_VALUE = "value"

    @staticmethod
    def readFromFile():
        itemProperties1DF:DataFrame = ItemProperties.__readFromFile("item_properties_part1.csv")

        itemProperties2DF:DataFrame = ItemProperties.__readFromFile("item_properties_part2.csv")

        return pd.concat([itemProperties1DF, itemProperties2DF])

    @staticmethod
    def __readFromFile(fileName:str):
        itemPropertiesFile:str = ".." + os.sep + "datasets" + os.sep + "retailrocket" + os.sep + fileName

        itemPropertiesDF: DataFrame = pd.read_csv(itemPropertiesFile, sep=',', usecols=[0, 1, 2, 3], header=0, encoding="ISO-8859-1", low_memory=False)
        itemPropertiesDF.columns = [ItemProperties.COL_TIME_STAMP, ItemProperties.COL_ITEM_ID, ItemProperties.COL_PROPERTY, ItemProperties.COL_VALUE]

        return itemPropertiesDF


if __name__ == "__main__":
    # np.random.seed(42)
    # random.seed(42)

    os.chdir("..")
    os.chdir("..")

    print(os.getcwd())
    itemProperties: DataFrame = ItemProperties.readFromFile()

    print(itemProperties.head())
