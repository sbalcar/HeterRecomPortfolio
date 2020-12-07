#!/usr/bin/python3

import csv
from typing import List

from pandas.core.frame import DataFrame #class

import pandas as pd

import os



class CategoryTree:
    COL_CATEGORY_ID = "categoryid"
    COL_PARENT_ID = "parentid"

    @staticmethod
    def readFromFile():
        eventsFile: str = ".." + os.sep + "datasets" + os.sep + "retailrocket" + os.sep + "category_tree.csv"

        categoryTreeDF: DataFrame = pd.read_csv(eventsFile, sep=',', usecols=[0, 1], header=0, encoding="ISO-8859-1", low_memory=False)
        categoryTreeDF.columns = [CategoryTree.COL_CATEGORY_ID, CategoryTree.COL_PARENT_ID]

        return categoryTreeDF




if __name__ == "__main__":

  #np.random.seed(42)
  #random.seed(42)

  os.chdir("..")
  os.chdir("..")

  print(os.getcwd())
  categoryTree:DataFrame = CategoryTree.readFromFile()

  print(categoryTree.head())