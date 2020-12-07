#!/usr/bin/python3

import csv
from typing import List

from pandas.core.frame import DataFrame #class

import pandas as pd

import os



class Events:
    COL_TIME_STAMP = "timestamp"
    COL_VISITOR_ID = "visitorid"
    COL_EVENT = "event"
    COL_ITEM_ID = "itemid"
    COL_TRANSACTION_ID = "transactionid"

    @staticmethod
    def readFromFile():
        eventsFile: str = ".." + os.sep + "datasets" + os.sep + "retailrocket" + os.sep + "events.csv"

        eventsDF: DataFrame = pd.read_csv(eventsFile, sep=',', usecols=[0, 1, 2, 3, 4], header=0, encoding="ISO-8859-1", low_memory=False)
        eventsDF.columns = [Events.COL_TIME_STAMP, Events.COL_VISITOR_ID, Events.COL_EVENT, Events.COL_ITEM_ID, Events.COL_TRANSACTION_ID]

        return eventsDF




if __name__ == "__main__":

  #np.random.seed(42)
  #random.seed(42)

  os.chdir("..")
  os.chdir("..")

  print(os.getcwd())
  evens:DataFrame = Events.readFromFile()

  print(evens.head())