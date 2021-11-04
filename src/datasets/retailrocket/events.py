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

    EVENT_VIEW = "view"
    EVENT_ADDTOCART = "addtocart"
    EVENT_TRANSACTION = "transaction"

    @staticmethod
    def getColNameUserID():
        return Events.COL_VISITOR_ID

    @staticmethod
    def getColNameItemID():
        return Events.COL_ITEM_ID


    @staticmethod
    def readFromFile():
        eventsFile: str = ".." + os.sep + "datasets" + os.sep + "retailrocket" + os.sep + "events.csv"

        eventsDF:DataFrame = pd.read_csv(eventsFile, sep=',', usecols=[0, 1, 2, 3, 4], header=0, encoding="ISO-8859-1", low_memory=False)
        eventsDF.columns = [Events.COL_TIME_STAMP, Events.COL_VISITOR_ID, Events.COL_EVENT, Events.COL_ITEM_ID, Events.COL_TRANSACTION_ID]

        return eventsDF


    @staticmethod
    def readFromFileWithFilter(minEventCount:int):

        eventsDF:DataFrame = Events.readFromFile()

        # filtrace eventu - vymazeme uzivatele, kteri maji mene nez K zaznamu
        userIdAndTimestampDF:DataFrame[int, int] = eventsDF.groupby(
            [Events.COL_VISITOR_ID], as_index=False)[Events.COL_TIME_STAMP].count()

        userIdAndTimestampSelDF:DataFrame[int, int] = userIdAndTimestampDF.loc[
            userIdAndTimestampDF[Events.COL_TIME_STAMP] > minEventCount]
        # print(userIdAndTimestampSelDF)
        userIDsSel:List[int] = list(userIdAndTimestampSelDF[Events.COL_VISITOR_ID].unique())

        eventsSelDF:DataFrame = eventsDF.loc[eventsDF[Events.COL_VISITOR_ID].isin(userIDsSel)]

        return eventsSelDF




if __name__ == "__main__":

  #np.random.seed(42)
  #random.seed(42)

  os.chdir("..")
  os.chdir("..")

  print(os.getcwd())
  evens:DataFrame = Events.readFromFile()
  evens = evens.loc[evens[Events.COL_EVENT] == "transaction"]

  print(evens.head(100))