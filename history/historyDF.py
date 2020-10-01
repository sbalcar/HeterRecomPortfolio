#!/usr/bin/python3

import datetime

from typing import List
from pandas.core.series import Series #class
from pandas.core.frame import DataFrame #class

import pandas as pd

from history.aHistory import AHistory #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class


class HistoryDF(AHistory):

    USER_ID = "userID"
    ITEM_ID = "itemID"
    POSITION = "position"
    OBSERVATION = "observation"
    CLICKED = "clicked"
    TIMESTAMP = "timestamp"


    def __init__(self, dbName:str):

        self.dbName:str = dbName

        historyData:pd.DataFrame = []
        self._historyDF:pd.DataFrame = pd.DataFrame(historyData, columns=[
            self.USER_ID, self.ITEM_ID, self.POSITION, self.OBSERVATION, self.CLICKED, self.TIMESTAMP])


    def insertRecommendation(self, userID:int, itemID:int, position:int, clicked:bool, timestamp=datetime.datetime.now()):

        new_row:dict = {self.USER_ID:userID, self.ITEM_ID:itemID, self.POSITION:position, self.CLICKED:clicked, self.TIMESTAMP:timestamp}

        self._historyDF = self._historyDF.append(new_row, ignore_index=True)


    def getPreviousRecomOfUser(self, userID:int, limit:int=100):

        uDF:DataFrame = self._historyDF[self._historyDF[self.USER_ID] == userID]

        result:List[tuple] = []
        for indexI, rowI  in uDF.tail(limit).iterrows():
            result.append((indexI, rowI[self.USER_ID], rowI[self.ITEM_ID], rowI[self.POSITION], rowI[self.OBSERVATION], rowI[self.CLICKED], rowI[self.TIMESTAMP]))

        return result


    def getPreviousRecomOfUserAndItem(self, userID:int, itemID:int, limit:int=100):

        uDF:DataFrame = self._historyDF[self._historyDF[self.USER_ID] == userID]
        uiDF:DataFrame = uDF[uDF[self.ITEM_ID] == itemID]

        result: List[tuple] = []
        for indexI, rowI in uiDF.tail(limit).iterrows():
            result.append((indexI, rowI[self.USER_ID], rowI[self.ITEM_ID], rowI[self.POSITION], rowI[self.OBSERVATION],
                           rowI[self.CLICKED], rowI[self.TIMESTAMP]))

        return result


    def isObjectClicked(self, userID:int, itemID:int, limit:int=100):
        hDF:DataFrame = self._historyDF.tail(limit)
        uDF:DataFrame = hDF[hDF[self.USER_ID] == userID]
        uiDF:DataFrame = uDF[uDF[self.ITEM_ID] == itemID]

        cuiDF:DataFrame = uiDF[uiDF[self.CLICKED]]

        return len(cuiDF.index)


    def delete(self, numberOfUserRecommendationToKeep:int):
        self._historyDF = self._historyDF.tail(numberOfUserRecommendationToKeep)


    def deletePreviousRecomOfUser(self, userID:int, numberOfUserRecommendationToKeep:int):
        userIDs:List[int] = list(set([self._historyDF[indexI, self.USER_ID] for indexI in len(self._historyDF.index)]))

        histOfUsers:List[DataFrame] = []

        userIdI:int
        for userIdI in userIDs:
            uDFI:DataFrame = self._historyDF[self._historyDF[self.USER_ID] == userID]
            histOfUsers.append(uDFI)

        self._historyDF = pd.concat(histOfUsers)


    def print(self):
        print(self._historyDF.head(10))