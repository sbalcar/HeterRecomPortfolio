#!/usr/bin/python3

import datetime

from typing import List
from pandas.core.series import Series #class
from pandas.core.frame import DataFrame #class

import pandas as pd
import numpy as np

from history.aHistory import AHistory #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class


class HistoryHierDF(AHistory):

    USER_ID = "userID"
    ITEM_ID = "itemID"
    POSITION = "position"
    OBSERVATION = "observation"
    CLICKED = "clicked"
    TIMESTAMP = "timestamp"


    def __init__(self, dbName:str):

        self.dbName:str = dbName

        self._historyDict:dict = {}


    def insertRecommendation(self, userID:int, itemID:int, position:int, uObservation:float, clicked:bool, timestamp=datetime.datetime.now()):

        if not userID in self._historyDict.keys():
            uHistoryDF:DataFrame = pd.DataFrame(columns=[
                    self.ITEM_ID, self.POSITION, self.OBSERVATION, self.CLICKED, self.TIMESTAMP])
            self._historyDict[userID] = uHistoryDF

        newRow:dict = {self.ITEM_ID:itemID, self.POSITION:position, self.OBSERVATION:uObservation, self.CLICKED:clicked, self.TIMESTAMP:timestamp}
        self._historyDict[userID] = self._historyDict[userID].append(newRow, ignore_index=True)


    def getPreviousRecomOfUser(self, userID:int, limit:int=100):

        if not userID in self._historyDict.keys():
            return []

        uDF:DataFrame = self._historyDict[userID]

        result:List[tuple] = []
        indexI:int
        rowI:DataFrame
        for indexI, rowI  in uDF.tail(limit).iterrows():
            result.append((indexI, rowI[self.USER_ID], rowI[self.ITEM_ID], rowI[self.POSITION], rowI[self.OBSERVATION], rowI[self.CLICKED], rowI[self.TIMESTAMP]))

        return result


    def getPreviousRecomOfUserAndItem(self, userID:int, itemID:int, limit:int=100):

        if not userID in self._historyDict.keys():
            return []

        uDF:DataFrame = self._historyDict[userID]
        uiDF:DataFrame = uDF[uDF[self.ITEM_ID] == itemID]

        result: List[tuple] = []
        indexI:int
        rowI:DataFrame
        for indexI, rowI in uiDF.tail(limit).iterrows():
            result.append((indexI, userID, rowI[self.ITEM_ID], rowI[self.POSITION], rowI[self.OBSERVATION],
                           rowI[self.CLICKED], rowI[self.TIMESTAMP]))

        return result


    def getInteractionCount(self, userID:int, limit:int):

        if not userID in self._historyDict.keys():
            return 0

        uDF:DataFrame = self._historyDict[userID]
        return len(uDF.index)


    def isObjectClicked(self, userID:int, itemID:int, limit:int=100):

        if not userID in self._historyDict.keys():
            return False

        uDF:DataFrame = self._historyDict[userID]
        uiDF:DataFrame = uDF[uDF[self.ITEM_ID] == itemID]

        cuiDF:DataFrame = uiDF[uiDF[self.CLICKED]]

        return len(cuiDF.index)


    def delete(self, numberOfUserRecommendationToKeep:int):

        userIdI:int
        userHistDFI:dict
        for userIdI, userHistDFI in self._historyDict.items():
            self._historyDict[userIdI] = userHistDFI.tail(numberOfUserRecommendationToKeep)


    def deletePreviousRecomOfUser(self, userID:int, numberOfUserRecommendationToKeep:int):
        if not userID in self._historyDict.keys():
            return
        self._historyDict[userID] = self._historyDict[userID].tail(numberOfUserRecommendationToKeep)


    def print(self):
        print("Number of users: " + str(len(self._historyDict.keys())))

        counter:int = 0
        userIdI:int
        userHistDFI:dict
        for userIdI, userHistDFI in self._historyDict.items():
            counter += len(userHistDFI.keys())

        print("Number of rows: " + str(counter))