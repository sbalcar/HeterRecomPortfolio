#!/usr/bin/python3

import datetime

from typing import List
from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from abc import ABC, abstractmethod


class AHistory(ABC):

    @abstractmethod
    def __init__(self):
        raise Exception("AHistory is abstract class, can't be instanced")

    @abstractmethod
    def insertRecommendation(self, userID:int, rItemID:int, position:int, uObservation:float, clicked:bool, timestamp=datetime.datetime.now()):
        assert False, "this needs to be overridden"

    def insertRecomAndClickedItemID(self, userID:int, recommendedItemIDs:List[int], uObservation:List[float], clickedItemID:int):

        position:List[int] = range(0, len(recommendedItemIDs))

        for rItemIdI, positionI, uObservationI in zip(recommendedItemIDs, position, uObservation):

            clickedI:bool = rItemIdI == clickedItemID
            self.insertRecommendation(userID, rItemIdI, positionI, uObservationI, clickedI)

    def insertRecomAndClickedItemIDs(self, userID:int, recommendedItemIDs:List[int], uObservation:List[float], clickedItemIDs:List[int]):

        position:List[int] = range(0, len(recommendedItemIDs))

        for rItemIdI, positionI, uObservationI in zip(recommendedItemIDs, position, uObservation):

            clickedI:bool = rItemIdI in clickedItemIDs
            self.insertRecommendation(userID, rItemIdI, positionI, uObservationI, clickedI)


    @abstractmethod
    def getPreviousRecomOfUser(self, userID:int, limit:int=100):
        assert False, "this needs to be overridden"

    @abstractmethod
    def getPreviousRecomOfUserAndItem(self, userID:int, itemID:int, limit:int=100):
        assert False, "this needs to be overridden"

    def getIgnoringValue(self, userID:int, itemID:int, limit:int=20):

        uRows:List[tuple] = self.getPreviousRecomOfUserAndItem(userID, itemID, limit=limit)

        def valueOfIgnoring(rowI:tuple):
            itemIdI:int = rowI[2]
            probOfObservI:float = rowI[4]
            if not itemIdI == itemID:
                return 0
            return probOfObservI

        return sum(map(valueOfIgnoring, uRows))

    @abstractmethod
    def isObjectClicked(self, userID:int, itemID:int, limit:int=100):
        assert False, "this needs to be overridden"

    @abstractmethod
    def delete(self, numberOfUserRecommendationToKeep:int):
        assert False, "this needs to be overridden"

    @abstractmethod
    def deletePreviousRecomOfUser(self, userID:int, numberOfUserRecommendationToKeep:int):
        assert False, "this needs to be overridden"

    @abstractmethod
    def print(selfs):
        pass
