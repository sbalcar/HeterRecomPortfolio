#!/usr/bin/python3

from typing import List
from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from abc import ABC, abstractmethod


class AHistory(ABC):

    @abstractmethod
    def __init__(self):
        raise Exception("AHistory is abstract class, can't be instanced")

    @abstractmethod
    def addRecommendation(self, itemID:int, recommendedItemIDs:List[int]):
        assert False, "this needs to be overridden"

    @abstractmethod
    def getIgnoringValue(self, itemID:int, uBehaviourDesc:UserBehaviourDescription):
        assert False, "this needs to be overridden"

    @abstractmethod
    def print(selfs):
        pass
