#!/usr/bin/python3

from typing import List
from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class

class AHistory:

    def __init__(self):
        raise Exception("AHistory is abstract class, can't be instanced")

    def addRecommendation(self, itemID:int, recommendedItemIDs:List[int]):
        assert False, "this needs to be overridden"

    def getValue(self, itemID:int, uBehaviourDesc:UserBehaviourDescription):
        assert False, "this needs to be overridden"

    def print(selfs):
        pass
