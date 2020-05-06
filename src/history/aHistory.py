#!/usr/bin/python3

from typing import List


class AHistory:

    def __init__(self):
        raise Exception("AHistory is abstract class, can't be instanced")

    def addRecommendation(self, itemID:int, recommendedItemIDs:List[int]):
        pass
