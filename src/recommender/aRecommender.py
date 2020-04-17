#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

class ARecommender:

    def train(self, historyDF:DataFrame, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
        pass

    def recommendToItem(self, itemID:int, numberOfItems:int):
        pass
