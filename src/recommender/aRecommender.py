#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

from history.aHistory import AHistory #class


class ARecommender:

    def __init__(self, argumentsDict:dict):
        raise Exception("ARecommender is abstract class, can't be instanced")

    def train(self, ratingsTrainDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
        pass

    def update(self, ratingsUpdateDF:DataFrame):
        pass

    def recommendToItem(self, itemID:int, ratingsTestDF:DataFrame, history:AHistory, numberOfItems:int):
        pass
