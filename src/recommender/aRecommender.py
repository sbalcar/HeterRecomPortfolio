#!/usr/bin/python3

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class


from history.aHistory import AHistory #class


class ARecommender:

    def __init__(self, argumentsDict:dict):
        raise Exception("ARecommender is abstract class, can't be instanced")

    def train(self, ratingsTrainDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
        assert False, "this needs to be overridden"

    def update(self, ratingsUpdateDF:DataFrame):
        assert False, "this needs to be overridden"

    def recommendToItem(self, itemID:int, ratingsTestDF:DataFrame, history:AHistory, numberOfItems:int):
        assert False, "this needs to be overridden"

