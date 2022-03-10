#!/usr/bin/python3

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.aDataset import ADataset #class

from history.aHistory import AHistory #class
from abc import ABC, abstractmethod

from typing import Dict

class ARecommender(ABC):

    ARG_ALLOWED_ITEMIDS:str = "allowedItemIDs"

    @abstractmethod
    def __init__(self, batchID:str, argumentsDict:Dict[str,object]):
        raise Exception("ARecommender is abstract class, can't be instanced")

    @abstractmethod
    def train(self, history:AHistory, dataset:ADataset):
        assert False, "this needs to be overridden"

    @abstractmethod
    def update(self, ratingsUpdateDF:DataFrame, argumentsDict:Dict[str,object]):
        assert False, "this needs to be overridden"

    @abstractmethod
    def recommend(self, userID:int, numberOfItems:int, argumentsDict:Dict[str,object]):
        assert False, "this needs to be overridden"

