#!/usr/bin/python3

from typing import List

import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from pandas.core.frame import DataFrame #class

from history.aHistory import AHistory #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class


class ASequentialSimulation(ABC):

    @abstractmethod
    def __init__(self, batchID:str, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame,
                 behaviourDF:DataFrame, argumentsDict:dict):
        raise Exception("ASequentialSimulation is abstract class, can't be instanced")

    @abstractmethod
    def run(self, portfolioDescs:List[APortfolioDescription], portFolioModels:List[pd.DataFrame],
            evaluatonTools:List, histories:List[AHistory]):
        assert False, "this needs to be overridden"

