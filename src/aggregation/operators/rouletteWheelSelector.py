#!/usr/bin/python3
import random

import numpy as np
import pandas as pd

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from abc import ABC, abstractmethod

class RouletteWheelSelector(ABC):

    # resultOfMethod:pd.Series([raitings],[itemIDs])
    def run(resultOfMethod:Series):
        # weighted random choice
        pick:float = random.uniform(0, sum(resultOfMethod.values))
        current:float = 0
        for itemIDI in resultOfMethod.index:
            current += resultOfMethod[itemIDI]
            if current > pick:
              return itemIDI
