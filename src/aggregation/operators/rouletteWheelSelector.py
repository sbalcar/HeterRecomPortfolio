#!/usr/bin/python3
import random

import numpy as np
import pandas as pd

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from abc import ABC, abstractmethod

from aggregation.operators.aDHondtSelector import ADHondtSelector #class


class RouletteWheelSelector(ADHondtSelector):

    ARG_EXPONENT:str = "exponent"

    def __init__(self, argumentsDict:dict):
        self._exponent:int = argumentsDict[RouletteWheelSelector.ARG_EXPONENT]
        if self._exponent is None or self._exponent < 0:
            raise ValueError("Argument userID isn't in range (0, infinity).")

    # resultOfMethod:pd.Series([raitings],[itemIDs])
    def select(self, votesOfCandidatesDict:Series):

        votesOfCandidatesSer:Series = None
        if self._exponent == 1:
            votesOfCandidatesSer:Series = Series(votesOfCandidatesDict, index=votesOfCandidatesDict.keys())
        elif self._exponent > 1:
            vcDict:dict = dict(map(lambda mIdJ: (mIdJ, votesOfCandidatesDict[mIdJ] ** self._exponent), votesOfCandidatesDict.keys()))

            votesOfCandidatesSer:Series = Series(vcDict, index=vcDict.keys())
        else:
            raise ValueError("Exception - State is invalid")

        # weighted random choice
        pick:float = random.uniform(0, sum(votesOfCandidatesSer.values))
        current:float = 0
        for itemIDI in votesOfCandidatesSer.index:
            current += votesOfCandidatesSer[itemIDI]
            if current > pick:
              return itemIDI
