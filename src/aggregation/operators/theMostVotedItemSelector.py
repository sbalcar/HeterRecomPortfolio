#!/usr/bin/python3
import random

import numpy as np
import pandas as pd

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from abc import ABC, abstractmethod

from aggregation.operators.aDHondtSelector import ADHondtSelector #class


class TheMostVotedItemSelector(ADHondtSelector):

    def __init__(self, argumentsDict:dict):
        {}

    # resultOfMethod:pd.Series([raitings],[itemIDs])
    def select(self, votesOfCandidatesDict:dict):

        # get the highest number of votes of remaining candidates
        maxVotes:float = max(votesOfCandidatesDict.values())
        # print("MaxVotes: ", maxVotes)

        # select candidate with highest number of votes
        selectedCandidateI:int = [votesOfCandI for votesOfCandI in votesOfCandidatesDict.keys() if
                                  votesOfCandidatesDict[votesOfCandI] == maxVotes][0]
        # print("SelectedCandidateI: ", selectedCandidateI)
        return selectedCandidateI