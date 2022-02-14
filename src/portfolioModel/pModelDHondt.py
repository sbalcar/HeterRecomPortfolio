#!/usr/bin/python3

import os
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame

from evaluationTool.evalToolDHondt import EvalToolDHondt


class PModelDHondt(pd.DataFrame):

    COL_METHOD_ID:str = "methodID"
    COL_VOTES:str = "votes"

    def __init__(self, recommendersIDs:List[str]):
        modelDHontData:List[List] = [[rIdI, 1] for rIdI in recommendersIDs]

        super(PModelDHondt, self).__init__(modelDHontData, columns=[PModelDHondt.COL_METHOD_ID, PModelDHondt.COL_VOTES])
        self.set_index(PModelDHondt.COL_METHOD_ID, inplace=True)

        self.linearNormalizing()


    def linearNormalizing(self):
        PModelDHondt.linearNormalizingPortfolioModelDHondt(self)

    @staticmethod
    def linearNormalizingPortfolioModelDHondt(portfolioModelDHondt:DataFrame):
        # linearly normalizing to unit sum of votes
        sumMethodsVotes:float = portfolioModelDHondt[PModelDHondt.COL_VOTES].sum()
        for methodIdI in portfolioModelDHondt.index:
            portfolioModelDHondt.loc[methodIdI, PModelDHondt.COL_VOTES] = portfolioModelDHondt.loc[methodIdI, PModelDHondt.COL_VOTES] / sumMethodsVotes


    def getModel(self, userID:int):
        return self

    def countResponsibility(self, userID:int, aggregatedItemIDs:List[int], methodsResultDict:dict, numberOfItems:int = 20, votes = None):

        # sumOfAllVotes:int = sum(methodsParamsDF["votes"].values)
        if votes is None:
            votesOfPartiesDictI: dict[str, int] = {mI: self.votes.loc[mI] for mI in self.index}
        else:
        # do not include votes weighting in the case of Thompsons Sampling
        # votesOfPartiesDictI:dict[str,int] = votes
            votesOfPartiesDictI: dict[str, int] = {mI: 1.0 for mI in self.index}

        candidatesOfMethods: np.asarray[str] = np.asarray([cI.keys() for cI in methodsResultDict.values()], dtype=object)
        # print("candidatesOfMethods: " + str(candidatesOfMethods))
        uniqueCandidatesI: List[str] = list(set(np.concatenate(candidatesOfMethods)))

        candidateOfdevotionOfPartiesDictDict: dict = {}

        candidateIDI: int
        for candidateIDI in aggregatedItemIDs:
        # for candidateIDI in uniqueCandidatesI:
            devotionOfParitiesDict: dict = {}

            parityIDJ:str
            for parityIDJ in self.index:
                devotionOfParitiesDict[parityIDJ] = methodsResultDict[parityIDJ].get(candidateIDI, 0) * votesOfPartiesDictI[parityIDJ]
            candidateOfdevotionOfPartiesDictDict[candidateIDI] = devotionOfParitiesDict
            # print(candidateOfdevotionOfPartiesDictDict)

        # selectedCandidate:list<(itemID:int, Series<(rating:int, methodID:str)>)>
        selectedCandidate: List[int, pd.Series[str, int]] = [
            (candidateI, candidateOfdevotionOfPartiesDictDict[candidateI]) for candidateI in aggregatedItemIDs]

        # list<(itemID:int, Series<(rating:int, methodID:str)>)>
        return selectedCandidate[:numberOfItems]
