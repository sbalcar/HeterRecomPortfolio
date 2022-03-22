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


    def getModel(self, userID:int, argsDict:dict={}):
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

    @staticmethod
    def getEqualResponsibilityForAll(itemIDs:List[int], methodIDs:List[str]):

        result:List[tuple] = []
        for itemIDI in itemIDs:
            respI:dict = dict([(methodIDI,1.0) for methodIDI in methodIDs])
            result.append((itemIDI, respI))

        return result

    def getMethodIDs(self):
        return self.index

    @classmethod
    def readModel(cls, fileName:str, counterI:int):

        f = open(fileName, "r")
        lines:List[str] = f.readlines()

        for rowIndexI in range(0, len(lines)):
            if lines[rowIndexI].startswith(str(counterI) + " / "):
                modelStr:str = lines[rowIndexI+3]
                model:DataFrame = pd.read_json(modelStr)
                model.__class__ = PModelDHondt
                return model

        return None

    @classmethod
    def sumModels(cls, pModel1, pModel2):
        print("")
        methodIdsThis:List[str] = list(pModel1.getMethodIDs())
        methodIdsThis.sort()
        methodIdsExtr:List[str] = list(pModel2.getMethodIDs())
        methodIdsExtr.sort()

        if not methodIdsThis == methodIdsExtr:
            print("chyba")

        valuesNew:List[float] = pModel1.loc[methodIdsThis, cls.COL_VOTES] + pModel2.loc[methodIdsThis, cls.COL_VOTES]
        data:List[tuple] = list(zip(methodIdsThis, valuesNew))

        df = pd.DataFrame(data, columns = [cls.COL_METHOD_ID, cls.COL_VOTES])
        df.set_index(PModelDHondt.COL_METHOD_ID, inplace=True)
        df.__class__ = PModelDHondt

        return df

    @classmethod
    def multiplyModel(cls, pModel1, value:float):
        valuesNew: List[float] = pModel1[cls.COL_VOTES] * value
        methodIdsThis:List[str] = pModel1.getMethodIDs()
        data:List[tuple] = list(zip(methodIdsThis, valuesNew))

        df = pd.DataFrame(data, columns = [cls.COL_METHOD_ID, cls.COL_VOTES])
        df.set_index(PModelDHondt.COL_METHOD_ID, inplace=True)
        df.__class__ = PModelDHondt

        return df

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats



if __name__ == "__main__":

    os.chdir("..")
    os.chdir("..")

    print(os.getcwd())

    fileName:str = "results/rrDiv90Ulinear0109R1/portfModelTimeEvolution-FDHondtFixedClk003ViewDivisor250.txt"
    modelDF:DataFrame = PModelDHondt.readModel(fileName, 3989)


    xLen = 15000
    yLen = 6

    xNew = range(xLen)
    yNew = [[-10] * xLen for _ in range(yLen)]

    for xI in range(xLen):
        modelDF:DataFrame = PModelDHondt.readModel(fileName, xI+1)
        for yJ in range(yLen):
            valueIJ:float = float(modelDF.iloc[yJ][PModelDHondt.COL_VOTES])
            #print(str(valueIJ))
            yNew[yJ][xI] = valueIJ

    print(yNew)
    x = np.arange(1990, 2020)  # (N,) array-like
    y = [np.random.randint(0, 5, size=30) for _ in range(5)]  # (M, N) array-like


    fig, ax = plt.subplots(figsize=(15, 7))
    ax.stackplot(xNew, yNew);
    plt.legend(modelDF.getMethodIDs())
    plt.show()