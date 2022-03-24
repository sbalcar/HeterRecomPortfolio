#!/usr/bin/python3

import os
from typing import List

import pandas as pd
from pandas import DataFrame

from evaluationTool.evalToolDHondt import EvalToolDHondt

from portfolioModel.pModelDHondt import PModelDHondt #class

from batchDefinition.inputRecomRRDefinition import InputRecomRRDefinition #class


class PModelDHondtPersonalisedStat(pd.DataFrame):

    COL_USER_ID = "userID"
    COL_MODEL = "model"
    COL_CLICK_COUNT = "clickCount"

    def __init__(self, recommendersIDs:List[str]):

        pM1:DataFrame = PModelDHondt(recommendersIDs)
        #print(pM1.head(10))

        modelDHontData = [[float('nan'), 0, pM1]]
        super(PModelDHondtPersonalisedStat, self).__init__(modelDHontData, columns=[PModelDHondtPersonalisedStat.COL_USER_ID,
                        PModelDHondtPersonalisedStat.COL_CLICK_COUNT, PModelDHondtPersonalisedStat.COL_MODEL])
        self.set_index(PModelDHondtPersonalisedStat.COL_USER_ID, inplace=True)

    def getModel(self, userID:int, argsDict:dict={}):
        if not userID in self.index:
            userIdWithMaxClicks:int = self[PModelDHondtPersonalisedStat.COL_CLICK_COUNT].idxmax()
            print("aaaaaaaaaaaaaa: " + str(userIdWithMaxClicks))
            modelOfUserID:DataFrame = (self.loc[userIdWithMaxClicks, PModelDHondtPersonalisedStat.COL_MODEL]).copy()
            modelOfUserID.__class__ = PModelDHondt
            self.loc[userID] = [0, modelOfUserID]

        return self.loc[userID][PModelDHondtPersonalisedStat.COL_MODEL]

    def countResponsibility(self, userID:int, aggregatedItemIDs:List[int], methodsResultDict:dict, numberOfItems:int = 20, votes = None):

        #print("userID: " + str(userID))
        model:DataFrame = self.getModel(userID)
        #print("model: " + str(model.head(10)))

        return model.countResponsibility(userID, aggregatedItemIDs, methodsResultDict, numberOfItems, votes)


    def getValue(self, userID:int, methodID:str):
        if not userID in self.index:
            return 0

        model:DataFrame = self.loc[userID][PModelDHondtPersonalisedStat.COL_MODEL]

        return model.loc[methodID][PModelDHondt.COL_VOTES]


    def setValue(self, userID:int, methodID:str, value):
        if not userID in self.index:
            self.loc[userID] = self.loc[None].copy()

        model:DataFrame = self.loc[userID][PModelDHondtPersonalisedStat.COL_MODEL]

        model.loc[methodID][PModelDHondt.COL_VOTES] = value


    def incrementClick(self, userID:int):
        self.at[userID, PModelDHondtPersonalisedStat.COL_CLICK_COUNT] =\
                self.loc[userID, PModelDHondtPersonalisedStat.COL_CLICK_COUNT] + 1

    def getNumberOfClick(self, userID:int):
        if not userID in self.index:
            return 0
        return self.at[userID, PModelDHondtPersonalisedStat.COL_CLICK_COUNT]

    @classmethod
    def readModel(cls, fileName:str, counterI:int):

        f = open(fileName, "r")
        lines:List[str] = f.readlines()

        model:DataFrame = None
        for rowIndexI in range(0, len(lines)):
            if lines[rowIndexI].startswith(str(counterI) + " / "):
                modelStr:str = lines[rowIndexI+3]
                model:DataFrame = pd.read_json(modelStr)
                model.__class__ = PModelDHondtPersonalisedStat
                break

        if model is None:
            return None

        for indexI in model.index:
            model.loc[indexI][PModelDHondtPersonalisedStat.COL_MODEL] = DataFrame(model.loc[indexI][PModelDHondtPersonalisedStat.COL_MODEL])
        return model



if __name__ == "__main__":

    os.chdir("..")
    os.chdir("..")

    print(os.getcwd())

    rIDs, rDscrs = InputRecomRRDefinition.exportPairOfRecomIdsAndRecomDescrs()
    #print(rIDs)
    model:DataFrame = PModelDHondtPersonalisedStat(rIDs)

    userID1:int = 10

    modelInter = model.getModel(userID1)
    print("Clicks: " + str(model.loc[userID1][PModelDHondtPersonalisedStat.COL_CLICK_COUNT]))

    model.incrementClick(userID1)
    model.incrementClick(userID1)

    print("Clicks: " + str(model.loc[userID1][PModelDHondtPersonalisedStat.COL_CLICK_COUNT]))

    userID2:int = 100

    print(model.getModel(userID2))
    model.incrementClick(userID2)
    model.incrementClick(userID2)
    model.incrementClick(userID2)

    print(model.getModel(200))

    print(modelInter)

