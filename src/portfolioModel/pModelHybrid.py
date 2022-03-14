#!/usr/bin/python3

import os
from typing import List

import pandas as pd
from pandas import DataFrame

from evaluationTool.evalToolDHondt import EvalToolDHondt

from portfolioModel.pModelDHondt import PModelDHondt #class
from portfolioModel.pModelDHondtPersonalisedStat import PModelDHondtPersonalisedStat #class
from batchDefinition.inputRecomRRDefinition import InputRecomRRDefinition #class


class PModelHybrid(DataFrame):

    COL_MODELID:str = "modelId"
    COL_MODEL:str = "model"

    ROW_MODEL_GLOBAL:str = "global"
    ROW_MODEL_PERSON:str = "person"

    def __init__(self, mGlobal:DataFrame, mPerson:DataFrame):
        if not isinstance(mGlobal, DataFrame):
            raise ValueError("Argument mGlobal isn't type DataFrame.")
        if not isinstance(mPerson, DataFrame):
            raise ValueError("Argument mPerson isn't type DataFrame.")

        modelDHontData = [[self.ROW_MODEL_GLOBAL, mGlobal],[self.ROW_MODEL_PERSON, mPerson]]
        super(PModelHybrid, self).__init__(modelDHontData, columns=[self.COL_MODELID, self.COL_MODEL])
        self.set_index(self.COL_MODELID, inplace=True)


    def getModelGlobal(self):
        return self.loc[self.ROW_MODEL_GLOBAL][self.COL_MODEL]

    def getModelPerson(self, userID:int):
        mP:DataFrame = self.loc[self.ROW_MODEL_PERSON][self.COL_MODEL]
        return mP.getModel(userID)


    def getModel(self, userID:int):

        mGlobal:DataFrame = self.getModelGlobal()
        mPerson:DataFrame = self.getModelPerson(userID)

        rPModel:DataFrame = PModelDHondt.sumModels(mGlobal, mPerson)
        rPModel.linearNormalizing()

        return rPModel

    def countResponsibility(self, userID:int, aggregatedItemIDs:List[int], methodsResultDict:dict, numberOfItems:int = 20, votes = None):

        #print("userID: " + str(userID))
        model:DataFrame = self.getModel(userID)
        #print("model: " + str(model.head(10)))

        return model.countResponsibility(userID, aggregatedItemIDs, methodsResultDict, numberOfItems, votes)


    def incrementClick(self, userID):
        self.at[userID, PModelDHondtPersonalisedStat.COL_CLICK_COUNT] =\
                self.loc[userID, PModelDHondtPersonalisedStat.COL_CLICK_COUNT] + 1


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


