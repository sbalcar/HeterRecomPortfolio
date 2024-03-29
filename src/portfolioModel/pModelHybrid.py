#!/usr/bin/python3

import os
from typing import List

import pandas as pd
from pandas import DataFrame

from evaluationTool.evalToolDHondt import EvalToolDHondt

from portfolioModel.pModelDHondt import PModelDHondt #class
from portfolioModel.pModelDHondtPersonalised import PModelDHondtPersonalised #class
from portfolioModel.pModelDHondtPersonalisedStat import PModelDHondtPersonalisedStat #class
from batchDefinition.inputRecomRRDefinition import InputRecomRRDefinition #class
from simulation.aSequentialSimulation import ASequentialSimulation #class


class PModelHybrid(DataFrame):

    ARG_MODE_SKIP:str = "skip"
    ARG_SKIP_CLICK_THRESHOLD:str = "threshold"

    COL_MODELID:str = "modelId"
    COL_MODEL:str = "model"

    ROW_MODEL_GLOBAL:str = "global"
    ROW_MODEL_PERSON:str = "person"

    def __init__(self, mGlobal:DataFrame, mPerson:DataFrame, argsDict:dict[str]={}):
        if not isinstance(mGlobal, DataFrame):
            raise ValueError("Argument mGlobal isn't type DataFrame.")
        if not isinstance(mPerson, DataFrame):
            raise ValueError("Argument mPerson isn't type DataFrame.")

        modelHybridData = [[self.ROW_MODEL_GLOBAL, mGlobal],[self.ROW_MODEL_PERSON, mPerson]]
        super(PModelHybrid, self).__init__(modelHybridData, columns=[self.COL_MODELID, self.COL_MODEL])
        self.set_index(self.COL_MODELID, inplace=True)

        self.modeSkip:bool = argsDict.get(self.ARG_MODE_SKIP, False)
        if self.modeSkip:
            self.skipClickThreshold:int = argsDict[self.ARG_SKIP_CLICK_THRESHOLD]

    def getModelGlobal(self):
        return self.loc[self.ROW_MODEL_GLOBAL][self.COL_MODEL]

    def getModelPersonAllUsers(self):
        mP:DataFrame = self.loc[self.ROW_MODEL_PERSON][self.COL_MODEL]
        return mP

    def getModelPerson(self, userID:int):
        mP:DataFrame = self.loc[self.ROW_MODEL_PERSON][self.COL_MODEL]
        print(type(mP))
        #mP.__class__ = PModelDHondtPersonalisedStat
        return mP.getModel(userID)


    def getModel(self, userID:int, argsDict:dict):

        status:float = argsDict[ASequentialSimulation.ARG_STATUS]
        print("status: " + str(status))

        mGlobal:DataFrame = self.getModelGlobal()

        if self.modeSkip:
            print("aaaaaaaaaaaaaaaaa")
            numberOfClick:int = self.getModelPersonAllUsers().getNumberOfClick(userID)
            print("numberOfClick: " + str(numberOfClick))
            if numberOfClick < self.skipClickThreshold:
                return mGlobal

        mGlobal.linearNormalizing()
        #print("GLOBAL:")
        #print(mGlobal.head(10))
        mGlobal = PModelDHondt.multiplyModel(mGlobal, 1.0 - status)

        mPerson:DataFrame = self.getModelPerson(userID)
        mPerson.linearNormalizing()
        mPerson = PModelDHondt.multiplyModel(mPerson, status)

        rPModel:DataFrame = PModelDHondt.sumModels(mGlobal, mPerson)
        rPModel.linearNormalizing()

        #print("VYSLEDNY:")
        #print(rPModel.head(10))

        return rPModel

    def countResponsibility(self, userID:int, aggregatedItemIDs:List[int], methodsResultDict:dict,
                            numberOfItems:int = 20, argsDict:dict={}):

        #print("userID: " + str(userID))
        model:DataFrame = self.getModel(userID, argsDict)
        #print("model: " + str(model.head(10)))

        return model.countResponsibility(userID, aggregatedItemIDs, methodsResultDict, numberOfItems, None)

    def incrementClick(self, userID):
        model:DataFrame = self.getModelPerson(userID)
        model.incrementClick(userID)


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


    @classmethod
    def readModel(cls, fileName:str, counterI:int):

        f = open(fileName, "r")
        lines:List[str] = f.readlines()
        #print(lines)

        model = None
        for rowIndexI in range(0, len(lines)):
            if lines[rowIndexI].startswith(str(counterI) + " / "):
                modelStr:str = lines[rowIndexI + 3]
                model:DataFrame = pd.read_json(modelStr)
                model.__class__ = PModelHybrid
                print(modelStr)
                break

        if model is None:
            print("Return None")
            return None

        for indexI in model.index:
            model.loc[indexI][PModelHybrid.COL_MODEL] = DataFrame(model.loc[indexI][PModelHybrid.COL_MODEL])
        return model


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")

    #fileName:str = "results/ml1mDiv90Ustatic08R1/portfModelTimeEvolution-HybridFDHondtSkipFixedClk003ViewDivisor250NRTrueClk003ViewDivisor250NRTrue.txt"
    #fileName:str = "results/rrDiv90Ustatic08R1/portfModelTimeEvolution-HybridFDHondtSkipFixedClk003ViewDivisor250NRTrueClk003ViewDivisor250NRTrue.txt"
    fileName:str = "results/stDiv90Ustatic08R1/portfModelTimeEvolution-HybridStatFDHondtFixedClk003ViewDivisor250NRFalseClk003ViewDivisor250NRFalse.txt"

    #model = PModelHybrid.readModel(fileName, 50000)
    #model = PModelHybrid.readModel(fileName, 15000)
    model = PModelHybrid.readModel(fileName, 150)
    #print(type(model))

    userIds = list(set(model.getModelPersonAllUsers().index.tolist()))
    print("Users:")
    print(userIds)

    print("Global model:")
    print(model.getModelGlobal())

    userIdI = 3627575.0
    r:List[str] = []
    for userIdI in userIds:
        model2Str = model.getModelPersonAllUsers().loc[str(userIdI), "model"]
        #print(type(model2Str))
        #print(model2Str)

        modelPersI:DataFrame = DataFrame(model2Str)
        #print(modelPersI)

        methodIdI:str = str(modelPersI['votes'].idxmax())
        #print("")
        #print(methodIdI)
        #print(model.getModelPerson(3565819.0))
        r.append(methodIdI)

    print("")

    print("RecomBprmf")
    print(r.count("RecomBprmf"))

    print("RecomCosinecb")
    print(r.count("RecomCosinecb"))

    print("RecomKnn")
    print(r.count("RecomKnn"))

    print("RecomThemostpopular")
    print(r.count("RecomThemostpopular"))

    print("RecomVmcontextknn")
    print(r.count("RecomVmcontextknn"))

    print("RecomW2V")
    print(r.count("RecomW2V"))
