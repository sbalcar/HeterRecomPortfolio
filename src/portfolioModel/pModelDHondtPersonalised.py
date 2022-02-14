#!/usr/bin/python3

import os
from typing import List

import pandas as pd
from pandas import DataFrame

from evaluationTool.evalToolDHondt import EvalToolDHondt

from portfolioModel.pModelDHondt import PModelDHondt #class


class PModelDHondtPersonalised(pd.DataFrame):

    COL_USER_ID = "userID"
    COL_MODEL = "model"

    def __init__(self, recommendersIDs:List[str]):

        pM1:DataFrame = PModelDHondt(recommendersIDs)
        #print(pM1.head(10))

        modelDHontData = [[float('nan'), pM1]]
        super(PModelDHondtPersonalised, self).__init__(modelDHontData, columns=[PModelDHondtPersonalised.COL_USER_ID, PModelDHondtPersonalised.COL_MODEL])
        self.set_index(PModelDHondtPersonalised.COL_USER_ID, inplace=True)

    def getModel(self, userID:int):
        if not userID in self.index:
            print("index: " + str(self.index))
            print("self.loc[None]: " + str(self.loc[float('nan')]))
            modelOfUserID:DataFrame = (self.loc[float('nan'), PModelDHondtPersonalised.COL_MODEL]).copy()
            modelOfUserID.__class__ = PModelDHondt
            self.loc[userID] = [modelOfUserID]

        return self.loc[userID][PModelDHondtPersonalised.COL_MODEL]

    def countResponsibility(self, userID:int, aggregatedItemIDs:List[int], methodsResultDict:dict, numberOfItems:int = 20, votes = None):

        #print("userID: " + str(userID))
        model:DataFrame = self.getModel(userID)
        #print("model: " + str(model.head(10)))

        return model.countResponsibility(userID, aggregatedItemIDs, methodsResultDict, numberOfItems, votes)


    def getValue(self, userID:int, methodID:str):
        if not userID in self.index:
            return 0

        model:DataFrame = self.loc[userID][PModelDHondtPersonalised.COL_MODEL]

        return model.loc[methodID][PModelDHondt.COL_VOTES]


    def setValue(self, userID:int, methodID:str, value):
        if not userID in self.index:
            self.loc[userID] = self.loc[None].copy()

        model:DataFrame = self.loc[userID][PModelDHondtPersonalised.COL_MODEL]

        model.loc[methodID][PModelDHondt.COL_VOTES] = value



if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")

#    m = PModelDHondtPersonalised(["metoda1", "metoda2", "metoda3"])
    #print(m.loc[None][PModelDHontPersonalised.COL_MODEL].head(10))

    userID:int = 5

#    print(m.getValue(userID, "metoda1"))
#    m.setValue(userID, "metoda1", 4)
#    print(m.getValue(userID, "metoda1"))

#    a = pd.read_json("result/a.txt")
    patients_df = pd.read_json("results/a.txt")
    #patients_df["index"] = pd.to_numeric(patients_df["index"])
    patients_df.index = patients_df.index.astype(int)

    #print(patients_df.loc["nan"].head())
    print(patients_df.head(200).to_string())
