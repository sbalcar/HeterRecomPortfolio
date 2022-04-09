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

    def getModel(self, userID:int, argsDict:dict={}):
        if not userID in self.index:
            #print("index: " + str(self.index))
            #print(self)
            #print("self.loc[None]: " + str(self.loc[float('nan')]))
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


    @classmethod
    def readModel(cls, fileName:str, counterI:int):

        f = open(fileName, "r")
        lines:List[str] = f.readlines()
        #print(lines)

        modelDF:DataFrame = None
        for rowIndexI in range(0, len(lines)):
            if lines[rowIndexI].startswith(str(counterI) + " / "):
                modelStr:str = lines[rowIndexI+3]
                modelDF:DataFrame = pd.read_json(modelStr, convert_axes=False ,convert_dates=False)
                modelDF.__class__ = PModelDHondtPersonalised
                break

        if modelDF is None:
            print("Return None")
            return None

        for indexI in modelDF.index:
            modelI:dict = modelDF.loc[indexI][PModelDHondtPersonalised.COL_MODEL]
            modelIDF:DataFrame = DataFrame(modelI)
            modelDF.loc[indexI][PModelDHondtPersonalised.COL_MODEL] = modelIDF

        modelDF.index = modelDF.index.astype(float)
        return modelDF




if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")

    fileName:str = "results/rrDiv90Ulinear0109R1/portfModelTimeEvolution-PersonalFDHondtFixedClk02ViewDivisor1000.txt"
    fileName:str = "results/stDiv90Ustatic08R1/portfModelTimeEvolution-HybridStatFDHondtFixedClk003ViewDivisor250NRFalseClk003ViewDivisor250NRFalse.txt"

    modelDF:DataFrame = PModelDHondtPersonalised.readModel(fileName, 19200)
    print(type(modelDF))

    vBprmf = []
    vCosinecb = []
    vKnn = []
    vThemostpopular = []
    vVmcontextknn = []
    vW2V = []

    userID:int
    userIDs:List[int] = [userIDI for userIDI in modelDF.index]
    for userIDI in userIDs:
        modelPersI:DataFrame = modelDF.loc[userIDI][PModelDHondtPersonalised.COL_MODEL]
        #print(modelPersI)

        nOfVRecomBprmfI:float = float(modelPersI.loc["RecomBprmf"])
        nOfVRecomCosinecbI:float = float(modelPersI.loc["RecomCosinecb"])
        nOfVRecomKnnI:float = float(modelPersI.loc["RecomKnn"])
        nOfVThemostpopularI:float = float(modelPersI.loc["RecomThemostpopular"])
        nVmcontextknnI:float = float(modelPersI.loc["RecomVmcontextknn"])
        nOfVW2VI:float = float(modelPersI.loc["RecomW2V"])

        vBprmf.append(nOfVRecomBprmfI)
        vCosinecb.append(nOfVRecomCosinecbI)
        vKnn.append(nOfVRecomKnnI)
        vThemostpopular.append(nOfVThemostpopularI)
        vVmcontextknn.append(nVmcontextknnI)
        vW2V.append(nOfVW2VI)

    import matplotlib.pyplot as plt
    import numpy as np

    plt.hist(vBprmf, 10, None, fc='none', lw=1.5, histtype='step')
    plt.hist(vCosinecb, 10, None, fc='none', lw=1.5, histtype='step')
    plt.hist(vKnn, 10, None, fc='none', lw=1.5, histtype='step')
    plt.hist(vThemostpopular, 10, None, fc='none', lw=1.5, histtype='step')
    plt.hist(vVmcontextknn, 10, None, fc='none', lw=1.5, histtype='step')
    plt.hist(vW2V, 10, None, fc='none', lw=1.5, histtype='step')
    plt.show()



    # Implementation of matplotlib function
    import matplotlib
    import numpy as np
    import matplotlib.pyplot as plt

    n_bins = 40

    x = [vBprmf, vCosinecb, vKnn, vThemostpopular, vVmcontextknn, vW2V]
    print(x)

    colors = ['green', 'blue', 'lime', 'red', 'black', 'yellow']
    legends = ["Bprmf", "Cosinecb", "Knn", "Themostpopular", "Vmcontextknn", "W2V"]

    plt.hist(x, n_bins,
             histtype='bar',
             color=colors,
             label=legends)

    plt.legend(prop={'size': 10})

    plt.title('Histogram of votes for all users\n\n',
              fontweight="bold")

    plt.show()