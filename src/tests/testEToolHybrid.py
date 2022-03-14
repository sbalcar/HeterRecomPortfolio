#!/usr/bin/python3

import os
from typing import List
from typing import Dict

from pandas.core.frame import DataFrame #class

import pandas as pd

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class
from evaluationTool.evalToolDHondtPersonal import EvalToolDHondtPersonal #class
from evaluationTool.evalToolHybrid import EToolHybrid #class

from batchDefinition.inputRecomSTDefinition import InputRecomSTDefinition #class

from portfolioModel.pModelDHondt import PModelDHondt #class
from portfolioModel.pModelDHondtPersonalised import PModelDHondtPersonalised #class
from portfolioModel.pModelHybrid import PModelHybrid #class


def test01():
    print("Test 01")

    userID:int = 1
    clickedItemID:int = 101

    # methods parametes
    portfolioModelData1:List[tuple] = [['metoda1',100], ['metoda2',80], ['metoda3',60]]
    portfolioModel1:DataFrame = pd.DataFrame(portfolioModelData1, columns=["methodID","votes"])
    portfolioModel1.set_index("methodID", inplace=True)
    portfolioModel1.__class__ = PModelDHondt
    portfolioModel1.linearNormalizing()

    portfolioModelData2:List[tuple] = [['metoda1',0], ['metoda2',20], ['metoda3',40]]
    portfolioModel2:DataFrame = pd.DataFrame(portfolioModelData2, columns=["methodID","votes"])
    portfolioModel2.set_index("methodID", inplace=True)
    portfolioModel2.__class__ = PModelDHondt
    portfolioModel2.linearNormalizing()

    pModel:DataFrame = PModelHybrid(portfolioModel1, portfolioModel2)
    #print(pModel.getModelGlobal())
    #print(pModel.getModelPerson(userID))
    print()
    print("////////////////////////////////////////")
    print(pModel.getModel(userID))


    rItemIDsWithResponsibility:List[tuple] = [(clickedItemID, {'metoda1': 0.0, 'metoda2': 1.0, 'metoda3': 0.0})]

    eTool:AEvalTool = EToolHybrid({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS:0.1,
                                   EvalToolDHondt.ARG_LEARNING_RATE_VIEWS:0.1/500, EvalToolDHondt.ARG_VERBOSE:False})
    eTool.click(userID, rItemIDsWithResponsibility, clickedItemID, pModel, {})

    print()
    print("////////////////////////////////////////")
    print(pModel.getModel(userID))
    #print(pModel.getModelGlobal())
    #print(pModel.getModelPerson(userID))


if __name__ == '__main__':
    os.chdir("..")

    test01()
#    test02()