#!/usr/bin/python3

from typing import List
from typing import Dict #class

from evaluationTool.aEvalTool import AEvalTool  # class

from pandas.core.frame import DataFrame  # class
from pandas.core.series import Series #class

from evaluationTool.evalToolDHondt import EvalToolDHondt #class
from evaluationTool.evalToolDHondtPersonal import EvalToolDHondtPersonal #class

from simulation.aSequentialSimulation import ASequentialSimulation #class
import numpy as np


class EToolHybrid(AEvalTool):

    def __init__(self, evalToolMGlobal:AEvalTool, evalToolMPerson:AEvalTool, argumentsDict:Dict[str,object]):
        if not isinstance(evalToolMGlobal, AEvalTool):
            raise ValueError("Argument evalToolMGlobal isn't type AEvalTool.")
        if not isinstance(evalToolMPerson, AEvalTool):
            raise ValueError("Argument evalToolMPerson isn't type AEvalTool.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        self._evalToolMGlobal:EvalToolDHondt = evalToolMGlobal
        self._evalToolMPerson:EvalToolDHondt = evalToolMPerson

    def click(self, userID:int, rItemIDsWithResponsibility:List, clickedItemID:int, portfolioModel:DataFrame, argumentsDict:Dict[str,object]):
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(rItemIDsWithResponsibility) is not Series and \
                type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type Series / list.")
        #if type(clickedItemID) is not int and type(clickedItemID) is not np.int64:
        #    raise ValueError("Argument clickedItemID isn't type int.")
        if not isinstance(portfolioModel, DataFrame):
            raise ValueError("Argument portfolioModel isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        status:float = argumentsDict[ASequentialSimulation.ARG_STATUS]

        mGlobal:DataFrame = portfolioModel.getModelGlobal()
        mPerson:DataFrame = portfolioModel.getModelPerson(userID)

        portfolioModel.getModelPersonAllUsers().incrementClick(userID)


        self._evalToolMGlobal.click(userID, rItemIDsWithResponsibility, clickedItemID, mGlobal, argumentsDict)
        self._evalToolMPerson.click(userID, rItemIDsWithResponsibility, clickedItemID, mPerson, argumentsDict)

        print("HOP")
        print("clickedItemID: " + str(clickedItemID))

    def displayed(self, userID:int, rItemIDsWithResponsibility:List, portfolioModel:DataFrame, argumentsDict:Dict[str,object]):
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(rItemIDsWithResponsibility) is not Series and \
                type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type Series / list.")
        if not isinstance(portfolioModel, DataFrame):
            raise ValueError("Argument portfolioModel isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        status:float = argumentsDict[ASequentialSimulation.ARG_STATUS]

        mGlobal:DataFrame = portfolioModel.getModelGlobal()
        mPerson:DataFrame = portfolioModel.getModelPerson(userID)

        self._evalToolMGlobal.displayed(userID, rItemIDsWithResponsibility, mGlobal, argumentsDict)
        self._evalToolMPerson.displayed(userID, rItemIDsWithResponsibility, mPerson, argumentsDict)