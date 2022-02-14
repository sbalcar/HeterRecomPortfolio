#!/usr/bin/python3

from typing import List
from typing import Dict #class

from evaluationTool.aEvalTool import AEvalTool  # class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class

from pandas.core.frame import DataFrame  # class
from pandas.core.series import Series #class

import numpy as np


class EToolDHondtPersonal(AEvalTool):

    def __init__(self, argumentsDict:Dict[str,object]):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        self.et:AEvalTool = EvalToolDHondt(argumentsDict)



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

        portfolioModelPer:DataFrame = portfolioModel.getModel(userID)

        self.et.click(userID, rItemIDsWithResponsibility, clickedItemID, portfolioModelPer, argumentsDict)

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

        portfolioModelPer:DataFrame = portfolioModel.getModel(userID)

        self.et.displayed(userID, rItemIDsWithResponsibility, portfolioModelPer, argumentsDict)
