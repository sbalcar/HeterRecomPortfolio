#!/usr/bin/python3

from typing import List
from typing import Dict #class

from evaluationTool.aEvalTool import AEvalTool  # class

from pandas.core.frame import DataFrame  # class
from pandas.core.series import Series #class

import numpy as np


class EToolDoNothing(AEvalTool):

    def __init__(self, argumentsDict:Dict[str,object]):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

    def click(self, rItemIDsWithResponsibility:List, clickedItemID:int, portfolioModel:DataFrame, argumentsDict:Dict[str,object]):
        if type(rItemIDsWithResponsibility) is not Series and \
                type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type Series / list.")
        if type(clickedItemID) is not int and type(clickedItemID) is not np.int64:
            raise ValueError("Argument clickedItemID isn't type int.")
        if type(portfolioModel) is not DataFrame:
            raise ValueError("Argument portfolioModel isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")


        print("HOP")
        print("clickedItemID: " + str(clickedItemID))

    def displayed(self, rItemIDsWithResponsibility:List, portfolioModel:DataFrame, argumentsDict:Dict[str,object]):
        if type(rItemIDsWithResponsibility) is not Series and \
                type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type Series / list.")
        if type(portfolioModel) is not DataFrame:
            raise ValueError("Argument portfolioModel isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        pass