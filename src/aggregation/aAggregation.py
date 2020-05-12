#!/usr/bin/python3

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

class AAgregation:

    # userDef:DataFrame<(methodID:str, votes:int)>
    def runWithResponsibility(self, methodsResultDict, userDef:DataFrame, topK=20):
        assert False, "this needs to be overridden"
    # return list<(itemID:int, Series<(rating:int, methodID:str)>)>

    @staticmethod
    def extractItemIDsFromRecommendation(recommendation:Series):
        assert False, "this needs to be overridden"
