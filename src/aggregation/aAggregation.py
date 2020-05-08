#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

from recommendation.resultsOfRecommendations import ResultsOfRecommendations #class

class AAgregation:

    # userDef:DataFrame<(methodID:str, votes:int)>
    def run(self, resultsOfRecommendations:ResultsOfRecommendations, userDef:DataFrame, numberOfItems:int=20):
        pass

    # userDef:DataFrame<(methodID:str, votes:int)>
    def runWithResponsibility(self, methodsResultDict, userDef:DataFrame, topK=20):
        pass
    # return list<(itemID:int, Series<(rating:int, methodID:str)>)>