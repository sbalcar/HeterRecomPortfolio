#!/usr/bin/python3
import random

import numpy as np
import pandas as pd

from numpy.random import beta
from typing import List
from typing import Dict #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from aggregation.aAggregation import AAgregation #class
from aggregation.tools.responsibilityDHont import countDHontResponsibility #function
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class

from history.aHistory import AHistory #class
from abc import ABC, abstractmethod

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class


class AggrRandomKfromN(AAgregation):
    ARG_MAIN_METHOD:str = "mainMethod"
    ARG_MAX_REC_SIZE:str = "maxRecSize"
    
    def __init__(self, history:AHistory, argumentsDict:Dict[str,object]):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")
        np.random.seed(42)
        self._history = history
        self.lastUsersRecommender = {}
        self.mainMethod =  argumentsDict[self.ARG_MAIN_METHOD]
        self.maxRecSize =  argumentsDict[self.ARG_MAX_REC_SIZE]
        
    def update(self, ratingsUpdateDF:DataFrame, argumentsDict:Dict[str,object]):
        pass


    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # modelDF:pd.DataFrame[methodID:String, r:int, n:int, alpha0:int, beta0:int], numberOfItems:int
    def run(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems, argumentsDict:Dict[str,object]={}):
        if type(methodsResultDict) is not dict:
            raise ValueError("Argument methodsResultDict isn't type dict.")
        if type(modelDF) is not DataFrame:
            raise ValueError("Argument modelDF isn't type DataFrame.")
        if type(numberOfItems) is not int:
            raise ValueError("Argument numberOfItems isn't type int.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        result:List[tuple] = self.runWithResponsibility(methodsResultDict, modelDF, numberOfItems)

        return list(map(lambda itemWithResponsibilityI: itemWithResponsibilityI[0], result))


    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # modelDF:DataFrame<(methodID:str, votes:int)>, numberOfItems:int
    def runWithResponsibility(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems:int, argumentsDict:Dict[str,object]={}):
        #print("userID: " + str(userID))

        if type(methodsResultDict) is not dict:
            raise ValueError("Argument methodsResultDict isn't type dict.")
        if type(numberOfItems) is not int:
            raise ValueError("Argument numberOfItems isn't type int.")
        if numberOfItems < 0:
            raise ValueError("Argument numberOfItems can't contain negative value.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        methodsResultDictI:dict = methodsResultDict
        methodsParamsDFI:DataFrame = modelDF
        
        recommenderID = self.mainMethod
        results = methodsResultDictI[recommenderID] #pd.Series
        results.sort_values(ascending=False,inplace=True)
                    
        if self.lastUsersRecommender.get(userID) is None:
            print("Main vetev")
            #repeated recommendation, use different recommender  
            resList = list(results.iteritems())[:numberOfItems]
                   
        else:
            print("Else vetev")
            results = results.iloc[0:self.maxRecSize]
            randResults = results.sample(numberOfItems)
            resList = list(randResults.iteritems())
        
        print(recommenderID)
            
        self.lastUsersRecommender[userID] = 1
        
        
        return resList

