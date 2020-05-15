#!/usr/bin/python3
from history.aHistory import AHistory #class
from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class

class AggregationDescription:

    # aggregationClass:Class, argumentsDict:dict
    def __init__(self, aggregationClass, argumentsDict:dict):
        if aggregationClass is None :
           raise ValueError("Argument aggregationClass is None")
        if type(argumentsDict) is not dict :
           raise ValueError("Argument argumentsDict isn't type dict")

        self._aggregationClass = aggregationClass;
        self._argumentsDict:dict = argumentsDict

    def exportAggregation(self, uBehaviourDesc:UserBehaviourDescription, history:AHistory):
        if type(uBehaviourDesc) is not UserBehaviourDescription :
           raise ValueError("Argument uBehaviourDesc isn't type UserBehaviourDescription")
        if not isinstance(history, AHistory):
           raise ValueError("Argument history isn't type AHistory")

        return self._aggregationClass(uBehaviourDesc, history, self._argumentsDict);
