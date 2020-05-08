#!/usr/bin/python3


class AggregationDescription:

    # aggregationClass:Class, argumentsDict:dict
    def __init__(self, aggregationClass, argumentsDict:dict):
        if aggregationClass is None :
           raise ValueError("Argument aggregationClass is None")
        if type(argumentsDict) is not dict :
           raise ValueError("Argument argumentsDict isn't type dict")

        self._aggregationClass = aggregationClass;
        self._argumentsDict:dict = argumentsDict

    def exportAggregation(self):
        return self._aggregationClass(self._argumentsDict);
