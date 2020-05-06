#!/usr/bin/python3


class AggregationDescription:

    # aggregationClass:Class, argumentsDict:dict
    def __init__(self, aggregationClass, argumentsDict:dict):

        if type(argumentsDict) is not dict :
           raise ValueError("Argument arguments is not type Arguments")

        self._aggregationClass = aggregationClass;
        self._argumentsDict:dict = argumentsDict

    def exportAggregation(self):
        return self._aggregationClass(self._argumentsDict);
