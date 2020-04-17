#!/usr/bin/python3

from configuration.arguments import Arguments #class

class AggregationDescription:

    # aggregationClass:Class, arguments:Arguments
    def __init__(self, aggregationClass, arguments:Arguments):

        if type(arguments) is not Arguments :
           raise ValueError("Argument arguments is not type Arguments")

        self._aggregationClass = aggregationClass;
        self._arguments:Arguments = arguments

    def exportAggregation(self):
        return self._aggregationClass(self._arguments);
