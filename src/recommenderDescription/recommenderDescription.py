#!/usr/bin/python3


class RecommenderDescription:

    # recommenderClass:Class, argumentsDict:dict
    def __init__(self, recommenderClass, argumentsDict:dict):

        if type(argumentsDict) is not dict :
            raise ValueError("Argument argumentsDict isn't type dict.")

        self._recommenderClass = recommenderClass
        self._argumentsDict:dict = argumentsDict

    def getArguments(self):
        return self._argumentsDict

    def exportRecommender(self):
        return self._recommenderClass(self._argumentsDict)

